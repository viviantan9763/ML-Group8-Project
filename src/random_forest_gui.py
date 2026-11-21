"""Inspect saved CV evidence or watch a new random forest search."""

import ast
import csv
import json
import math
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

from sklearn.model_selection import ParameterSampler

from src.random_forest import SEARCH_SPACE


ROOT = Path(__file__).resolve().parents[1]
BG = "#eef4f6"
INK = "#183146"
TEAL = "#007d79"
MUTED = "#506676"
LINE = "#d4e2e8"
FOLD_END = re.compile(r"^\[CV (\d+)/\d+; (\d+)/\d+\] END .*? f1: \(test=([\d.]+)\)")
METRICS = {
    "f1": "Default-class F1",
    "precision": "Precision",
    "recall": "Recall",
    "average_precision": "Average precision",
    "roc_auc": "ROC AUC",
    "balanced_accuracy": "Balanced accuracy",
}


class TuningWindow:
    def __init__(self, root):
        self.root = root
        self.events = queue.Queue()
        self.process = None
        self.replay = None
        self.visible = 0
        self.candidates = []
        self.summary = {}
        self.fits = 0
        self.count = tk.StringVar(value="48")
        self.jobs = tk.StringVar(value="4")
        self.status = tk.StringVar()
        self.details = tk.StringVar()
        self.comparison = tk.StringVar()

        root.title("Random Forest | Tuning Evidence")
        root.geometry("1120x800")
        root.minsize(930, 720)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10), background=BG, foreground=INK)
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 26))
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Treeview", rowheight=25, fieldbackground="white", background="white")
        style.configure("Horizontal.TProgressbar", background=TEAL, troughcolor=LINE)

        frame = ttk.Frame(root, padding=(22, 16))
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Random forest tuning", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            frame,
            text="Credit card default   /   Random search   /   3 stratified folds   /   Training data only",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 12))

        controls = ttk.Frame(frame)
        controls.pack(fill="x")
        for label, variable, maximum in (("Candidates", self.count, 500), ("Workers", self.jobs, 32)):
            ttk.Label(controls, text=label).pack(side="left", padx=(0, 6))
            ttk.Spinbox(controls, textvariable=variable, from_=1, to=maximum, width=5).pack(
                side="left", padx=(0, 14)
            )
        self.run_button = ttk.Button(controls, text="Run search", command=self.start_search)
        self.run_button.pack(side="left", padx=(0, 8))
        self.replay_button = ttk.Button(controls, text="Replay saved scores", command=self.start_replay)
        self.replay_button.pack(side="left")
        ttk.Button(controls, text="Search space", command=self.show_space).pack(side="right")
        self.progress = ttk.Progressbar(frame, mode="determinate")
        self.progress.pack(fill="x", pady=(12, 4))
        ttk.Label(frame, textvariable=self.status, style="Muted.TLabel").pack(anchor="w")

        tabs = ttk.Notebook(frame)
        tabs.pack(fill="both", expand=True, pady=(10, 0))
        evidence = ttk.Frame(tabs, padding=12)
        activity = ttk.Frame(tabs, padding=12)
        tabs.add(evidence, text="  CV evidence  ")
        tabs.add(activity, text="  Live fit log  ")
        self.tabs, self.activity, self.evidence = tabs, activity, evidence
        ttk.Label(evidence, textvariable=self.comparison, font=("Segoe UI Semibold", 12)).pack(anchor="w")
        self.chart = tk.Canvas(evidence, height=174, background="white", highlightthickness=0)
        self.chart.pack(fill="x", pady=8)
        self.chart.bind("<Configure>", lambda _: self.draw_chart())

        ttk.Label(evidence, text="Ranked candidates · select a row to inspect it").pack(anchor="w")
        grid = ttk.Frame(evidence)
        grid.pack(fill="both", expand=True, pady=(4, 6))
        columns = ("rank", "sample", "mean", "sd", "precision", "recall", "auc")
        self.table = ttk.Treeview(grid, columns=columns, show="headings", height=4, selectmode="browse")
        for name, title in zip(columns, ("Rank", "Draw", "Mean F1", "F1 SD", "Precision", "Recall", "ROC AUC")):
            self.table.heading(name, text=title)
            self.table.column(name, width=90, minwidth=60, anchor="center")
        scroll = ttk.Scrollbar(grid, command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.table.pack(fill="both", expand=True)
        self.table.bind("<<TreeviewSelect>>", self.inspect_candidate)
        ttk.Label(evidence, textvariable=self.details, justify="left").pack(anchor="w")

        self.log = tk.Text(activity, wrap="word", background="white", foreground=INK,
                           font=("Consolas", 11), state="disabled")
        log_scroll = ttk.Scrollbar(activity, command=self.log.yview)
        self.log.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)

        self.load_results(ROOT / "results")
        root.after(80, self.poll_events)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def load_results(self, folder):
        summary_path = folder / "random_forest_cv.json"
        csv_path = folder / "random_forest_search.csv"
        if not (summary_path.exists() and csv_path.exists()):
            self.status.set("Run a search to generate tuning evidence.")
            self.replay_button.state(["disabled"])
            return
        self.summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if "draw" in rows[0]:
            self.candidates = sorted(rows, key=lambda row: int(row["draw"]))
        else:
            sampled = ParameterSampler(
                self.summary.get("search_space", SEARCH_SPACE), len(rows),
                random_state=self.summary["random_state"],
            )
            by_params = {tuple(sorted(ast.literal_eval(row["params"]).items())): row for row in rows}
            self.candidates = [by_params[tuple(sorted(params.items()))] for params in sampled]
        self.visible = len(self.candidates)
        self.replay_button.state(["!disabled"])
        self.progress.configure(maximum=len(rows), value=len(rows))
        self.status.set(
            f"{len(rows)} candidates × {self.summary['cv_folds']} folds  |  "
            f"{self.summary['training_rows']:,} training rows  |  Seed {self.summary['random_state']}"
        )
        self.refresh_evidence()

    def refresh_evidence(self):
        baseline = self.summary["baseline"]["f1"]["mean"]
        shown = self.candidates[:self.visible]
        best = max((float(row["mean_test_f1"]) for row in shown), default=None)
        text = f"Baseline F1  {baseline:.4f}"
        if best is not None:
            text += f"       Best candidate F1  {best:.4f}       Change  {(best - baseline) * 100:+.2f} pp"
        self.comparison.set(text)
        self.table.delete(*self.table.get_children())
        ranked = sorted(enumerate(shown), key=lambda item: -float(item[1]["mean_test_f1"]))
        for rank, (i, row) in enumerate(ranked, 1):
            values = [f"{float(row[key]):.4f}" for key in (
                "mean_test_f1", "std_test_f1", "mean_test_precision", "mean_test_recall", "mean_test_roc_auc",
            )]
            self.table.insert("", "end", iid=str(i), values=(rank, i + 1, *values))
        if ranked:
            self.table.selection_set(str(ranked[0][0]))
            self.inspect_candidate()
        else:
            self.details.set("")
        self.draw_chart()

    def inspect_candidate(self, _event=None):
        selection = self.table.selection()
        if not selection:
            return
        i = int(selection[0])
        row = self.candidates[i]
        params = ast.literal_eval(row["params"])
        settings = [f"{key.removeprefix('model__')} = {value}" for key, value in params.items()]
        folds = "   /   ".join(f"{float(row[f'split{j}_test_f1']):.4f}" for j in range(self.summary["cv_folds"]))
        self.details.set(
            f"Draw {i + 1}   |   Fold F1   {folds}\n"
            + "   ·   ".join(settings[:4]) + "\n" + "   ·   ".join(settings[4:])
        )

    def draw_chart(self):
        canvas = self.chart
        canvas.delete("all")
        if not self.candidates:
            return
        scores = [float(row["mean_test_f1"]) for row in self.candidates]
        baseline = self.summary["baseline"]["f1"]["mean"]
        low = max(0, math.floor(min(scores + [baseline]) * 20) / 20 - 0.01)
        high = min(1, math.ceil(max(scores + [baseline]) * 20) / 20 + 0.01)
        left, right = 55, max(100, canvas.winfo_width() - 18)
        top, bottom = 27, 142

        def y(score):
            return bottom - (score - low) / (high - low) * (bottom - top)

        def x(index):
            return left + (right - left) * (index + 0.5) / len(scores)

        canvas.create_text(left, 12, anchor="w", text="Mean F1 by draw   •   Best so far   /   Baseline",
                           fill=INK, font=("Segoe UI", 9))
        for tick in range(5):
            value = low + (high - low) * tick / 4
            canvas.create_line(left, y(value), right, y(value), fill=LINE)
            canvas.create_text(left - 8, y(value), text=f"{value:.2f}", fill=MUTED, anchor="e")
        canvas.create_line(left, y(baseline), right, y(baseline), fill=MUTED, dash=(4, 3))
        best = 0
        line = []
        for index, score in enumerate(scores[:self.visible]):
            pos, level = x(index), y(score)
            canvas.create_oval(pos - 2, level - 2, pos + 2, level + 2, fill="#537f9b", outline="")
            if line:
                line.extend((pos, y(best)))
            best = max(best, score)
            line.extend((pos, y(best)))
        if len(line) >= 4:
            canvas.create_line(*line, fill=TEAL, width=2)
        canvas.create_text(left, 159, anchor="w", text="First draw", fill=MUTED)
        canvas.create_text(right, 159, anchor="e", text=f"Draw {len(scores)}", fill=MUTED)

    def show_space(self):
        space = self.summary.get("search_space", SEARCH_SPACE)
        lines = [f"{key.removeprefix('model__')}\n  {', '.join(map(str, values))}" for key, values in space.items()]
        metrics = ""
        if self.summary:
            metrics = "\n\nCV means (baseline → best search candidate)\n" + "\n".join(
                f"{label}    {self.summary['baseline'][key]['mean']:.4f} → {self.summary['tuned'][key]['mean']:.4f}"
                for key, label in METRICS.items()
            )
        messagebox.showinfo("Search space and CV comparison", "\n".join(lines) + metrics, parent=self.root)

    def start_replay(self):
        if self.replay is not None:
            self.root.after_cancel(self.replay)
        self.tabs.select(self.evidence)
        self.visible = 0
        self.progress.configure(maximum=len(self.candidates), value=0)
        self.replay_step()

    def replay_step(self):
        self.visible += 1
        self.progress.configure(value=self.visible)
        self.status.set(f"Saved run replay  |  Draw {self.visible} of {len(self.candidates)}")
        self.refresh_evidence()
        self.replay = (
            self.root.after(180, self.replay_step) if self.visible < len(self.candidates) else None
        )

    def start_search(self):
        try:
            count, jobs = int(self.count.get()), int(self.jobs.get())
            if not (1 <= count <= 500 and 1 <= jobs <= 32):
                raise ValueError
        except ValueError:
            messagebox.showerror("Check search settings", "Enter 1–500 candidates and 1–32 workers.", parent=self.root)
            return
        if self.replay is not None:
            self.root.after_cancel(self.replay)
            self.replay = None
        folder = ROOT / "results" / "rf_gui_runs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        command = [
            sys.executable, "-u", "-m", "src.random_forest", "--live-progress",
            "--n-iter", str(count), "--jobs", str(jobs), "--output-dir", str(folder),
            "--model-path", str(folder / "random_forest.joblib"),
        ]
        try:
            self.process = subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as error:
            messagebox.showerror("Could not start search", str(error), parent=self.root)
            return
        self.run_button.state(["disabled"])
        self.replay_button.state(["disabled"])
        self.tabs.select(self.activity)
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.fits = 0
        self.progress.configure(maximum=count * 3, value=0)
        self.status.set("Starting baseline CV on the training split…")
        threading.Thread(target=self.read_process, args=(self.process, folder), daemon=True).start()

    def read_process(self, process, folder):
        with process.stdout as stream:
            for line in stream:
                self.events.put(("line", line.rstrip()))
        self.events.put(("done", (process.wait(), folder)))

    def poll_events(self):
        for _ in range(100):
            try:
                event, value = self.events.get_nowait()
            except queue.Empty:
                break
            if event == "line":
                match = FOLD_END.match(value)
                if match:
                    fold, candidate, f1 = match.groups()
                    self.fits += 1
                    self.progress.configure(value=self.fits)
                    self.status.set(f"Random search  |  {self.fits} of {self.progress['maximum']:.0f} fold fits complete")
                    value = f"Draw {candidate}   •   Fold {fold} complete   •   F1 {f1}"
                elif value.startswith("[CV") and "] START" in value:
                    continue
                elif value.startswith("Refitting"):
                    self.status.set(value)
                self.log.configure(state="normal")
                self.log.insert("end", value + "\n")
                self.log.see("end")
                self.log.configure(state="disabled")
            else:
                code, folder = value
                self.process = None
                self.run_button.state(["!disabled"])
                if code == 0:
                    self.load_results(folder)
                    self.tabs.select(self.evidence)
                else:
                    self.status.set(f"Search exited with code {code}. See the live fit log.")
                    if self.candidates:
                        self.replay_button.state(["!disabled"])
        self.root.after(80, self.poll_events)

    def close(self):
        if self.process is not None:
            if not messagebox.askyesno("End search?", "Close the window and end the current run?", parent=self.root):
                return
            self.process.terminate()
        self.root.destroy()


def main():
    root = tk.Tk()
    TuningWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
