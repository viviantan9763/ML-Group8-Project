"""Streamlit entry point for the Random Forest Builder."""

import sys
from pathlib import Path

import streamlit as st


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from m3_workspace.app.builder import render


st.set_page_config(page_title="Random Forest Builder", page_icon="🌲", layout="wide")
render()
