#!/bin/bash
# Launch the unified Streamlit interface
cd "$(dirname "$0")" || exit 1
streamlit run unified_app.py

