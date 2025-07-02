#!/bin/bash
# Launch the unified Streamlit interface from repository root
cd "$(dirname "$0")/knowledgeplus_design-main" || exit 1
streamlit run app.py
