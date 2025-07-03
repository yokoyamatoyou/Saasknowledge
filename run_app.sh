#!/bin/bash
# Launch the unified Streamlit interface from repository root
cd "$(dirname "$0")/knowledgeplus_design-main" || exit 1

if [ -z "$OPENAI_API_KEY" ]; then
  echo "OPENAI_API_KEY environment variable not set"
  echo "Please export your key before running the app"
  exit 1
fi

streamlit run app.py
