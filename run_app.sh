#!/bin/bash
# Launch the unified Streamlit interface from repository root
cd "$(dirname "$0")/knowledgeplus_design-main" || exit 1

# Load variables from .env if present and OPENAI_API_KEY isn't already set
if [ -z "$OPENAI_API_KEY" ] && [ -f ../.env ]; then
  set -a
  # shellcheck disable=SC1091
  . ../.env
  set +a
fi

if [ -z "$OPENAI_API_KEY" ]; then
  echo "OPENAI_API_KEY environment variable not set"
  echo "Add it to .env or export it before running the app"
  exit 1
fi

streamlit run app.py
