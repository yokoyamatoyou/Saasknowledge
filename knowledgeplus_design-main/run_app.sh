#!/bin/bash
# Launch the unified Streamlit interface
cd "$(dirname "$0")" || exit 1

# Load variables from ../.env if present so optional overrides apply
if [ -f ../.env ]; then
  set -a
  # shellcheck disable=SC1091
  . ../.env
  set +a
fi

if [ -z "$OPENAI_API_KEY" ]; then
  echo "OPENAI_API_KEY environment variable not set"
  echo "Add it to ../.env or export it before running the app"
  exit 1
fi

streamlit run app.py

