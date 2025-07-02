#!/bin/bash
# Install minimal and extra dependencies separately to avoid heavy downloads in one step
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
pip install -r "$REPO_ROOT/knowledgeplus_design-main/requirements-light.txt"
pip install -r "$REPO_ROOT/knowledgeplus_design-main/requirements-extra.txt"
