#!/bin/bash
# Install only the heavier optional dependencies
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
pip install -r "$REPO_ROOT/knowledgeplus_design-main/requirements-extra.txt"
