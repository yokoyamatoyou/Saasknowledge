#!/bin/bash
# Install dependencies required for running the test suite
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
pip install -r "$REPO_ROOT/knowledgeplus_design-main/requirements-tests.txt"
