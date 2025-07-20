#!/bin/bash
# Install dependencies required for running the test suite
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Install the lighter core packages first so the basic modules are available
pip install -r "$REPO_ROOT/knowledgeplus_design-main/requirements-light.txt"
# Then install the additional libraries needed only for tests
pip install -r "$REPO_ROOT/knowledgeplus_design-main/requirements-tests.txt"
