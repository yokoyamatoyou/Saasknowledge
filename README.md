# Saasknowledge

A simple Streamlit prototype for uploading files, creating a knowledge base and chatting with it. Some dependencies such as `torch` and `transformers` are quite heavy. If you only need the core features you can install a lighter set of packages first and add the extras later.
To keep pull requests stable when using Codex, install dependencies in two steps: run `scripts/install_light.sh` first and then `scripts/install_extra.sh`.

* Read the [full setup guide](knowledgeplus_design-main/README.md) for details.
* Install packages with `pip install -r knowledgeplus_design-main/requirements-light.txt` for the basics.
* Add advanced features later with `pip install -r knowledgeplus_design-main/requirements-extra.txt`.
* Or run the helper scripts: `scripts/install_light.sh` first and `scripts/install_extra.sh` when you need the heavy libraries.
* A combined `scripts/install_full.sh` exists but installing everything in one step may trigger network errors, so the two-step approach is recommended.

### Handling large dependencies

Libraries such as **torch** and **transformers** can be several hundred
megabytes. Installing them separately helps avoid network timeouts. Run the
light requirements first and then fetch the extras only when needed:

```bash
scripts/install_light.sh
scripts/install_extra.sh  # add the heavy packages later
```
When using Codex for automated contributions, run the light script first and
install the extras separately. This avoids network errors that can lead to pull
request failures.

## Codex tips

Automated commits must keep the repository clean to prevent pull request
failures. Run `pytest -q` before submitting changes and install dependencies in
two stages so heavy packages download separately. When running `run_app.sh` the
process stays active waiting for connections—this is normal and should not be
treated as an error.
* Copy `knowledgeplus_design-main/.env.example` to `.env` and set the OpenAI key before starting:

  ```bash
  export OPENAI_API_KEY=your_api_key
  ```
  - Optionally edit `KNOWLEDGE_BASE_DIR` and `CHAT_HISTORY_DIR` in `.env` to change where data is stored.

You can also override the default `knowledge_base/` directory location by setting `KNOWLEDGE_BASE_DIR` before running the app:

  ```bash
  export KNOWLEDGE_BASE_DIR=/path/to/storage
  ```

You can similarly override where chat histories are stored by setting
`CHAT_HISTORY_DIR`:

  ```bash
  export CHAT_HISTORY_DIR=/path/to/chat_history
  ```

The sidebar defaults to collapsed.  Set `SIDEBAR_DEFAULT_VISIBLE=true` before
launching to keep it expanded when the app starts:

  ```bash
  export SIDEBAR_DEFAULT_VISIBLE=true
  ```

You can override the default knowledge base name by setting
`DEFAULT_KB_NAME` before running the app:

  ```bash
  export DEFAULT_KB_NAME=my_kb
  ```

You can tune the default blend between vector similarity and keyword search by
setting `HYBRID_VECTOR_WEIGHT` and `HYBRID_BM25_WEIGHT` before launching the
app.  The values should sum to 1.0:

  ```bash
  export HYBRID_VECTOR_WEIGHT=0.7
  export HYBRID_BM25_WEIGHT=0.3
  ```

These variables are evaluated when helper modules such as
`upload_utils` and `chat_history_utils` are imported. Set them before
running the application or tests so that all uploads and chat logs are
stored under the specified directories.

* Launch the app with:

  ```bash
  ./run_app.sh          # macOS/Linux
  run_app.bat           # Windows
  # or run Streamlit directly from the repository root
  streamlit run app.py
  ```
  The helper scripts source `.env` if it exists so optional settings like
  `KNOWLEDGE_BASE_DIR` are applied. They then verify that `OPENAI_API_KEY` is
  set, exiting early if the key is missing to avoid confusing startup errors.

When the server starts it will remain active to serve the web UI. This waiting
state is expected and should not be treated as a failure when Codex prepares a
pull request.

If you require model files that are normally fetched from HuggingFace, be aware
that this environment blocks direct downloads from that service. Retrieve the
same files from a mirror or copy them locally before launching the app.

## Pre-commit

Set up git hooks so code style checks run automatically:

```bash
pip install pre-commit
pre-commit install
```

## Testing

Before running `pytest`, install the minimal dependencies required for the test
suite. The helper script below pulls in packages such as **Pillow**, which the
tests rely on:

```bash
scripts/install_light.sh
```

If certain tests depend on heavier libraries, you can also run:

```bash
scripts/install_extra.sh
```

After installing the necessary packages, execute the tests from the repository
root:

```bash
pytest -q
```

## Thumbnail grid

The management interface shows uploaded items in a 3×3 thumbnail grid. Use
`display_thumbnail_grid()` to edit titles and tags for each item after upload.
See [docs/integration_plan.md](knowledgeplus_design-main/docs/integration_plan.md)
for the design overview and additional details.

For a deeper look at how the project evolves, check the same
[docs/integration_plan.md](knowledgeplus_design-main/docs/integration_plan.md)
file. It documents the phased design notes and lists key repository
guidelines to keep contributions consistent.

