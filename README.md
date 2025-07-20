# Saasknowledge

A simple Streamlit prototype for uploading files, creating a knowledge base and chatting with it. Some dependencies such as `torch` and `transformers` are quite heavy. If you only need the core features you can install a lighter set of packages first and add the extras later.
To keep pull requests stable when using Codex, install dependencies in two steps: run `scripts/install_light.sh` first and then `scripts/install_extra.sh`.

* Read the [full setup guide](knowledgeplus_design-main/README.md) for details.
* Install packages with `pip install -r knowledgeplus_design-main/requirements-light.txt` for the basics.
* Add advanced features later with `pip install -r knowledgeplus_design-main/requirements-extra.txt`.
* Or run the helper scripts: `scripts/install_light.sh` first and `scripts/install_extra.sh` when you need the heavy libraries.
* A combined `scripts/install_full.sh` exists but installing everything in one step may trigger network errors, so the two-step approach is recommended.
* PDF export requires the Japanese font `ipaexg.ttf` placed in the repository root. See `knowledgeplus_design-main/README.md` for details.
* Critical packages such as **PyMuPDF** (imported as `fitz`), **pytesseract**, **opencv-python**, `ezdxf` and `trimesh` are listed in `knowledgeplus_design-main/requirements.txt`. Install them all at once with:

  ```bash
  pip install -r knowledgeplus_design-main/requirements.txt
  ```

The FAQ generator relies on `requests` and `beautifulsoup4` to fetch and parse
web pages. These libraries are included in `requirements-light.txt`.

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

The application stores data under `knowledge_base/` at the repository root by default. You can override this location by setting `KNOWLEDGE_BASE_DIR` before running the app:

  ```bash
  export KNOWLEDGE_BASE_DIR=/path/to/storage
  ```

You can similarly override where chat histories are stored by setting
`CHAT_HISTORY_DIR`:

  ```bash
  export CHAT_HISTORY_DIR=/path/to/chat_history
  ```

Set `PERSONA_DIR` to control where persona definitions are saved:

```bash
export PERSONA_DIR=/path/to/personalities
```

The sidebar defaults to expanded. Use the `<<` or `>>` button to collapse or
expand it. Set `SIDEBAR_DEFAULT_VISIBLE=false` before launching if you prefer
the sidebar hidden at startup:

  ```bash
  export SIDEBAR_DEFAULT_VISIBLE=false
  ```

The last sidebar state is saved to `chat_history/sidebar_state.json`. If the
sidebar keeps reappearing, delete this file or start the app with
`SIDEBAR_DEFAULT_VISIBLE=false` to reset it. The default value is `true` so
omitting the variable keeps the sidebar visible when the app starts.

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

Embedding requests are sent in batches of 10 by default. Adjust the batch size
using `EMBEDDING_BATCH_SIZE` if you need to tune API throughput:

  ```bash
  export EMBEDDING_BATCH_SIZE=5
  ```

Automatic chat titles use `gpt-3.5-turbo` by default. Set `TITLE_MODEL` to
override the model used for title generation:

  ```bash
  export TITLE_MODEL=gpt-4o-mini
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

Set up git hooks so code style checks run automatically. The hooks run
`black`, `ruff` and `isort` to keep formatting and imports consistent:

```bash
pip install pre-commit
pre-commit install
```

## Testing

Before running `pytest`, install the libraries required by the test suite. The
quickest way is to run the dedicated helper script (the CI workflow calls this
script as well):

```bash
scripts/install_tests.sh
```

This installs `numpy`, `requests`, `PyMuPDF`, **Pillow** and other dependencies
the tests expect. If some tests need heavier packages (for example those that
rely on `torch` or `transformers`), install them with:

```bash
scripts/install_extra.sh
```

After installing the necessary packages, execute the tests from the repository
root:

```bash
pytest -q
```

The suite now includes checks that malformed chunk files are ignored by
`HybridSearchEngine` and that embedding requests handle OpenAI timeouts
gracefully. A PNG upload test dynamically generates a tiny image in memory to

## Multimodal upload workflow

1. Choose **個別処理** or **まとめて処理** in the management tab.
2. Drag & drop documents and images into the uploader.
3. Click **選択したファイルの処理を開始** to process everything with a single button.
   The search index refreshes automatically when batch mode completes.
4. If proposed chunks contain conflicting rules, a warning appears so you can adjust the metadata or cancel before saving.

## Thumbnail grid

## Thumbnail grid

The management interface shows uploaded items in a 3×3 thumbnail grid. Use
`display_thumbnail_grid()` to edit titles and tags for each item after upload.
See [docs/integration_plan.md](knowledgeplus_design-main/docs/integration_plan.md)
for the design overview and additional details.

For a deeper look at how the project evolves, check the same
[docs/integration_plan.md](knowledgeplus_design-main/docs/integration_plan.md)
file. It documents the phased design notes and lists key repository
guidelines to keep contributions consistent.

## FAQ generation

Run `generate_faq.py` to create frequently asked questions for a knowledge base.
You can pass raw text or a URL via the `--source` option:

```bash
python knowledgeplus_design-main/generate_faq.py my_kb --source "https://example.com"
```

The script fetches the URL, extracts text with Beautiful Soup and then calls
GPT‑4.1 mini. Temperature starts at `0` and increases by `0.01` every five
questions up to a maximum of `0.8`.


## Metrics report

Use `show_metrics.py` to display search performance metrics collected during the current session.

```bash
python show_metrics.py
```

The script prints a JSON summary including average response time and recommendations for improving search quality.
