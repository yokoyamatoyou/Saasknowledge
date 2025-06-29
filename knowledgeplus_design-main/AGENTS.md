# KnowledgePlus

This repository contains two related Streamlit applications:

- **knowledge_gpt_app** – a knowledge retrieval chatbot powered by GPT.
- **mm_kb_builder** – tools for building multimodal knowledge bases.
Example metadata lives under `mm_kb_builder/multimodal_data`, but images are omitted from version control.

Each folder includes its own `requirements.txt`. A consolidated list of
all dependencies is provided at the repository root for convenience.

## Knowledge base location

Both apps now use a single `knowledge_base/` directory at the repository root.
If you were storing data in the previous `rag_knowledge_base` or
`multimodal_data` folders, move those contents into `knowledge_base/` before
upgrading.

## Installation

Create a virtual environment and install the packages:

```bash
pip install -r requirements.txt
```

Alternatively, install dependencies for each component separately:

```bash
# For the chatbot app
pip install -r knowledge_gpt_app/requirements.txt

# For the knowledge base builder
pip install -r mm_kb_builder/requirements.txt
```


## Configuration

Set the `OPENAI_API_KEY` environment variable so the applications can access the OpenAI API:

```bash
# Windows
set OPENAI_API_KEY=your_api_key_here

# macOS/Linux
export OPENAI_API_KEY=your_api_key_here
```

The applications will fail to start if this variable is missing.

## IPAexGothic font

`knowledge_gpt_app/utils/export.py` generates PDF files and requires the
`ipaexg.ttf` font to render Japanese text.  Download the IPAexGothic font from
<https://moji.or.jp/ipafont/> and place the `ipaexg.ttf` file in the repository
root so that PDF export works correctly.

## Running the app

Launch the unified interface using the helper scripts at the repository root. These scripts run `unified_app.py` which accepts documents, images and CAD files in one place:

```bash
# Windows
run_app.bat

# macOS/Linux
./run_app.sh
```

For backwards compatibility a small wrapper script is also provided at
`knowledge_gpt_app/unified_app.py`. Running `streamlit run
knowledge_gpt_app/unified_app.py` will simply delegate to the root
`unified_app.py` so existing workflows continue to work.


## Integration Plan

See [docs/integration_plan.md](docs/integration_plan.md) for an overview of the integration phases.

### Phase 3 Notes

Processed chunks, embeddings and metadata are now stored under
`knowledge_base/<kb_name>` using `save_processed_data()`. The metadata JSON
includes these paths so that other components can access the original files.
Whenever a chunk is stored the chatbot search index is refreshed automatically.

If you upload a file with the same name but different content, a version
number is appended so the existing file is preserved. To rebuild the search
index for an existing knowledge base you can run:

```bash
python reindex_kb.py <kb_name>
```

This reloads all chunks from disk and regenerates the BM25 index.

### Phase 5 Notes

Minor UI fixes were made in the unified app. A common Intel-themed style is now
applied on startup and file processing is wrapped in `st.spinner` so progress is
clear. A small `torch.classes` workaround remains in place to avoid Streamlit
reload errors.

The upload screen offers **個別処理** or **まとめて処理**. When batch mode is
selected, the search index is refreshed only once after all files finish. Index
updates can also be set to **手動** so that `refresh_search_engine()` is called
only when triggered by the user.

In manual mode, click **検索インデックス更新** in the interface to rebuild
the search engine when needed.

`save_processed_data()` returns paths for the stored chunk, embedding, metadata,
image and original file. These are placed under
`knowledge_base/<kb_name>/{chunks|embeddings|metadata|images|files}`. If an
uploaded file already exists with different contents, a version suffix such as
`_v1` or `_v2` is appended automatically.

Optionally, FAQs can be generated immediately after processing using
`generate_faq.py`.

## Testing

Run the automated test suite with `pytest` from the repository root:

```bash
pytest -q
```

The tests cover utilities for saving uploads and rebuilding indexes. Additional
cases will be added as the integration progresses.

Sample files used by the tests live under `tests/data`.  For tests that
exercise the OpenAI API, make sure the `OPENAI_API_KEY` environment variable is
exported so the calls succeed.  These API tests can be disabled with
`-k "not openai"` when running `pytest` if desired.

## Repository Guidelines

For current development tips and repository consistency checks, see [docs/integration_plan.md](docs/integration_plan.md#additional-repository-guidance). Key points include avoiding NumPy array truth checks, preventing nested Streamlit expanders, centralizing configuration and file utilities, and increasing automated tests and linting.
