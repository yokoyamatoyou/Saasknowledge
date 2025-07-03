# Saasknowledge

A simple Streamlit prototype for uploading files, creating a knowledge base and chatting with it. Some dependencies such as `torch` and `transformers` are quite heavy. If you only need the core features you can install a lighter set of packages first and add the extras later.

* Read the [full setup guide](knowledgeplus_design-main/README.md) for details.
* Install packages with `pip install -r knowledgeplus_design-main/requirements-light.txt` for the basics.
* Add advanced features later with `pip install -r knowledgeplus_design-main/requirements-extra.txt`.
* Alternatively run `scripts/install_light.sh` for the basics, `scripts/install_extra.sh` when you need the heavy libraries, or `scripts/install_full.sh` to install everything in one go.

### Handling large dependencies

Libraries such as **torch** and **transformers** can be several hundred
megabytes. Installing them separately helps avoid network timeouts. Run the
light requirements first and then fetch the extras only when needed:

```bash
scripts/install_light.sh
scripts/install_extra.sh  # add the heavy packages later
```
* Copy `knowledgeplus_design-main/.env.example` to `.env` and set the OpenAI key before starting:

  ```bash
  export OPENAI_API_KEY=your_api_key
  ```

You can also override the default `knowledge_base/` directory location by setting `KNOWLEDGE_BASE_DIR` before running the app:

  ```bash
  export KNOWLEDGE_BASE_DIR=/path/to/storage
  ```

You can similarly override where chat histories are stored by setting
`CHAT_HISTORY_DIR`:

  ```bash
  export CHAT_HISTORY_DIR=/path/to/chat_history
  ```

* Launch the app with:

  ```bash
  ./run_app.sh
  ```

If you require model files that are normally fetched from HuggingFace, be aware
that this environment blocks direct downloads from that service. Retrieve the
same files from a mirror or copy them locally before launching the app.

## Testing

Before running `pytest`, install the test dependencies:

```bash
pip install -r knowledgeplus_design-main/requirements-light.txt
# If you need optional features
pip install -r knowledgeplus_design-main/requirements-extra.txt
```

Run the automated tests after installing the light requirements. Execute from the
repository root:

```bash
pytest -q
```

