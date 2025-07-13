# MM KB Builder

This component provides a Streamlit interface for creating a multimodal knowledge base.
Uploaded images and text are processed into embeddings so that the chatbot can search them later.

## Usage

1. Install dependencies from the repository root:
   ```bash
   scripts/install_light.sh
   scripts/install_extra.sh  # heavy libraries like torch
   ```
2. Set your `OPENAI_API_KEY` environment variable.
3. Launch the builder:
   ```bash
   streamlit run app.py
   ```

Processed data will be stored under `knowledge_base/<kb_name>/` where
`<kb_name>` defaults to the value of `DEFAULT_KB_NAME` defined in
`config.py`.

Sample metadata previously lived under `mm_kb_builder/multimodal_data`. With the unified layout place any example files under `knowledge_base/<kb_name>/` before running the app. Images remain untracked by git.
