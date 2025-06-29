# MM KB Builder

This component provides a Streamlit interface for creating a multimodal knowledge base.
Uploaded images and text are processed into embeddings so that the chatbot can search them later.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your `OPENAI_API_KEY` environment variable.
3. Launch the builder:
   ```bash
   streamlit run app.py
   ```

Processed data will be stored under `knowledge_base/<kb_name>/` where
`<kb_name>` defaults to the value of `DEFAULT_KB_NAME` defined in
`config.py`.

Sample metadata for a demo knowledge base is under `multimodal_data/`. The accompanying images are not tracked in git; copy any example images into `multimodal_data/<kb_name>/images/` before running the app.
