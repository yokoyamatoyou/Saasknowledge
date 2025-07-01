# Saasknowledge

A simple Streamlit prototype for uploading files, creating a knowledge base and chatting with it.

* Read the [full setup guide](knowledgeplus_design-main/README.md) for details.
* Set the OpenAI key before starting:

  ```bash
  export OPENAI_API_KEY=your_api_key
  ```

* Launch the app with:

  ```bash
./knowledgeplus_design-main/run_app.sh
```

If you require model files that are normally fetched from HuggingFace, be aware
that this environment blocks direct downloads from that service. Retrieve the
same files from a mirror or copy them locally before launching the app.

