# Saasknowledge

A simple Streamlit prototype for uploading files, creating a knowledge base and chatting with it. Some dependencies such as `torch` and `transformers` are quite heavy. If you only need the core features you can install a lighter set of packages first and add the extras later.

* Read the [full setup guide](knowledgeplus_design-main/README.md) for details.
* Install packages with `pip install -r knowledgeplus_design-main/requirements-light.txt` for the basics.
* Add advanced features later with `pip install -r knowledgeplus_design-main/requirements-extra.txt`.
* Copy `knowledgeplus_design-main/.env.example` to `.env` and set the OpenAI key before starting:

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

