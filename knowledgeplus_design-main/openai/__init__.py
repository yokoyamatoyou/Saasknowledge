from types import SimpleNamespace

class OpenAI:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.embeddings = SimpleNamespace(create=lambda **k: SimpleNamespace(data=[SimpleNamespace(embedding=[0.0])]))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **k: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))])))
