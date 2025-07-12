import os

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072

# Default knowledge base name can be overridden via the environment
# to simplify customization in different deployments.
DEFAULT_KB_NAME = os.getenv("DEFAULT_KB_NAME", "default_kb")

# Model used for automatic conversation title generation.
# The default is GPT-3.5 but it can be overridden via the TITLE_MODEL
# environment variable so deployments can switch to a different provider
# such as Gemini.
TITLE_GENERATION_MODEL = os.getenv("TITLE_MODEL", "gpt-3.5-turbo")

# Default weights for the hybrid search engine. They can be overridden via
# HYBRID_VECTOR_WEIGHT and HYBRID_BM25_WEIGHT environment variables.
HYBRID_VECTOR_WEIGHT = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.7"))
HYBRID_BM25_WEIGHT = float(os.getenv("HYBRID_BM25_WEIGHT", "0.3"))
