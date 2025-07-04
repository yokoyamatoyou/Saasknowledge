import os

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072

# Default knowledge base name can be overridden via the environment
# to simplify customization in different deployments.
DEFAULT_KB_NAME = os.getenv("DEFAULT_KB_NAME", "default_kb")
