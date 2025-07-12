class OpenAIClientError(RuntimeError):
    """Raised when the OpenAI client cannot be initialized."""

class GPTGenerationError(RuntimeError):
    """Raised when GPT response generation fails."""

class KnowledgeBaseError(RuntimeError):
    """Raised for errors accessing or processing the knowledge base."""
