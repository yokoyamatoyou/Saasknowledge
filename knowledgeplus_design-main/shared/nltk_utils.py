import importlib
import logging

try:
    nltk = importlib.import_module("nltk")
    _USING_STUB = False
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal installs
    nltk = importlib.import_module("nltk_stub")
    _USING_STUB = True

logger = logging.getLogger(__name__)


def ensure_nltk_resources() -> bool:
    """Ensure required NLTK resources are available.

    When ``nltk`` is not installed, a minimal stopword list from the bundled
    stub is used and no downloads are attempted.
    """
    try:
        logger.info("Checking NLTK resources...")
        if _USING_STUB:  # pragma: no cover - simple environment
            logger.info("NLTK not available; using bundled stopwords only")
            return True
        resources = [
            "punkt",
            "stopwords",
            "averaged_perceptron_tagger",
            "wordnet",
            "omw-1.4",
        ]
        for resource in resources:
            try:
                if resource in ["punkt", "stopwords"]:
                    nltk.data.find(
                        f"tokenizers/{resource}"
                        if resource == "punkt"
                        else f"corpora/{resource}"
                    )
                elif resource == "averaged_perceptron_tagger":
                    nltk.data.find(f"taggers/{resource}")
                elif resource in ["wordnet", "omw-1.4"]:
                    nltk.data.find(f"corpora/{resource}")
                else:
                    nltk.data.find(resource)
                logger.info(f"NLTK resource '{resource}' already present")
            except LookupError:
                logger.info(f"Downloading NLTK resource '{resource}'...")
                nltk.download(resource, quiet=True)
                logger.info(f"Downloaded NLTK resource '{resource}'")
        return True
    except Exception as e:  # pragma: no cover - rarely fails
        logger.error(f"Error downloading NLTK resources: {e}")
        return False
