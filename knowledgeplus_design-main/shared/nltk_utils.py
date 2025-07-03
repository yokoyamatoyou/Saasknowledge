import logging
import nltk

logger = logging.getLogger(__name__)


def ensure_nltk_resources() -> bool:
    """Ensure required NLTK resources are available."""
    try:
        logger.info("Checking NLTK resources...")
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
                        f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}"
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
