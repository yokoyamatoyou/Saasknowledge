"""Download required NLTK resources."""

import logging

import nltk

logger = logging.getLogger(__name__)

RESOURCES = [
    "punkt",
    "stopwords",
    "averaged_perceptron_tagger",
    "wordnet",
    "omw-1.4",
]

for resource in RESOURCES:
    nltk.download(resource)

logger.info("NLTK resource download completed.")
