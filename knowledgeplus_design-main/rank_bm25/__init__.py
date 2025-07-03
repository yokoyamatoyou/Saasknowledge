class BM25Okapi:
    def __init__(self, corpus):
        self.corpus = corpus
    def get_scores(self, tokens):
        return [1.0 for _ in self.corpus]
