import torch
import numpy as np


class Search:
    def __init__(self, chunked, embeds, model):
        self.embeds = embeds
        self.model = model
        self.chunked = chunked

    def search(self, query, top=10):
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        torch.cuda.empty_cache()

        scores = []
        for e in self.embeds:
            scores.append(np.dot(np.array(query_embedding), np.array(e)))

        # scores = list(np.dot(np.array(self.embeds), query_embedding))
        rank = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for i in range(top):
            results.append([rank[i][1], self.chunked[rank[i][0]]])

        return results
