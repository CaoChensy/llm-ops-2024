import torch
import numpy as np
from rank_bm25 import BM25Plus
from jieba import cut_for_search


class Search:
    def __init__(self,
                 chunked,
                 embeds,
                 model,
                 chunked_tokens=None,
                 top=10,
                 thresh=None,
                 c=60):
        """chunked: 文本
           embeds：向量化的文本
           model：向量模型
        """
        self.embeds = embeds
        self.model = model
        self.chunked = chunked
        self.chunked_tokens = chunked_tokens
        self.bm25 = BM25Plus(self.chunked_tokens) if self.chunked_tokens else None
        self.top = top
        self.thresh = thresh
        self.c = c

    @staticmethod
    def rank(scores, texts, top=10, thresh=None):
        """
        return [[score, text]]
        """
        ranks = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for i in range(top):
            if thresh:
                if ranks[i][1] > thresh:
                    pass
                else:
                    break
            results.append([ranks[i][1], texts[ranks[i][0]]])

        # 去重
        unique = set()
        final_results = []
        for result in results:
            if result[1] not in unique:
                final_results.append(result)
                unique.add(result[1])

        return final_results

    @staticmethod
    def rrf(bm25_results, embedding_results, c=60):
        rrf_scores = {}
        for idx, result in enumerate(bm25_results):
            rank = idx+1
            rrf_scores[result[1]] = {"rrf": 1/(rank+c), "bm25_score": result[0]}

        for idx, result in enumerate(embedding_results):
            rank = idx + 1
            if result[1] in rrf_scores:
                bm25_rrf = rrf_scores[result[1]]["rrf"]
                rrf_scores[result[1]]["rrf"] = 1/(rank+c) + bm25_rrf
                rrf_scores[result[1]]["embedding_score"] = result[0]
            else:
                rrf_scores[result[1]] = {"rrf": 1/(rank+c), "embedding_score": result[0]}

        rrf_scores = dict(sorted(rrf_scores.items(), key=lambda x: x[1]["rrf"], reverse=True))
        return rrf_scores

    def search(self, query):
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        torch.cuda.empty_cache()

        scores = []
        for e in self.embeds:
            scores.append(np.dot(np.array(query_embedding), np.array(e)))
        # scores = list(np.dot(np.array(self.embeds), query_embedding))

        return self.rank(scores, self.chunked, self.top, self.thresh)

    def bm25_search(self, query):
        query_lst = list(cut_for_search(query))
        bm25_scores = self.bm25.get_scores(query_lst)
        bm25_thresh = self.thresh * 100 if self.thresh else None
        bm25_result = self.rank(bm25_scores, self.chunked, self.top, bm25_thresh)

        return bm25_result

    def mix_search(self, query):
        """
        bm25+embedding混合检索
        query
        top: 召回数量， 默认10
        thresh: 召回阈值， 百分比值，如0.8
        # weights: [bm25, embedding] bm25和embedding的占比，默认[0.5, 0.5]
        *注： 权重计算用rrf公式
        *注: 先限定召回数量top，再限定阈值thresh，最终根据rrf返回结果。因此最终数量可能小于召回数量top
        """
        bm25_result = self.bm25_search(query)
        embedding_result = self.search(query)

        rrf_scores = self.rrf(bm25_result, embedding_result, self.c)

        return rrf_scores
