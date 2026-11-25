from loader import load_and_split
from hybrid import HybridRetriever
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):

        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query, results, top_k=5):
        if not results:
            return []

        pairs = [(query, r["text"]) for r in results]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        return [{**r, "score": float(s)} for r, s in ranked[:top_k]]

class RerankRetriever: 

    def __init__(self, candidates=20):

        self.hybrid = HybridRetriever()
        self.reranker = Reranker()
        self.candidates = candidates

    def build(self, chunks):

        self.hybrid.build(chunks)

    def search(self, query, top_k=5):
        candidates = self.hybrid.search(query, top_k=self.candidates, candidates = self.candidates)
        return self.reranker.rerank(query, candidates, top_k=top_k)


def show(title,results):
    print(f" 【{title}】")
    for rank, r in enumerate(results, start=1):
        print(f" {rank}.[{r['score']:.3f}] {r['source']} 第{r['chunk_index']}块")

if __name__ == "__main__":
    chunks = load_and_split()
    retriever = RerankRetriever()

    retriever.build(chunks)
    print("索引建立完成，输入问题对比效果，直接按回车是退出\n" )

    while True:
        query=input("问题：").strip()
        if not query:
            break

        show("混合检索", retriever.hybrid.search(query, top_k =5))
        show("混合检索+重排序", retriever.search(query, top_k =5))
        print("\n")