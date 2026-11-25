import json
import sys
from collections import defaultdict
from  pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"src"))

from loader import load_and_split
from reranker import RerankRetriever

TESTSET = ROOT/"eval/testset.json"
RESULTS = ROOT/"eval/result.md"
MAX_K = 5
KS = [1, 3, 5]

def is_hit(chunk,evidences):
    return evidences in chunk["text"]

def recall_at_k(results, evidences, k):
    "how many of the evidences are found in the top k results / total number of evidences"
    top = results[:k]
    found = sum(any(is_hit(c,e) for c in top) for e in evidences)
    return found / len(evidences)

def reciprocal_rank(results, evidences):
    "the reciprocal of the rank of the first result that contains any of the evidences"
    for rank, c in enumerate(results, start=1):
        if any(is_hit(c,e) for e in evidences):
            return 1 / rank
    return 0.0

def check_labels(testset, chunk):
    "each evidence in the testset should be found in at least one chunk, otherwise it's a labeling error"
    problems = []
    for q in testset:
        for e in q["evidence"]:
            if not any(is_hit(c,e) for c in chunk):
                problems.append((q["question"], e))

    return problems

def main():
    testset = json.loads(TESTSET.read_text(encoding = "utf-8"))
    chunks = load_and_split(str(ROOT/"data"))

    problems = check_labels(testset, chunks)
    if problems:
        print("以下证据句在文本块里找不到， 请检查原文，可能有错字、标点不同，或者正好被切块切断了。")

        for qid, e in problems:
            print(f"问题: {qid}, 证据句: {e}")

        return 

    retriever = RerankRetriever()
    retriever.build(chunks)
    configs = {
        "向量检索": retriever.hybrid.vector,
        "混合检索": retriever.hybrid,
        "混合＋重排序": retriever,
    }

    scored = [
        q for q in testset if q["evidence"]
    ]

    print(f"共{len(testset)}题，其中{len(scored)}题参与评估\n")

    summary = {}

    by_type = defaultdict(dict)
    misses = defaultdict(list)

    for name, r in configs.items():
        print(f"正在评估:{name}...")
        rows = []
        for q in scored:
            results = r.search(q["question"], top_k=MAX_K)
            row = {f"R@{k}": recall_at_k(results, q["evidence"], k) for k in KS}
            row["RR"] = reciprocal_rank(results, q["evidence"])
            row["type"] = q["type"]
            rows.append(row)
            if row[f"R@{MAX_K}"] < 1:
                misses[name].append(q)

        n = len(rows)
        summary[name] = {
            f"R@{k}": sum(x[f"R@{k}"] for x in rows) / n for k in KS
        }

        summary[name]["MRR"] = sum(x["RR"] for x in rows) / n

        groups = defaultdict(list)
        for x in rows:
            groups[x["type"]].append(x[f"R@{MAX_K}"])

        for t, vals in groups.items():
            by_type[t][name] = sum(vals) / len(vals)

    names = list(configs)
    lines = ["## 评估结果", "", f"评测题数： {len(scored)}", ""]
    lines += ["| 方案 |" + " | ".join(f"Recall@{k}" for k in KS) + " | MRR| "]
    lines += ["|---" * (len(KS) + 2) + "|"]
    for name in names:
        s = summary[name]
        lines += [f"| {name} |" + " | ".join(f"{s[f'R@{k}']:.3f}" for k in KS) + f" | {s['MRR']:.3f}|"]
    lines += ["", "### 各题型 Recall@5", ""]
    lines += ["| 题型 | 题数 | "+" | ".join(names) + " |"]
    lines += ["|---" * (len(names) + 2) + "|"]
    for t in by_type:
        count = sum(1 for q in scored if q["type"] == t)
        lines += [f"| {t} | {count} | " + " | ".join(f"{by_type[t][n]:.3f}" for n in names) + " |"]

    lines += ["", f"### 《{names[-1]}》前 {MAX_K}名没找全证据的题", ""]
    for q in misses [names[-1]]:
        lines += [f"- 第{q['id']}题 ({q['type']}) {q['question']}"]

    report = "\n".join(lines)
    RESULTS.write_text("\n".join(lines), encoding="utf-8")
    print("\n"+report)
    print(f"\n评估结果已保存到 {RESULTS}")


if __name__ == "__main__":
    main()