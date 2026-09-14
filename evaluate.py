# evaluate.py
import json
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from query import get_retriever, rerank_docs, rewrite_query
from config import RETRIEVE_TOP_K, USE_RERANK, RERANK_TOP_K


def to_meta_page(footer_page: int) -> int:
    """把 PDF 页脚页码转成 metadata 的 0 起始页码。"""
    return footer_page - 1


def main():
    test_cases = json.load(open("test_cases.json", encoding="utf-8"))
    retriever = get_retriever()
    hit = 0

    for case in test_cases:
        # 直接调用检索层，不走生成/拒答逻辑，避免拒答干扰评测
        search_query = rewrite_query(case["question"], [])
        docs = retriever.invoke(search_query)

        # 打印原始召回前5页页码（诊断用）
        raw_pages = [d.metadata.get("page", -1) + 1 for d in docs[:5]]
        print(f"\n原始召回Top5页码: {raw_pages}")

        if USE_RERANK:
            docs = rerank_docs(search_query, docs)
            rerank_pages = [d.metadata.get("page", -1) + 1 for d in docs]
            print(f"Rerank后页码: {rerank_pages}")
        else:
            docs = docs[:RETRIEVE_TOP_K]

        # 命中判断：只要检索结果中存在期望页码（±1容差），即算命中
        found = any(
            abs(d.metadata.get("page", -1) - to_meta_page(p)) <= 1
            for d in docs
            for p in case["expect_pages"]
        )

        hit += found
        print(f"{'✓' if found else '✗'} {case['question']}（期望 {case['expect_pages']}）")

    print(f"\n=== 最终评测结果 ===")
    print(f"命中率：{hit}/{len(test_cases)} = {hit / len(test_cases):.1%}")


if __name__ == "__main__":
    main()