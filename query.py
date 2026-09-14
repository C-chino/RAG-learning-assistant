# query.py
import os
import math

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from config import (
    PERSIST_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    OLLAMA_URL,
    LLM_API_KEY,
    LLM_BASE_URL,
    PROMPT_TEMPLATE,
    RETRIEVE_TOP_K,
    USE_RERANK,
    RECALL_TOP_K,
    RERANK_TOP_K,
    RERANK_MODEL,
    ABSOLUTE_THRESHOLD,
    RELATIVE_THRESHOLD,
)

_reranker = None

def get_reranker():
    """加载 Rerank 模型（单例缓存，避免重复加载）"""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL, max_length=1024)
    return _reranker

def get_llm():
    """动态选择大模型：有 API Key 走云端，否则走本地 Ollama"""
    if LLM_API_KEY:
        # 云端部署模式（Streamlit Cloud 等）
        return ChatOpenAI(
            model="deepseek-chat",
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=0.1
        )
    else:
        # 本地开发模式
        return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL)

def get_retriever():
    """获取向量库检索器"""
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
    k = RECALL_TOP_K if USE_RERANK else RETRIEVE_TOP_K
    return vectordb.as_retriever(search_kwargs={"k": k})

def rerank_docs(question, docs):
    """重排 + 幻觉检测（双阈值机制）"""
    if not docs:
        return []

    reranker = get_reranker()
    pairs = [[question, d.page_content] for d in docs]
    scores = reranker.predict(pairs)

    # 将 logits 转换为 0-1 之间的概率分
    probs = [1 / (1 + math.exp(-s)) for s in scores]
    ranked = sorted(zip(docs, probs), key=lambda x: x[1], reverse=True)

    # ===== 调试：打印 Top-3 得分，方便你判断阈值 =====
    top3_scores = [round(p, 3) for _, p in ranked[:3]]
    print(f"[Rerank Top3得分]: {top3_scores}")

    # ===== 双阈值过滤逻辑 =====
    # 1. 如果最高分连绝对底线（0.35）都达不到，直接触发拒答
    if not ranked or ranked[0][1] < ABSOLUTE_THRESHOLD:
        return []

    # 2. 保留 Top1，以及后续得分高于相对阈值（0.20）的文档，剔除纯噪音
    filtered = [ranked[0][0]]
    for doc, score in ranked[1:]:
        if score >= RELATIVE_THRESHOLD:
            filtered.append(doc)

    return filtered[:RERANK_TOP_K]

def build_context(retrieved_docs):
    """拼接上下文；来源按（文件，页码）去重并按页码排序。"""
    parts, seen, sources = [], set(), []

    for i, doc in enumerate(retrieved_docs):
        parts.append(f"[片段{i + 1}] {doc.page_content}")

        name = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")

        key = (name, page)
        if key not in seen:
            seen.add(key)
            if isinstance(page, int):
                sources.append((page, f"[{os.path.basename(name)} p.{page + 1}]"))
            else:
                sources.append((9999, f"[{os.path.basename(name)} p.?]"))

    try:
        sources.sort(key=lambda x: int(x[0]) if str(x[0]).isdigit() else 9999)
    except Exception:
        pass

    sources_str = " ".join(s for _, s in sources)
    return "\n\n".join(parts), sources_str

def rewrite_query(question: str, history: list) -> str:
    """结合历史对话，把当前问题重写为独立、完整的检索查询"""
    if not history:
        return question

    history_str = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history[-3:]])

    prompt = f"""你是一个检索助手。请根据以下对话历史，将用户的最新提问重写为一个独立、完整、包含具体实体和概念的检索查询。
只需要输出重写后的查询语句，不要回答问题，不要加引号，不要加任何解释。

对话历史：
{history_str}

用户最新提问：{question}
重写后的查询："""

    llm = get_llm()
    rewritten = llm.invoke(prompt).content.strip()
    print(f"\n[Query重写] 原问题: {question} -> 重写后: {rewritten}")
    return rewritten

def ask(question: str, history: list = None, retriever=None):
    """主问答函数（支持多轮对话 + 幻觉拒答）"""
    if retriever is None:
        retriever = get_retriever()
    if history is None:
        history = []

    # 1. 查询重写（解决多轮对话指代不明问题）
    search_query = rewrite_query(question, history)

    # 2. 向量检索
    retrieved = retriever.invoke(search_query)

    # 3. 重排 + 双阈值幻觉检测
    if USE_RERANK:
        retrieved = rerank_docs(search_query, retrieved)

    # 4. 如果检索结果为空（低于阈值），直接拒答
    if not retrieved:
        return "资料中未找到相关内容，无法回答该问题。", []

    # 5. 拼接上下文与历史对话
    context, sources = build_context(retrieved)
    history_str = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history[-3:]])

    # 6. 生成回答
    llm = get_llm()
    prompt = PROMPT_TEMPLATE.format(history=history_str, context=context, question=question)
    answer = llm.invoke(prompt).content

    return answer, sources

if __name__ == "__main__":
    retriever = get_retriever()
    while True:
        q = input("\n问题（输入 q 退出）: ").strip()
        if q.lower() == "q":
            break
        answer, sources = ask(q, retriever=retriever)
        print("\n回答：")
        print(answer)
        print(f"\n来源：{sources}")