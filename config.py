# config.py
import os

# ===== 路径配置 =====
DATA_DIR = "./data"
PERSIST_DIR = "./chroma_db"

# ===== 切分参数 =====
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# ===== 检索配置 =====
RETRIEVE_TOP_K = 5
RECALL_TOP_K = 10
RERANK_TOP_K = 5
USE_RERANK = True

# config.py
# 换回 small 版本（约 100MB）
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
# 换回 base 版本（约 400MB）
RERANK_MODEL = "BAAI/bge-reranker-base"

# ===== LLM 配置（本地 Ollama，云端 API）=====
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# 如果配置了 API Key，则使用云端 API
LLM_API_KEY = os.getenv("LLM_API_KEY", None)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")

# ===== 拒答阈值 =====
ABSOLUTE_THRESHOLD = 0.35
RELATIVE_THRESHOLD = 0.20

# ===== Prompt =====
PROMPT_TEMPLATE = """你是课程学习助手。请严格根据以下资料回答问题。
如果资料中没有答案，请回答“资料中未找到相关内容”。
请分点作答，正文中不要标注来源，来源会由系统统一附加。

【历史对话】
{history}

【资料】
{context}

【问题】
{question}
"""