# 📚 RAG 课程学习助手

> 基于 LangChain + Chroma + BGE + DeepSeek 构建的垂直领域 RAG 问答助手。支持上传 PDF、多轮对话、答案溯源与幻觉拒答。

## 🔗 在线体验

👉 **[点击访问 Streamlit 在线演示](https://rag-learning-assistant-juy3shcjfzq5gfunwtvmnr.streamlit.app/)**

> ⚠️ 云端为免费版 1GB 内存，首次加载需 1-2 分钟（下载轻量模型）。本地运行 `bge-large` + `bge-reranker-v2-m3` 可获得更优效果。

## 🎥 演示视频

👉 **[点击观看 1 分钟完整演示（含多轮对话与拒答机制）](https://github.com/user-attachments/assets/afc23090-b185-41fa-8513-9320a8bf3393)**

## ✨ 核心功能

- **文档解析**：基于 `PyPDFLoader` 加载 PDF，`RecursiveCharacterTextSplitter` 语义切分（Chunk 512 / Overlap 50）。
- **检索增强**：Chroma 向量库 + `bge-small-zh-v1.5` 嵌入 + `bge-reranker-base` 重排（Rerank）。
- **多轮对话**：引入 **Query Rewriting** 模块，将“具体有哪些”等模糊指代重写为完整查询，解决多轮对话中的语义丢失问题。
- **幻觉拒答**：基于**双阈值机制**（绝对底线 0.1 + 相对阈值 0.05）过滤低置信度文档，避免模型胡编乱造。
- **可视化界面**：Streamlit 对话界面，支持流式输出与答案溯源展示。

## 📊 核心实验数据

自建 **43 题**标注评测集，通过 **9 组消融实验**（Chunk Size × Overlap）锁定最优切分参数（512/50）。

| 阶段 | 检索命中率（Recall@5） | 优化手段 |
| :--- | :--- | :--- |
| 基线（Rerank 关闭） | 62% | 原始向量检索 |
| 优化后（Rerank 开启） | **75%** | 引入 BGE-Reranker 重排 + 双阈值过滤 |

> 📌 **失败案例分析**：剩余 25% 的失败题目集中在 `git log` vs `git reflog` 等细粒度命令区分，以及章节总结类问题（缺乏全局上下文）。下一步计划引入 **BM25 混合检索** 和 **滑动窗口聚合** 来突破这一瓶颈。

## 🛠️ 技术栈

| 组件 | 本地开发 | 云端部署 |
| :--- | :--- | :--- |
| LLM | Qwen2.5:7b (Ollama) | DeepSeek API |
| Embedding | BAAI/bge-large-zh-v1.5 | BAAI/bge-small-zh-v1.5 |
| Reranker | BAAI/bge-reranker-v2-m3 | BAAI/bge-reranker-base |
| 向量库 | ChromaDB | ChromaDB |
| 框架 | LangChain + Streamlit | LangChain + Streamlit |

## 📁 项目结构

```
Rag-assistant/
├── data/                  # 课程 PDF 文件
├── chroma_db/             # 向量库（自动生成）
├── config.py              # 全局配置（切分参数、模型、阈值）
├── ingest.py              # 文档加载、切分、索引构建
├── query.py               # 检索、重排、双阈值拒答、Query Rewriting
├── app.py                 # Streamlit 界面
├── evaluate.py            # 评测脚本（Recall@5）
├── test_cases.json        # 43 题标注评测集
└── requirements.txt       # 项目依赖
```

## 🚀 快速开始

### 本地运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动本地 Ollama 并拉取模型
ollama pull qwen2.5:7b

# 3. 将 PDF 放入 data/ 目录

# 4. 构建索引
python ingest.py

# 5. 启动界面
streamlit run app.py
```

### 云端部署（Streamlit Community Cloud）

已在 **Streamlit Community Cloud** 部署，通过 Secrets 注入 `LLM_API_KEY` 和 `HF_ENDPOINT`，模型自动降级为轻量版以适配 1GB 内存限制。

配置 Secrets 示例：
```toml
LLM_API_KEY = "sk-你的DeepSeek API Key"
HF_ENDPOINT = "https://hf-mirror.com"
```

## 📄 License

MIT License


