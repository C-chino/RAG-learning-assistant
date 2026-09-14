# app.py
import os
import sys

# ==========================================
# ⚠️ 必须放在最顶部：解决 Streamlit Cloud 上 ChromaDB 的 SQLite3 报错
# ==========================================
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st

from ingest import add_files
from query import ask, get_retriever, get_reranker
from config import DATA_DIR

os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="课程学习助手", page_icon="📚")
st.title("课程学习助手")


# ==========================================
# 核心优化：使用 Streamlit 官方缓存加载模型
# 保证 retriever 和 reranker 只在应用启动时加载一次，永不销毁
# ==========================================
@st.cache_resource
def load_resources():
    retriever = get_retriever()
    # 预加载重排模型，防止首次提问卡顿
    get_reranker()
    return retriever


retriever = load_resources()

# ==========================================
# 侧边栏：上传文档
# ==========================================
with st.sidebar:
    st.header("文档管理")
    uploaded = st.file_uploader(
        "上传课程 PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded and st.button("建立索引"):
        saved = []
        for f in uploaded:
            path = os.path.join(DATA_DIR, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            saved.append(f.name)

        with st.spinner("正在切分并建立索引，请稍候..."):
            n = add_files(saved)

        # 关键：新上传文档并重建索引后，必须清空模型缓存，否则检索不到新内容
        load_resources.clear()
        st.success(f"已索引 {len(saved)} 个文件，共 {n} 个片段")

# ==========================================
# 主界面：对话历史展示
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("查看来源"):
                st.markdown(msg["sources"])

# ==========================================
# 主界面：用户输入与生成回答
# ==========================================
question = st.chat_input("对课程资料提问...")

if question:
    # 1. 记录并展示用户提问
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 2. 生成助手回答
    with st.chat_message("assistant"):
        with st.spinner("检索并生成中..."):

            # ===== 严格提取历史对话（成对提取，防止越界）=====
            history = []
            msgs = st.session_state.messages[:-1]  # 排除刚加入的当前提问
            for i in range(0, len(msgs) - 1, 2):
                if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                    history.append({
                        "user": msgs[i]["content"],
                        "assistant": msgs[i + 1]["content"]
                    })

            # 传入 history 以支持多轮对话，传入 retriever 避免重复加载
            answer, sources = ask(question, history=history, retriever=retriever)

            st.markdown(answer)
            if sources:
                with st.expander("查看来源"):
                    st.markdown(sources)

        # 3. 记录助手回答
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })