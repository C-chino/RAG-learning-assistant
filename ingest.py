# ingest.py
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    DATA_DIR,
    PERSIST_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)


def load_documents(data_dir: str, filenames: list = None):
    """加载 PDF。filenames 为 None 时加载目录下全部 PDF。"""
    docs = []
    if filenames is None:
        filenames = [
            f for f in os.listdir(data_dir)
            if f.lower().endswith(".pdf")
        ]

    for filename in filenames:
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"跳过不存在的文件：{path}")
            continue
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    print(f"加载 {len(docs)} 页文档")
    return docs


def split_documents(docs):
    """切分文档为 chunk。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"切分出 {len(chunks)} 个 chunk")
    return chunks


def get_vectordb():
    """打开已有索引（没有则新建）。"""
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding,
    )


def add_files(filenames: list):
    """把指定文件追加进索引，返回切分出的 chunk 数。"""
    docs = load_documents(DATA_DIR, filenames)
    if not docs:
        return 0
    chunks = split_documents(docs)
    vectordb = get_vectordb()
    vectordb.add_documents(chunks)
    return len(chunks)


if __name__ == "__main__":
    docs = load_documents(DATA_DIR)
    if not docs:
        print("data 目录下没有 PDF 文件，请先放入课程 PDF。")
    else:
        chunks = split_documents(docs)
        vectordb = get_vectordb()
        vectordb.add_documents(chunks)
        print("索引构建完成")