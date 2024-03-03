import os
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


# 加载embedding
def load_embedding_mode(model_name):
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        # model_kwargs={"device": "cpu"},
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


# 保存
def store_chroma(docs, embeddings, persist_directory="VectorStore"):
    if not os.path.exists(persist_directory):
        db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
        db.persist()
    else:
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        db_embedding = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
        db.add_embeddings(db_embedding)
        db.persist()
    return db


def get_vector_store(persist_directory, embeddings):
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
