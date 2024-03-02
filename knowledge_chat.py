import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from models.load_model import load_llm
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from models.chatgml3_6b_model import ChatGLMService

# 加载embedding的字典
embedding_model_dict = {
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "gte-large": "thenlper/gte-large",
}

llm_dict = {
    'ChatGLM3-6B': {
        'model_name': 'THUDM/chatglm3-6b'
    }
}


# 加载文档
def load_documents(directory="library"):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    # 初始化文本分割器
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=2
    )

    # 切分文本
    split_documents = text_splitter.split_documents(documents)
    return split_documents


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


def get_vector_store(persist_directory="VectorStore", embeddings="thenlper/gte-large"):
    return Chroma(persist_directory='VectorStore', embedding_function=embeddings)


# 获取本地知识库
def get_knowledge_based_answer(
    query,
    large_language_model,
    vector_store,
    top_k,
    web_content,
    chat_history=[],
    history_len=3,
    temperature=0.01,
    top_p=0.9,
):
    if web_content:
        prompt_template = f"""基于以下已知信息，简洁和专业的来回答用户的问题。
                            如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
                            已知网络检索内容：{web_content}""" + """
                            已知内容:
                            {context}
                            问题:
                            {question}"""
    else:
        prompt_template = """基于以下已知信息，请简洁并专业地回答用户的问题。
            如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。

            已知内容:
            {context}

            问题:
            {question}"""
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    chatLLM = ChatGLMService()
    chatLLM.model_name = llm_dict[large_language_model]['model_name']

    chatLLM.history = chat_history[-history_len:] if history_len > 0 else []
    chatLLM.temperature = temperature
    chatLLM.top_p = top_p

    knowledge_chain = RetrievalQA.from_llm(
        llm=chatLLM,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": top_k}),
        prompt=prompt)
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}")

    knowledge_chain.return_source_documents = True
    result = knowledge_chain({"query": query})

    return result['result']


if __name__ == "__main__":

    doc = load_documents()

