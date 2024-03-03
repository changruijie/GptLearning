from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from models.chatglm3_6b_model import ChatGLMService
from models.mistral_7b_model import MistralService
from langchain import PromptTemplate, LLMChain

model_type_dict = {
    "chatglm": "chatglm",
    "mistral": "mistral"
}


# 获取本地知识库
def get_knowledge_based_answer(
        query,
        llm,
        prompt,
        vector_store,
        top_k,
):
    knowledge_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": top_k}),
        prompt=prompt)
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}")

    knowledge_chain.return_source_documents = True
    result = knowledge_chain({"query": query})

    return result


# 使用模型本身获取问题结果
def get_general_answer(query, context, llm, prompt):
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    answer = llm_chain.run({"context": context, "question": query})

    return answer


def load_llm(model_path, model_type):
    prompt_template = """基于以下已知信息，请简洁并专业地回答用户的问题。
        如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。

        已知内容:
        {context}

        问题:
        {question}"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    if model_type == model_type_dict.get("chatglm"):
        # 使用chatglm
        chat_llm = ChatGLMService(model_path)
        llm = chat_llm.load_llm()
    elif model_type == model_type_dict.get("mistral"):
        chat_llm = MistralService(model_path)
        llm = chat_llm.load_llm()
    else:
        raise ValueError("不存在的模型，请补充")

    return prompt, llm
