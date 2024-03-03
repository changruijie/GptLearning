import gradio as gr
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from predict.question_answer import load_llm, model_type_dict,get_knowledge_based_answer,get_general_answer
from docments.handle_documents import load_file
from docments.handle_vector import store_chroma, get_vector_store
import torch

embedding_model_dict = {
    "gte": "thenlper/gte-large",
}
init_llm = "mistralai/Mistral-7B-Instruct-v0.2"
init_embedding_model = "thenlper/gte-large"
init_model_type = model_type_dict['mistral']
model_type_dict = {
    "mistral":"mistral",
    "chatglm":"chatglm"
}
llm_model_dict = {
    "chatglm": "THUDM/chatglm3-6b",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2"
}


class KnowledgeBasedChatLLM:

    llm: object = None
    embeddings: object = None
    prompt: object = None

    def init_model_config(
        self,
        model_name: str = init_llm,
        model_type: str = init_model_type,
        embedding_model_name: str = init_embedding_model,
    ):

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name, )
        self.llm = None
        torch.cuda.empty_cache()
        print(model_name,model_type,embedding_model_name)
        self.prompt, self.llm = load_llm(model_name, model_type)

    def init_knowledge_vector_store(self, filepath):
        docs = load_file(filepath)
        vector_store = store_chroma(docs, self.embeddings, 'VectorStore')
        # vector_store = FAISS.from_documents(docs, self.embeddings)
        # vector_store.save_local('faiss_index')
        return vector_store

    def get_knowledge_based_answer(self,
                                   query,
                                   web_content,
                                   top_k: int = 6,
                                   ):
        if web_content:
            result = get_general_answer(query, '', self.llm, self.prompt)
        else:
            result = get_knowledge_based_answer(query, self.llm, self.prompt, self.init_knowledge_vector_store('books'), top_k)
        return result


def update_status(history, status):
    history = history + [[None, status]]
    print(status)
    return history


knowledge_based_chat_llm = KnowledgeBasedChatLLM()


def init_model():
    try:
        print("开始加载模型配置")
        knowledge_based_chat_llm.init_model_config()
        print("模型配置加载成功")
        knowledge_based_chat_llm.llm._call("你好")
        return """初始模型已成功加载，可以开始对话"""
    except Exception as e:
        print(f"加载模型出错: {e}")  # 打印详细的异常信息
        return """模型未成功加载，请重新选择模型后点击"重新加载模型"按钮"""


def clear_session():
    return '', None


def reinit_model(model_name,model_type,embedding_model_name,history):
    try:
        print(model_name,model_type,embedding_model_name)
        knowledge_based_chat_llm.init_model_config(model_name,model_type,embedding_model_name)
        model_status = """模型已成功重新加载，可以开始对话"""
    except Exception as e:
        model_status = """模型未成功重新加载，请点击重新加载模型"""
    return history + [[None, model_status]]


def init_vector_store(file_obj):
    print(f'file_obj:{file_obj}')
    vector_store = knowledge_based_chat_llm.init_knowledge_vector_store(
        file_obj.name)
    return vector_store


def predict(input,
            use_web,
            top_k,
            history=None):
    if history is None:
        history = []

    resp = knowledge_based_chat_llm.get_knowledge_based_answer(
        query=input,
        web_content=use_web,
        top_k=top_k)
    print(f"history: {history}")
    print(f"input: {input}")
    print(f"resp: {resp}")
    history.append((input, resp))
    # history.append((input, resp['result']))
    return '', history, history


model_status = init_model()

if __name__ == "__main__":
    block = gr.Blocks()
    with block as demo:
        gr.Markdown("""<h1><center>local_library_llm</center></h1>
            <center><font size=3>
            本项目基于LangChain和大型语言模型系列模型, 提供基于本地知识的自动问答应用. <br>
            </center></font>
            """)
        model_status = gr.State(model_status)
        with gr.Row():
            with gr.Column(scale=1):
                model_choose = gr.Accordion("模型选择")
                with model_choose:
                    large_language_model = gr.Dropdown(list(
                        llm_model_dict.values()),
                        label="large language model"
                        )

                    language_model_type = gr.Dropdown(list(
                        model_type_dict.values()),
                        label="model type"
                        )

                    embedding_model = gr.Dropdown(list(
                        embedding_model_dict.values()),
                                                  label="Embedding model")
                    load_model_button = gr.Button("重新加载模型")
                model_argument = gr.Accordion("模型参数配置")
                with model_argument:

                    top_k = gr.Slider(1,
                                      10,
                                      value=6,
                                      step=1,
                                      label="vector search top k",
                                      interactive=True)

                    history_len = gr.Slider(0,
                                            5,
                                            value=3,
                                            step=1,
                                            label="history len",
                                            interactive=True)

                    temperature = gr.Slider(0,
                                            1,
                                            value=0.01,
                                            step=0.01,
                                            label="temperature",
                                            interactive=True)
                    top_p = gr.Slider(0,
                                      1,
                                      value=0.9,
                                      step=0.1,
                                      label="top_p",
                                      interactive=True)

                file = gr.File(label='请上传知识库文件',
                               file_types=['.txt', '.md', '.docx', '.pdf'])

                init_vs = gr.Button("知识库文件向量化")

                use_web = gr.Radio(["True", "False"],
                                   label="Web Search",
                                   value="False")

            with gr.Column(scale=4):
                chatbot = gr.Chatbot([[None, model_status.value]],
                                     label='ChatLLM')
                message = gr.Textbox(label='请输入问题')
                state = gr.State()

                with gr.Row():
                    clear_history = gr.Button("🧹 清除历史对话")
                    send = gr.Button("🚀 发送")

            load_model_button.click(
                reinit_model,
                show_progress="full",
                inputs=[large_language_model,language_model_type, embedding_model, chatbot],
                outputs=chatbot,
            )
            init_vs.click(
                init_vector_store,
                show_progress="full",
                inputs=[file],
                outputs=[],
            )
            send.click(predict,
                       inputs=[
                           message, use_web, top_k, state
                       ],
                       outputs=[message, chatbot, state])
            clear_history.click(fn=clear_session,
                                inputs=[],
                                outputs=[chatbot, state],
                                queue=False)

            message.submit(predict,
                           inputs=[
                               message, use_web, top_k, state
                           ],
                           outputs=[message, chatbot, state])
        gr.Markdown("""提醒：<br>
        1. 使用时请先上传自己的知识文件，并且文件中不含某些特殊字符，否则将返回error. <br>
        """)
    # threads to consume the request
    demo.queue() \
        .launch(debug=True)
