import gradio as gr
import os
import time
from predict.question_answer import get_knowledge_based_answer
from docments.handle_vector import get_vector_store
from docments.handle_vector import load_embedding_mode
from docments.handle_vector import store_chroma
from docments.handle_documents import load_file


# 输出好恶
def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


# 上传文档
def add_text(history, text):
    directory = os.path.dirname(text.name)
    documents = load_file("", directory)
    embeddings = load_embedding_mode('thenlper/gte-large')
    store_chroma(documents, embeddings)
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)


# 上传文件
def add_file(history, file):
    # 拿到上传的文件夹
    directory = os.path.dirname(file.name)
    documents = load_file(directory)
    # 默认使用阿里达摩院的embedding模型
    embeddings = load_embedding_mode('thenlper/gte-large')
    store_chroma(documents, embeddings)
    history = history + [((file.name,), None)]
    return history


def bot(history):
    message = history[-1][0]
    if isinstance(message, tuple):
        response = "文件上传成功！"
    else:
        # chatgml3 - chatglm3直接加载本地会报错，后续具体分析怎么做
        model_path = "THUDM/chatglm3-6b"
        # model_path = "./model/mistral"
        vector_store = get_vector_store("VectorStore", load_embedding_mode('thenlper/gte-large'))
        response = get_knowledge_based_answer(message, model_path, vector_store, 3, '')

    # response = "**That's cool!**"
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        # avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter, or upload an image",
            container=False,
        )
        btn = gr.UploadButton("📁", file_types=["image", "video", "audio", "text", "file"])

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot, api_name="bot_response"
    )
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

    chatbot.like(print_like_dislike, None, None)

demo.queue()
if __name__ == "__main__":
    demo.launch()
