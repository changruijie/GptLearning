import gradio as gr
import os
import time
import knowledge_chat

# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video). Plus shows support for streaming text.


def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)


def add_file(history, file):
    # 拿到上传的文件夹
    directory = os.path.dirname(file.name)
    documents = knowledge_chat.load_documents(directory)
    knowledge_chat.store_chroma(documents, knowledge_chat.embedding_model_dict["gte-large"])
    history = history + [((file.name,), None)]
    return history


def bot(history):
    message = history[-1][0]
    if isinstance(message, tuple):
        response = "文件上传成功！"
    else:
        response = knowledge_chat.get_knowledge_based_answer(message,
                                                             'ChatGLM3-6B',
                                                             knowledge_chat.get_vector_store("VectorStore",knowledge_chat.embedding_model_dict["gte-large"]),
                                                             3,
                                                             '',
                                                             history)

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
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
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
