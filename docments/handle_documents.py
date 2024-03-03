import os
from langchain.text_splitter import CharacterTextSplitter,MarkdownTextSplitter
from langchain.document_loaders import UnstructuredFileLoader,UnstructuredMarkdownLoader
from langchain.document_loaders import UnstructuredPDFLoader
from rapidocr_onnxruntime import RapidOCR


def load_file(file_path):
    # 从docs_path路径加载文件（加载所有文件）
    docs = []
    if os.path.isfile(file_path):
        docs = load_type_file(file_path)
    elif os.path.isdir(file_path):
        for doc in os.listdir(file_path):
            doc_path: str = f'{file_path}/{doc}'
            split_docs = load_type_file(doc_path)
            docs.extend(split_docs)
    else:
        print(f"{file_path} 不是一个有效的文件路径或文件夹路径。")

    return docs


def load_type_file(doc_path: str):
    if doc_path.endswith('.txt'):
        return load_txt_file(doc_path)
    elif doc_path.endswith('.md'):
        return load_md_file(doc_path)
    elif doc_path.endswith('.pdf'):
        return load_pdf_file(doc_path)
    elif doc_path.endswith('.jpg'):
        return load_jpg_file(doc_path)


# 加载txt文件
def load_txt_file(txt_file, chunk_size=500, chunk_overlap=10):
    loader = UnstructuredFileLoader(txt_file)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(docs)
    return split_docs


# 加载md文件
def load_md_file(md_file, chunk_size=500, chunk_overlap=10):
    loader = UnstructuredMarkdownLoader(md_file)
    docs = loader.load()
    text_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(docs)
    return split_docs


# 加载pdf文件
def load_pdf_file(pdf_file, chunk_size=500, chunk_overlap=10):
    loader = UnstructuredPDFLoader(pdf_file)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(docs)
    return split_docs


# 加载jpg文件
def load_jpg_file(jpg_file, chunk_size=500, chunk_overlap=10):
    ocr = RapidOCR()
    result, _ = ocr(jpg_file)
    docs = ""
    if result:
        ocr_result = [line[1] for line in result]
        docs += "\n".join(ocr_result)
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.create_documents([docs])
    return split_docs



