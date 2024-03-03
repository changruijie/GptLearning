import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import os


# 将LLM保存到本地
def save_llm_local(model_id: str, bnb_config, file_path: str):
    if bnb_config:
        # 加载并配置模型
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",  # 自动选择运行设备
            quantization_config=bnb_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f'file_path:{file_path}')
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    write_to_folder(file_path, model, tokenizer)
    return model, tokenizer


def write_to_folder(file_path, model, tokenizer):
    # # 获取文件夹路径
    # folder_path = os.path.dirname(file_path)

    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # 保存模型
    model.save_pretrained(file_path)
    tokenizer.save_pretrained(file_path)


if __name__ == "__main__":
    # 配置BitsAndBytes的设定，用于模型的量化以提高效率。
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 启用位元量化
        bnb_4bit_compute_dtype=torch.float16,  # 计算时使用的数据类型
        bnb_4bit_quant_type="nf4",  # 量化类型
        bnb_4bit_use_double_quant=True,  # 使用双重量化
    )

    # 1、保存mistral_7b
    # save_llm_local("mistralai/Mistral-7B-Instruct-v0.2", quantization_config,
    #                "./drive/MyDrive/local_library/model/mistral")
    # 2、保存chatglm3_6b
    save_llm_local("THUDM/chatglm3-6b", None,
                   "./drive/MyDrive/local_library/model/chatglm3_6b")