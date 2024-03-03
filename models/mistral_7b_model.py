import os.path

from transformers import BitsAndBytesConfig, pipeline
from langchain import HuggingFacePipeline
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from models.save_model import save_llm_local


class MistralService(LLM):
    # 基于本地 InternLM 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_path: str):
        super().__init__()
        # 从本地初始化模型
        if os.path.exists(model_path):
            print("正在从本地加载模型...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path).to(torch.bfloat16).cuda()
            print("完成本地模型的加载")
        else:
            print("正在从远程加载模型...")
            # 配置BitsAndBytes的设定，用于模型的量化以提高效率。
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,  # 启用位元量化
                bnb_4bit_compute_dtype=torch.float16,  # 计算时使用的数据类型
                bnb_4bit_quant_type="nf4",  # 量化类型
                bnb_4bit_use_double_quant=True,  # 使用双重量化
            )
            self.model, self.tokenizer = save_llm_local(model_path, quantization_config, "mistral")
            print("完成远程模型的加载")

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        # 重写调用函数
        response, history = self.model.chat(self.tokenizer, prompt, history=[])
        return response

    @property
    def _llm_type(self) -> str:
        return "Mistral-7B"

    def load_llm(self):
        # 创建一个用于文本生成的pipeline
        text_generation_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=3200,
            do_sample=True,
            top_k=5,  # top_k 控制了模型生成词汇时考虑的概率最高的 k 个词
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # 创建HuggingFacePipeline实例，用于后续语言生成
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        return llm