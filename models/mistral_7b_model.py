import os.path

from transformers import BitsAndBytesConfig, pipeline
from langchain import HuggingFacePipeline
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class MistralService(LLM):
    # 基于本地 InternLM 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __int__(self, model_path: str, model_id="", bnb_config=""):
        super().__init__()
        # 从本地初始化模型
        if os.path.exists(model_path):
            print("正在从本地加载模型...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path).to(torch.bfloat16).cuda()
            print("完成本地模型的加载")
        else:
            print("正在从远程加载模型...")
            if bnb_config:
                # 加载并配置模型
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",  # 自动选择运行设备
                    quantization_config=bnb_config,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
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