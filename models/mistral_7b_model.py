# 加载大模型
# langchain使用mistral-7b生成文本案例
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

    def __int__(self, model_path: str, model_id, bnb_config):
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
