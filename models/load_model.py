from transformers import pipeline
from langchain import HuggingFacePipeline


# 加载大模型
def load_llm(model, tokenizer):
    # 创建一个用于文本生成的pipeline
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=3200,
        do_sample=True,
        top_k=5,  # top_k 控制了模型生成词汇时考虑的概率最高的 k 个词
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    # 创建HuggingFacePipeline实例，用于后续语言生成
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return llm
