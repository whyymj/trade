from typing import Dict
from services.fund_intel.clients import LLMClient


class NewsClassifier:
    def __init__(self):
        self.llm_client = LLMClient()

    def classify(self, text: str) -> Dict:
        industries = ["宏观", "行业", "全球", "政策", "公司"]
        industry_text = ", ".join(industries)

        prompt = f"""你是一个行业分类专家，请将以下文本分类到以下行业之一：{industry_text}

文本：{text}

只返回行业名称，不要其他内容。"""

        try:
            result = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                provider="deepseek",
                use_cache=True,
            )
            result = result.strip()
            if result in industries:
                return {"industry": result, "confidence": 0.8}
        except:
            pass

        return {"industry": "其他", "confidence": 0.3}
