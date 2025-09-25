from typing import Dict, Optional
import openai
from config import config
from pydantic import BaseModel


class DecisionInput(BaseModel):
    text: str


class ParsedOptions(BaseModel):
    option1: str
    option2: str
    conditions: str


class DecisionResult(BaseModel):
    option1: str
    option2: str
    conditions: str
    decision: str
    reason: str


class LLMService:
    def __init__(self):
        """初始化大模型服务，支持任何OpenAI兼容API"""
        openai.api_key = config.llm.api_key

        if config.llm.api_base:
            openai.api_base = config.llm.api_base

        self.model = config.llm.model
        self.params = config.llm.parameters.dict()
        self.timeout = self.params.pop("timeout")  # 单独处理超时参数

    def parse_options(self, text: str) -> ParsedOptions:
        """使用大模型解析文本中的选项和条件"""
        prompt = f"""
        请分析以下文本，提取出两个选项和相关的客观条件。
        文本: "{text}"

        请按照以下格式返回，只返回JSON，不要添加任何额外内容：
        {{
            "option1": "第一个选项",
            "option2": "第二个选项",
            "conditions": "提取的客观条件"
        }}

        如果无法明确提取两个选项，请尽量合理拆分。
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个文本分析助手，擅长提取选项和条件。"},
                    {"role": "user", "content": prompt}
                ],
                timeout=self.timeout, **self.params
            )

            result = response.choices[0].message.content
            import json
            parsed = json.loads(result)

            return ParsedOptions(
                option1=parsed.get("option1", "选项1"),
                option2=parsed.get("option2", "选项2"),
                conditions=parsed.get("conditions", "")
            )

        except Exception as e:
            print(f"解析选项时出错: {str(e)}")
            return ParsedOptions(
                option1="选项1",
                option2="选项2",
                conditions=text
            )

    def make_decision(self, options: ParsedOptions) -> DecisionResult:
        """使用大模型做出决策"""
        prompt = f"""
        请基于以下两个选项和客观条件，做出一个决策。
        选项1(天使): {options.option1}
        选项2(恶魔): {options.option2}
        客观条件: {options.conditions}

        请按照以下格式返回，只返回JSON，不要添加任何额外内容：
        {{
            "decision": "选择的选项（必须是选项1或选项2的内容）",
            "reason": "做出这个选择的理由，用轻松幽默的卡通风格表达，不要太长"
        }}

        理由需要符合卡通风格，有趣一点，比如：
        - 天使说这个选择更健康哦~
        - 恶魔觉得这个选择更有趣！
        - 虽然有点小任性，但还是选这个吧！
        - 对用户使用一些有趣的称谓，避免过于严肃
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个有趣的决策助手，用卡通风格给出决策建议。"},
                    {"role": "user", "content": prompt}
                ],
                timeout=self.timeout,
                **self.params
            )

            result = response.choices[0].message.content
            import json
            decision = json.loads(result)

            if decision["decision"] not in [options.option1, options.option2]:
                decision["decision"] = options.option1  # 默认选择第一个选项

            return DecisionResult(
                option1=options.option1,
                option2=options.option2,
                conditions=options.conditions,
                decision=decision["decision"],
                reason=decision["reason"]
            )

        except Exception as e:
            print(f"做出决策时出错: {str(e)}")
            import random
            decision = random.choice([options.option1, options.option2])
            return DecisionResult(
                option1=options.option1,
                option2=options.option2,
                conditions=options.conditions,
                decision=decision,
                reason="哎呀，思考时出了点小意外，就选这个吧~"
            )


llm_service = LLMService()
