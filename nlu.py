"""
NLU 模块
========
BaseNLU 定义接口，DeepSeekNLU 实现。
后期替换本地模型只需继承 BaseNLU 并实现 classify()。
"""
import json
import httpx
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional
from openai import OpenAI

import config

VALID_INTENTS = [
    "SNIFF",    # 嗅闻/用鼻子探索
    "EXPLORE",  # 移动到某地点
    "EXAMINE",  # 仔细检查某物品
    "APPROACH", # 温柔靠近某人或动物
    "BARK",     # 吠叫（对某个目标）
    "FOLLOW",   # 跟随某人或动物
    "HIDE",     # 躲藏
    "WAIT",     # 休息等待/安静陪伴
    "OBSERVE",  # 主动观察/蹲守
    "GIVE",     # 把物品交给某人/动物
    "UNKNOWN",  # 无法识别
]

_NLU_SYSTEM = """你是文字冒险游戏《寻家记》的NLU模块。
主角是流浪小狗小饼，在不同场景中与NPC互动。

你的任务：将玩家输入分类为意图+实体。

意图列表：
- SNIFF: 嗅闻、用鼻子探索某处
- EXPLORE: 移动/前往/去看某地点
- EXAMINE: 仔细检查/查看某具体物品
- APPROACH: 温柔靠近某人或动物（慢慢走近、蹭、趴在旁边）
- BARK: 吠叫/叫/汪汪（对某个目标）
- FOLLOW: 跟随某人或动物
- HIDE: 躲藏/藏起来/蹲守/埋伏
- WAIT: 休息/等待/安静陪伴/趴着不动
- OBSERVE: 主动盯着/看/注意某个具体目标
- GIVE: 把某物品交给/递给某人或动物
- UNKNOWN: 完全无法识别

场景实体参考（识别时用这些中文名，不用转换ID）：
- 人物：老奶奶、卖气球大爷、卖鱼大叔、面馆老板娘、张叔
- 动物：松鼠、猫头鹰、刺猬/刺猬一家、阿黄/老猫
- 地点：大树、草丛、喷泉、长椅、小路、老橡树、空地、树根窝、存粮处、鱼摊、菜摊、面馆、推车、铁门、房间、门口
- 物品：松果

只返回JSON，不带markdown，格式：
{"intent": "SNIFF", "entities": {"target": "长椅", "location": ""}, "confidence": 0.92}

confidence: 1.0=非常确定, 0.7=比较确定, 0.4=不太确定, 0.2=基本猜测"""


@dataclass
class NLUResult:
    intent: str
    entities: Dict[str, str]
    confidence: float
    raw_text: str


class BaseNLU(ABC):
    @abstractmethod
    def classify(self, text: str, scene_context: str = "") -> NLUResult:
        pass


class DeepSeekNLU(BaseNLU):

    def __init__(self):
        kwargs = {
            "api_key": config.DEEPSEEK_API_KEY,
            "base_url": config.DEEPSEEK_BASE_URL,
        }
        if config.DEEPSEEK_PROXY:
            kwargs["http_client"] = httpx.Client(proxy=config.DEEPSEEK_PROXY)
        self.client = OpenAI(**kwargs)

    def classify(self, text: str, scene_context: str = "") -> NLUResult:
        user_content = f"场景：{scene_context}\n玩家输入：{text}" if scene_context else f"玩家输入：{text}"
        try:
            resp = self.client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": _NLU_SYSTEM},
                    {"role": "user",   "content": user_content},
                ],
                max_tokens=120,
                temperature=0.05,
            )
            raw = resp.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            intent = data.get("intent", "UNKNOWN")
            if intent not in VALID_INTENTS:
                intent = "UNKNOWN"
            return NLUResult(
                intent=intent,
                entities=data.get("entities", {}),
                confidence=float(data.get("confidence", 0.5)),
                raw_text=text,
            )
        except Exception as e:
            print(f"[NLU Error] {e}")
            return NLUResult(intent="UNKNOWN", entities={}, confidence=0.0, raw_text=text)
