"""
StoryWeaver 全局配置
====================
切换模型只需修改 LLM_PROVIDER 并实现对应的子类即可。
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ===== DeepSeek API =====
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_PROXY   = os.getenv("DEEPSEEK_PROXY", "")          # e.g. http://127.0.0.1:7897
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL    = "deepseek-chat"

# ===== 游戏节奏 =====
PHASE_1_END = 10    # 第1~10轮：傍晚
PHASE_2_END = 18    # 第11~18轮：黄昏；第19轮起：入夜
NLU_CONFIDENCE_THRESHOLD = 0.35

# ===== 日志 =====
LOG_PATH = "./logs/game_log.jsonl"
