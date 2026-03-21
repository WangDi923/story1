"""
日志模块
========
每回合记录 NLU 结果、状态变化、NPC 触发等，写入 JSONL 文件。
日志列表同时在 UI 的日志 Tab 实时显示。
"""
import os
import json
import datetime
from typing import List, Dict, Any

import config


def init_log_file():
    os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)


def write_log(entry: Dict[str, Any]):
    """追加写入 JSONL 日志文件"""
    init_log_file()
    entry["timestamp"] = datetime.datetime.now().isoformat()
    try:
        with open(config.LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[Logger Error] {e}")


def format_log_for_display(logs: List[Dict[str, Any]]) -> str:
    """将日志列表格式化为可读文本，用于 UI 显示"""
    if not logs:
        return "暂无日志记录。"
    lines = []
    for entry in logs:
        turn = entry.get("turn", "?")
        player_input = entry.get("player_input", "")
        nlu = entry.get("nlu", {})
        changes = entry.get("state_changes", [])
        preview = entry.get("response_preview", "")

        lines.append(f"{'─' * 50}")
        lines.append(f"📌 第 {turn} 回合")
        lines.append(f"👆 玩家输入：{player_input}")

        if nlu:
            lines.append(
                f"🧠 NLU识别：意图={nlu.get('intent','?')}  "
                f"实体={nlu.get('entities',{})}  "
                f"置信度={nlu.get('confidence','?')}"
            )
        else:
            lines.append("🧠 NLU：（本回合未调用 NLU）")

        if changes:
            lines.append(f"🔄 状态变化：{' | '.join(changes)}")

        lines.append(f"📖 回应预览：{preview}")

    return "\n".join(lines)
