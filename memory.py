"""
Memory 管理模块
===============
新版：主线好感度驱动，场景2 NPC 挂在主线NPC下。
"""
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from game_state import GameState

from quests import get_quest_hint

# 每个NPC属于哪条主线
NPC_CONTEXT = {
    # 公园NPC（各自是主线）
    "squirrel":     {"mainline": "squirrel", "scene": "park"},
    "grandma":      {"mainline": "grandma",  "scene": "park"},
    "vendor":       {"mainline": "vendor",   "scene": "park"},
    # 森林NPC（挂在松鼠主线）
    "owl":          {"mainline": "squirrel", "scene": "forest"},
    "hedgehog":     {"mainline": "squirrel", "scene": "forest"},
    # 菜市场NPC（挂在老奶奶主线）
    "fish_vendor":  {"mainline": "grandma",  "scene": "market"},
    "noodle_lady":  {"mainline": "grandma",  "scene": "market"},
    # 废墟NPC（挂在大爷主线）
    "old_cat":      {"mainline": "vendor",   "scene": "ruins"},
    "zhang":        {"mainline": "vendor",   "scene": "ruins"},
}


def build_state_summary(npc_name: str, state: "GameState") -> str:
    """构建注入NPC prompt的状态摘要"""
    ctx = NPC_CONTEXT.get(npc_name)
    if not ctx:
        return "• 无特别相关状态。"

    mainline = ctx["mainline"]
    affinity = state.npc_affinity.get(mainline, 0)
    lines = []

    # 1. 好感度描述（影响NPC语气和态度）
    if affinity >= 60:
        lines.append("• 小饼和你（们）关系很好，你对它非常友善和信任。")
    elif affinity >= 40:
        lines.append("• 小饼给你留下了不错的印象，你挺喜欢它的。")
    elif affinity >= 25:
        lines.append("• 小饼表现还行，你对它有好感但还在观察。")
    elif affinity >= 10:
        lines.append("• 小饼刚认识你不久，你对它态度一般。")
    else:
        lines.append("• 你对小饼不太信任，甚至有点不高兴。")

    # 2. 任务上下文（场景2 NPC 用自己的提示）
    hint = get_quest_hint(mainline, state, npc_name=npc_name)
    if hint:
        lines.append(f"• {hint}")

    # 3. 背包信息（仅对需要物品的NPC）
    if npc_name == "squirrel":
        quest_step = state.npc_quest_step.get("squirrel", 0)
        if quest_step == 2:
            lines.append("• 小饼还没找到松果。")

    # 4. 场景轮数
    from game_state import SCENE_META
    scene_meta = SCENE_META.get(state.current_scene, {})
    max_turns = scene_meta.get("max_turns", 20)
    remaining = max(0, max_turns - state.scene_turn_count)
    if remaining <= 3 and state.current_scene == "park":
        lines.append(f"• 天快黑了，时间不多了（剩{remaining}轮）。")

    if not lines:
        lines.append("• 无特别相关状态。")

    return "\n".join(lines)


def get_npc_history(npc_name: str, state: "GameState") -> List[Dict[str, str]]:
    return state.npc_memories.get(npc_name, [])


def append_npc_history(npc_name: str, role: str, content: str, state: "GameState"):
    if npc_name not in state.npc_memories:
        state.npc_memories[npc_name] = []
    state.npc_memories[npc_name].append({"role": role, "content": content})
    if len(state.npc_memories[npc_name]) > 10:
        state.npc_memories[npc_name] = state.npc_memories[npc_name][-10:]
