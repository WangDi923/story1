"""
Memory 管理模块
===============
三层 memory 结构统一覆盖 NPC 和小动物。
标签策略：规则自动打标 + 接口预留 AI 兜底。
"""
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from game_state import GameState

NPC_RELEVANT = {
    "grandma": ["grandma_bonded", "grandma_bond_progress", "direction_known", "clues", "phase"],
    "vendor":  ["direction_known", "clues", "phase"],
    "jogger":  ["jogger_warned", "jogger_danger", "danger_resolved", "rescue_failed",
                "jogger_photo", "grandma_bonded", "phase"],
    "cat":     ["cat_ally", "cat_approach_progress", "clues", "phase"],
    "squirrel":["clues", "phase"],
    "pigeons": ["jogger_warned", "phase"],
}


def build_state_summary(npc_name: str, state: "GameState") -> str:
    from game_state import CLUE_META
    relevant = NPC_RELEVANT.get(npc_name, [])
    lines = []

    if "grandma_bonded" in relevant:
        if state.flag_grandma_bonded:
            lines.append("• 小饼已经和你建立了亲近关系，你愿意分享你看到的信息。")
        else:
            lines.append(f"• 小饼还没完全赢得你的信任（进度{state.grandma_bond_progress}/2），"
                         f"不要主动提起年轻男生的事。")

    if "direction_known" in relevant and state.flag_direction_known:
        lines.append("• 小饼已经知道主人去了全家便利店。")

    if "clues" in relevant and state.clues_found:
        clue_names = [CLUE_META.get(c, c) for c in state.clues_found]
        lines.append(f"• 小饼目前已发现的线索：{', '.join(clue_names)}")

    if "phase" in relevant:
        phase_text = {
            1: "傍晚，公园人还比较多。",
            2: "天色渐暗，人开始少了。",
            3: "天快黑了，你要准备离开了。"
        }
        lines.append(f"• 当前时间：{phase_text.get(state.phase, '')}")

    if "jogger_warned" in relevant and state.flag_jogger_warned:
        lines.append("• 你之前已经被小饼惊到过一次，有些不耐烦。")

    if "jogger_danger" in relevant and state.flag_jogger_danger:
        lines.append("• 你正在生气，已经准备打电话叫人。")

    if "danger_resolved" in relevant and state.flag_danger_resolved:
        lines.append("• 小饼刚刚让你消了气，你现在对它态度好了一些。")

    if "jogger_photo" in relevant:
        if state.flag_danger_resolved and not state.flag_jogger_photo_revealed:
            lines.append("• 你手机里有一张刚才拍的公园照片，背景里有个年轻男生。"
                         "如果小饼对你友好，可以拿出来给它看。")
        elif state.flag_jogger_photo_revealed:
            lines.append("• 你已经把手机里的照片展示给小饼看过了。")

    if "cat_ally" in relevant:
        if state.flag_cat_ally:
            lines.append("• 小饼已经和你建立了信任，你们是朋友。")
        else:
            lines.append(f"• 小饼还没完全赢得你的信任（进度{state.cat_approach_progress}/1），保持警惕。")

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
