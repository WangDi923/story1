"""
任务系统
========
三条主线任务链：松鼠→森林、老奶奶→菜市场、大爷→废墟
场景2改为自由探索：NPC互动设标记，全部完成后触发结局。
"""
from typing import Optional, Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from game_state import GameState


# =====================================================
# 任务定义
# =====================================================
# requires_explored: 场景2最终步骤需要的探索标记（无顺序）

NPC_QUESTS: Dict[str, list] = {
    "squirrel": [
        # --- 公园 ---
        {"step": 1, "scene": "park",   "desc": "靠近松鼠",
         "trigger": "approach_squirrel"},
        {"step": 2, "scene": "park",   "desc": "听松鼠说想吃松果",
         "trigger": "squirrel_dialogue_hungry"},
        {"step": 3, "scene": "park",   "desc": "在草丛里找到松果",
         "trigger": "find_pine_nuts"},
        {"step": 4, "scene": "park",   "desc": "把松果送给松鼠",
         "trigger": "give_pine_nuts",
         "transitions_to": "forest"},
        # --- 森林（自由探索 → 最终触发）---
        {"step": 5, "scene": "forest", "desc": "帮刺猬赶走野猫",
         "trigger": "chase_wildcat",
         "requires_explored": ["owl_talked", "hedgehog_visited"],
         "ending": "forest_home"},
    ],
    "grandma": [
        # --- 公园 ---
        {"step": 1, "scene": "park",   "desc": "靠近老奶奶",
         "trigger": "approach_grandma"},
        {"step": 2, "scene": "park",   "desc": "听老奶奶讲故事",
         "trigger": "listened_story"},
        {"step": 3, "scene": "park",   "desc": "吃饼干 / 蹭腿",
         "trigger": "ate_snack"},
        {"step": 4, "scene": "park",   "desc": "跟老奶奶去菜市场",
         "trigger": "grandma_invite", "affinity_req": 30,
         "transitions_to": "market"},
        # --- 菜市场（自由探索 → 最终触发）---
        {"step": 5, "scene": "market", "desc": "帮老奶奶提菜回家",
         "trigger": "carry_groceries",
         "requires_explored": ["guarded_cart"],
         "ending": "grandma_home"},
    ],
    "vendor": [
        # --- 公园 ---
        {"step": 1, "scene": "park",   "desc": "靠近大爷",
         "trigger": "approach_vendor"},
        {"step": 2, "scene": "park",   "desc": "和大爷聊天",
         "trigger": "vendor_dialogue"},
        {"step": 3, "scene": "park",   "desc": "帮大爷卖气球",
         "trigger": "sell_success"},
        {"step": 4, "scene": "park",   "desc": "跟大爷回家",
         "trigger": "vendor_invite", "affinity_req": 40,
         "transitions_to": "ruins"},
        # --- 废墟（自由探索 → 最终触发）---
        {"step": 5, "scene": "ruins",  "desc": "夜晚安静陪伴大爷",
         "trigger": "night_company",
         "requires_explored": ["befriended_cat"],
         "ending": "ruins_home"},
    ],
}


def get_current_quest(npc: str, state: "GameState") -> Optional[Dict]:
    """获取NPC当前应完成的任务步骤"""
    quests = NPC_QUESTS.get(npc)
    if not quests:
        return None
    current_step = state.npc_quest_step.get(npc, 0)
    for q in quests:
        if q["step"] == current_step + 1:
            return q
    return None  # 全部完成


def check_quest_trigger(npc: str, trigger: str, state: "GameState") -> bool:
    """检查trigger是否匹配当前任务步骤（含好感和探索前置检查）"""
    quest = get_current_quest(npc, state)
    if not quest:
        return False
    if quest["trigger"] != trigger:
        return False
    # 检查好感度要求
    affinity_req = quest.get("affinity_req", 0)
    if state.npc_affinity.get(npc, 0) < affinity_req:
        return False
    # 检查探索前置条件
    for flag in quest.get("requires_explored", []):
        if not state.is_explored(flag):
            return False
    return True


def advance_quest(npc: str, state: "GameState") -> Dict[str, Any]:
    """推进任务，返回效果"""
    quest = get_current_quest(npc, state)
    if not quest:
        return {}

    state.npc_quest_step[npc] = quest["step"]

    effects: Dict[str, Any] = {"advanced": True, "step": quest["step"], "desc": quest["desc"]}

    if "transitions_to" in quest:
        effects["transitions_to"] = quest["transitions_to"]
    if "ending" in quest:
        effects["ending"] = quest["ending"]

    return effects


def get_quest_hint(npc: str, state: "GameState", npc_name: str = None) -> str:
    """返回当前任务提示文本，用于注入NPC prompt
    npc: 主线NPC名
    npc_name: 实际对话的NPC名（场景2 NPC用自己的提示）
    """
    # 场景2 NPC：用自己的角色提示，不用主线quest提示
    if npc_name and npc_name != npc:
        return _get_scene2_npc_hint(npc_name, state)

    quest = get_current_quest(npc, state)
    if not quest:
        return "所有任务已完成。"

    step = quest["step"]

    # 松鼠线提示
    if npc == "squirrel":
        if step == 5:
            # 森林最终步：根据探索进度动态提示
            if not state.is_explored("owl_talked"):
                return "你带小饼来到了森林，这是你的地盘！主动说：'走，我带你去认识猫头鹰大叔！'。"
            if not state.is_explored("hedgehog_visited"):
                return "猫头鹰说了野猫偷粮的事，你主动说：'走，我们去找刺猬阿姨，她家蘑菇被偷了！'"
            return "刺猬需要帮忙赶走野猫，你主动说：'我们去存粮处蹲守，抓住那只野猫！'"
        hints = {
            1: "你在大树下，注意到有只小狗在附近。你有点好奇但还在警惕。",
            2: "你肚子好饿，你想吃松果。主动告诉小饼：'吱！我好饿，草丛那边好像有松果，你能帮我去找找吗？'说完在回复末尾加 [WANTS_NUTS]。",
            3: "你在等小饼帮你找松果，主动提醒它：'去草丛里翻翻嘛，松果就在那里！'",
            4: "小饼叼着松果回来了！你超级开心，接过松果，邀请小饼跟你去森林玩。在回复末尾加 [QUEST_COMPLETE]。",
        }
        return hints.get(step, "")

    # 老奶奶线提示
    if npc == "grandma":
        if step == 5:
            # 菜市场最终步
            if not state.is_explored("guarded_cart"):
                return "你在菜市场买菜，主动对小饼说：'乖孩子，帮奶奶看着推车好不好？奶奶去挑菜。'"
            return "你买完菜了，小饼一路很乖。你想带它回家。在回复末尾加 [QUEST_COMPLETE]。"
        hints = {
            1: "你坐在喷泉边的长椅上，看到一只脏兮兮的小狗在附近晃。你觉得它挺可怜的。",
            2: "小饼靠过来了，你挺喜欢它的。你主动给它讲你以前养的小白的故事。不管小饼做什么友好动作（蹭你、摇尾巴、安静趴着），你都很开心地继续讲。讲完后在回复末尾加 [LISTENED_STORY]。",
            3: "你从包里拿出饼干，主动递给小饼说'来，尝尝奶奶做的饼干'。只要小饼没有咬你或对你叫，就在回复末尾加 [ATE_SNACK]。如果它咬你裤脚或对你叫，在回复末尾加 [BITE_PANTS]。",
            4: "你准备去菜市场买菜了。你挺喜欢这只小狗的，主动招手说：'走吧，跟奶奶去买菜。'在回复末尾加 [QUEST_COMPLETE]。",
        }
        return hints.get(step, "")

    # 大爷线提示
    if npc == "vendor":
        if step == 5:
            # 废墟最终步
            if not state.is_explored("befriended_cat"):
                return "你带小饼回到了你的住处，主动说：'来，我给你介绍一下阿黄，它也住这儿。'"
            return "夜深了，你坐在门口。如果小饼安静地趴在你脚边，你会说些心里话。在回复末尾加 [QUEST_COMPLETE]。"
        hints = {
            1: "你一个人推着气球车，今天生意不好。看到一只小狗在你旁边，你觉得挺有缘的。",
            2: "小狗还在，你跟它聊聊。你叹口气说今天一个气球都没卖出去，然后主动问小饼：'你能不能帮我招招客人呀？站在旁边摇摇尾巴就行。'说完在回复末尾加 [VENDOR_DIALOGUE]。",
            3: "小饼在你旁边做了友好动作（摇尾巴、蹭你、趴着都算），就当它在帮你吸引客人。描写一个小朋友被可爱的小饼吸引过来，买了一个气球。在回复末尾加 [SELL_SUCCESS]。",
            4: "天快黑了，你该收摊了。你主动对小饼说：'走吧，跟我回去，别在外面流浪了。'在回复末尾加 [QUEST_COMPLETE]。",
        }
        return hints.get(step, "")

    return ""


def _get_scene2_npc_hint(npc_name: str, state: "GameState") -> str:
    """场景2 NPC 的角色提示（不暴露主线quest进度）"""
    hints = {
        "owl":         "小饼来到了森林。你知道最近有野猫偷刺猬一家的蘑菇存粮，如果小饼问起可以告诉它。",
        "hedgehog":    "小饼来到了你家附近。你为存粮被偷发愁，希望有人能帮忙在【存粮处】蹲守抓小偷。",
        "fish_vendor": "老奶奶带着一只小狗来买菜了。你是老奶奶的老客户。",
        "noodle_lady": "老奶奶带着一只小狗来了。你心善，想给小狗留点吃的。",
        "old_cat":     "大爷带回来一只小狗，你有点好奇，想凑过去蹭蹭。",
        "zhang":       "大爷家来了一只小狗，你过来串门看看。",
    }
    return hints.get(npc_name, "")


def get_quest_display(npc: str, state: "GameState") -> Optional[str]:
    """返回用于UI显示的任务描述"""
    quest = get_current_quest(npc, state)
    if not quest:
        return None

    # 场景2最终步：显示探索进度
    requires = quest.get("requires_explored", [])
    if requires:
        done = sum(1 for r in requires if state.is_explored(r))
        total = len(requires)
        if done < total:
            scene_names = {"forest": "森林", "market": "菜市场", "ruins": "废墟"}
            scene_name = scene_names.get(quest["scene"], quest["scene"])
            return f"探索{scene_name}（{done}/{total}）"

    return quest["desc"]
