"""
StoryWeaver v4.0 - 豆豆回家记
COMP5423 NLP Group Project

依赖安装：
    pip install openai gradio

运行：
    python storyweaver_v4.py
"""

import os
from dotenv import load_dotenv
load_dotenv()  # 自动读取 .env 文件
import html as html_mod
import json
import re
from typing import Optional, List, Tuple, Dict
import httpx
from openai import OpenAI
import gradio as gr

# ── 配置 ──────────────────────────────────────────────────────────────────────
# 从环境变量读取，启动前执行：export DEEPSEEK_API_KEY="sk-xxxxxx"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
MODEL             = "deepseek-chat"
# 本地代理可选，不设置则直连：export DEEPSEEK_PROXY="http://127.0.0.1:7897"
PROXY             = os.getenv("DEEPSEEK_PROXY", "")

MAX_NPCS  = 3   # 每场景最多NPC数
MAX_ITEMS = 6   # 背包最多物品数

# ── 场景 ──────────────────────────────────────────────────────────────────────
SCENES = [
    {"id": "park",      "name": "城市公园",   "emoji": "🌳"},
    {"id": "market",    "name": "菜市场",     "emoji": "🛒"},
    {"id": "alley",     "name": "老街小巷",   "emoji": "🏮"},
    {"id": "community", "name": "阳光小区",   "emoji": "🏘️"},
    {"id": "reunion",   "name": "与主人重逢", "emoji": "🏠"},
]

SCENE_BACKGROUND = {
    "park": (
        "城市公园，周日下午，人来人往。豆豆刚和主人李叔叔走散，站在喷泉广场。"
        "能隐约闻到李叔叔身上皂香的方向——朝公园东侧飘来。"
        "场景目标：找到公园东侧出口，跟上李叔叔气味的方向。"
        "这里有各种人和动物，有的可以帮助豆豆找到出口，有的可能会耽误时间。"
    ),
    "market": (
        "热闹嘈杂的早市菜市场，充满各种气味。"
        "豆豆顺着皂香往前走，方向是穿过市场进入后面的老街。"
        "场景目标：穿过菜市场找到通往老街的路，最好能弄到一根骨头道具。"
        "这里有动物有人类，有的知道捷径，有的会提供道具，也有危险（城管）。"
    ),
    "alley": (
        "安静的老城区街巷，青石板路，屋檐下挂着风铃。"
        "皂香方向越来越清晰，应该是附近某个小区，但这一带有好几个小区。"
        "场景目标：确认主人所在的小区名称（阳光小区）。"
    ),
    "community": (
        "阳光小区。豆豆顺着气味找到了这里，心跳越来越快。"
        "小区有三栋楼，气味好像来自3号楼，但需要最终确认。"
        "场景目标：确认3号楼，找到主人李叔叔或让他找到自己。"
    ),
    "reunion": "豆豆找到了主人！故事终点。",
}

SCENE_GOAL = {
    "park":      "找到公园东侧出口，跟上李叔叔气味的方向",
    "market":    "穿过菜市场找到通往老街的路，最好弄到一根骨头",
    "alley":     "确认主人所在的小区名称（阳光小区）",
    "community": "确认3号楼，找到主人李叔叔或让他找到自己",
    "reunion":   "与主人重逢",
}

INTENT_LIST = ["INVESTIGATE", "BEFRIEND", "SIGNAL", "TRADE", "EVADE", "TRAIL", "OBSERVE", "USE_ITEM"]
INTENT_LABELS = {
    "INVESTIGATE": "🔍 探索",
    "BEFRIEND":    "🤝 示好",
    "SIGNAL":      "📣 引注意",
    "TRADE":       "🔄 交换",
    "EVADE":       "💨 躲避",
    "TRAIL":       "👣 跟随",
    "OBSERVE":     "👁️ 观察",
    "USE_ITEM":    "🎒 用道具",
}


# ══════════════════════════════════════════════════════════════════════════════
# ██  API 工具  ██
# ══════════════════════════════════════════════════════════════════════════════

_client = None

def get_client():
    global _client
    if _client is None:
        if not DEEPSEEK_API_KEY:
            raise EnvironmentError("请先设置环境变量 DEEPSEEK_API_KEY")
        http_client = httpx.Client(proxy=PROXY) if PROXY else None
        _client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
            http_client=http_client,
        )
    return _client

def _parse_json(raw):
    raw = re.sub(r"```json|```", "", raw).strip()
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
    if not m:
        raise ValueError("无法解析JSON: " + raw[:150])
    return json.loads(m.group())


# ══════════════════════════════════════════════════════════════════════════════
# ██  NLU 模块（独立，可替换）  ██
# ══════════════════════════════════════════════════════════════════════════════

NLU_SYSTEM = (
    "你是NLP意图分类器。判断文字冒险游戏中玩家行动的意图类型。\n"
    "意图：INVESTIGATE(探索环境) | BEFRIEND(示好NPC) | SIGNAL(引起注意) | "
    "TRADE(用物品交换) | EVADE(躲避威胁) | TRAIL(跟随目标) | OBSERVE(等待观察) | USE_ITEM(使用道具)\n"
    "只返回JSON：{\"intent\":\"TYPE\",\"confidence\":0.9}"
)

def classify_intent(option_text, context=None):
    """
    NLU模块：意图识别
    替换说明：把此函数内部换成自己的模型即可，接口不变
    """
    try:
        client = get_client()
        msg = "行动：\"" + option_text + "\""
        if context:
            msg += "\n背景：" + str(context)
        resp = client.chat.completions.create(
            model=MODEL, max_tokens=80,
            messages=[
                {"role": "system", "content": NLU_SYSTEM},
                {"role": "user",   "content": msg},
            ],
        )
        result = _parse_json(resp.choices[0].message.content)
        intent = result.get("intent", "OBSERVE").upper()
        return intent if intent in INTENT_LIST else "OBSERVE"
    except Exception:
        return _fallback_intent(option_text)

def _fallback_intent(text):
    rules = [
        (["嗅","闻","探","查看","检查","翻","找"], "INVESTIGATE"),
        (["跟随","跟着","尾随"],                    "TRAIL"),
        (["蹭","靠近","示好","摇尾","讨好"],         "BEFRIEND"),
        (["叫","吠","汪","吸引","引起"],             "SIGNAL"),
        (["交换","递给","拿出","给他"],              "TRADE"),
        (["躲","逃","绕开","避开","钻"],             "EVADE"),
        (["等","观察","看","静","待"],               "OBSERVE"),
        (["用","使用","取出"],                       "USE_ITEM"),
    ]
    for kws, intent in rules:
        if any(k in text for k in kws):
            return intent
    return "OBSERVE"


# ══════════════════════════════════════════════════════════════════════════════
# ██  NPC生成模块  ██
# ══════════════════════════════════════════════════════════════════════════════

NPC_GEN_SYSTEM = (
    "你是文字冒险游戏NPC生成器。\n\n"
    "世界观：豆豆是会说话的小狗，人类听不懂（只听到汪汪），动物之间可以正常交流。\n\n"
    "根据场景背景生成2-3个NPC，每个需要：\n"
    "name(名字) | type(animal/human) | emoji(一个emoji) | "
    "personality(性格20-40字) | likes(喜好/弱点10-20字) | "
    "hint(能提供什么帮助15-25字) | trust_needed(1-3整数) | "
    "reward_type(clue/item/companion/shortcut/danger) | is_danger(bool)\n\n"
    "要求：至少1个动物NPC；危险NPC不超过1个；性格要有特点。\n\n"
    "只返回JSON数组：\n"
    "[{\"name\":\"x\",\"type\":\"animal\",\"emoji\":\"🐱\",\"personality\":\"x\","
    "\"likes\":\"x\",\"hint\":\"x\",\"trust_needed\":2,\"reward_type\":\"clue\",\"is_danger\":false}]"
)

def generate_npcs(scene_id, scene_name, inventory, clues=None, companions=None):
    client = get_client()
    inv_str  = "、".join(inventory)  if inventory  else "空"
    clue_str = "；".join(clues)      if clues      else "无"
    comp_str = "、".join(companions) if companions else "无"
    user_msg = (
        "【场景】" + scene_name +
        "\n【背景】" + SCENE_BACKGROUND.get(scene_id, "") +
        "\n【豆豆背包】" + inv_str +
        "\n【豆豆已掌握的线索】" + clue_str +
        "\n【同行伙伴】" + comp_str +
        "\n\n请生成NPC。如果线索或伙伴不为空，可以让NPC对豆豆的经历有所耳闻，增强故事连贯感。"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL, max_tokens=1200,
            messages=[
                {"role": "system", "content": NPC_GEN_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
        )
        raw = resp.choices[0].message.content
        raw = re.sub(r"```json|```", "", raw).strip()
        m = re.search(r"\[[\s\S]*\]", raw)
        npcs = json.loads(m.group()) if m else []
        for npc in npcs:
            npc.update({"trust": 0, "turns": 0, "helped": False, "dialogue_history": []})
        return npcs
    except Exception:
        return [{"name":"神秘居民","type":"human","emoji":"👤",
                 "personality":"沉默寡言","likes":"耐心等待",
                 "hint":"也许等一等他会开口","trust_needed":1,
                 "reward_type":"clue","is_danger":False,
                 "trust":0,"turns":0,"helped":False,"dialogue_history":[]}]


# ══════════════════════════════════════════════════════════════════════════════
# ██  叙事生成（NLG）  ██
# ══════════════════════════════════════════════════════════════════════════════

NARRATIVE_SYSTEM = (
    "你是\"豆豆回家记\"文字冒险游戏的叙事引擎。\n\n"
    "世界观：豆豆是会说话的小狗，人类听不懂（只听到汪汪），动物之间正常交流。\n"
    "叙事要体现反差：豆豆对人类说了一大段，人类只是歪头；豆豆和动物流利交流旁人不知道。\n"
    "豆豆内心独白用括号或破折号，要孩子气可爱。第三人称，温暖幽默，2-4句。\n\n"
    "你的职责：\n"
    "1. 根据场景背景、历史剧情和玩家操作生成叙事\n"
    "2. 生成3个行动选项（15-25字，动词开头）：\n"
    "   选项A：主动探索类(INVESTIGATE/TRAIL/USE_ITEM)\n"
    "   选项B：和某个NPC互动类(BEFRIEND/SIGNAL/TRADE)——在npc_target填该NPC名字\n"
    "   选项C：保守观察类(OBSERVE/EVADE)\n"
    "3. 判断canLeave，回合>=7必须true\n"
    "4. 判断玩家本次操作的意图类型（intent）：\n"
    "   INVESTIGATE(探索环境) | BEFRIEND(示好NPC) | SIGNAL(引起注意) | "
    "TRADE(用物品交换) | EVADE(躲避威胁) | TRAIL(跟随目标) | OBSERVE(等待观察) | USE_ITEM(使用道具)\n"
    "   新场景开场白时填 OBSERVE\n\n"
    "只返回JSON：\n"
    "{\"narrative\":\"叙事\",\"options\":[\"A\",\"B\",\"C\"],\"npc_target\":null,\"intent\":\"OBSERVE\","
    "\"stateChanges\":{\"addItem\":null,\"removeItem\":null,\"addClue\":null,\"canLeave\":false}}"
)

def call_narrative(gs, choice):
    client = get_client()
    scene  = SCENES[gs["scene_idx"]]
    sid    = scene["id"]
    inv    = "、".join(gs["inventory"]) if gs["inventory"] else "空"
    clues  = "；".join(gs["clues"])     if gs["clues"]     else "无"
    npcs   = gs.get("scene_npcs", [])
    npc_summary = "；".join(
        n["name"] + "(好感" + str(n["trust"]) + "/" + str(n["trust_needed"]) + "," + ("已帮助" if n.get("helped") else "未帮助") + ")"
        for n in npcs
    ) if npcs else "无"
    recent = gs["history"][-12:]
    role_map = {"narrator": "📖叙事", "player": "▶玩家", "npc": "💬NPC"}
    hist = "\n".join(
        "  " + role_map.get(role, role) + ": " + t
        for role, t in recent
    ) if recent else "  （故事刚开始）"

    lines = [
        "【场景】" + scene["name"],
        "【背景】" + SCENE_BACKGROUND.get(sid, ""),
        "【目标】" + SCENE_GOAL[sid],
        "【回合】" + str(gs["turn"] + 1),
        "【背包】" + inv,
        "【线索】" + clues,
        "【场景NPC】" + npc_summary,
        "【历史】", hist,
    ]
    if gs["turn"] >= 7:
        lines.append("【注意】回合>=7，canLeave必须为true")
    lines.append("")
    lines.append("玩家操作：" + choice if choice else "新场景开始，生成开场白（canLeave:false，intent:OBSERVE）")

    resp = client.chat.completions.create(
        model=MODEL, max_tokens=800,
        messages=[
            {"role": "system", "content": NARRATIVE_SYSTEM},
            {"role": "user",   "content": "\n".join(lines)},
        ],
    )
    return _parse_json(resp.choices[0].message.content)


# ══════════════════════════════════════════════════════════════════════════════
# ██  NPC对话（分动物/人类两个prompt）  ██
# ══════════════════════════════════════════════════════════════════════════════

NPC_SYS_ANIMAL = (
    "你在扮演文字冒险游戏中的动物NPC，与小狗豆豆用动物语言对话。\n"
    "豆豆可以完全说话，动物之间正常交流。\n\n"
    "生成3个豆豆说的话（直接引语风格，15-25字）：\n"
    "A：热情/好奇的话 B：投其所好/使用道具 C：示弱/诉苦\n\n"
    "只返回JSON：\n"
    "{\"npc_dialogue\":\"NPC说的话\",\"options\":[\"豆豆说A\",\"豆豆说B\",\"豆豆说C\"],"
    "\"trust_change\":1,\"dialogue_end\":false,\"reward\":null,\"reward_type\":null}"
)

NPC_SYS_HUMAN = (
    "你在扮演文字冒险游戏中的人类NPC，与小狗豆豆互动。\n"
    "人类完全听不懂豆豆说话，只听到汪汪，会按自己理解误读，制造喜剧效果。\n\n"
    "生成3个豆豆的肢体动作/叫声（不能是说话，15-20字）：\n"
    "A：主动热情的动作 B：使用道具或特殊行动 C：温柔示弱的表情动作\n\n"
    "只返回JSON：\n"
    "{\"npc_dialogue\":\"人类的误解反应\",\"options\":[\"动作A\",\"动作B\",\"动作C\"],"
    "\"trust_change\":1,\"dialogue_end\":false,\"reward\":null,\"reward_type\":null}"
)

def call_npc_dialogue(gs, choice, use_item=None):
    dlg      = gs["dialogue"]
    npc_name = dlg["npc_name"]
    npc_info = dlg.get("npc_info", {})
    inv      = "、".join(gs["inventory"]) if gs["inventory"] else "空"
    npc_type = npc_info.get("type", "human")
    system   = NPC_SYS_ANIMAL if npc_type == "animal" else NPC_SYS_HUMAN

    context_lines = [
        "【NPC】" + npc_name,
        "【性格】" + npc_info.get("personality", "普通NPC"),
        "【喜好/弱点】" + npc_info.get("likes", "未知"),
        "【能提供】" + npc_info.get("hint", "未知"),
        "【好感度】" + str(dlg["trust"]) + "/" + str(npc_info.get("trust_needed", 2) + 1)
        + "（达到" + str(npc_info.get("trust_needed", 2)) + "给奖励）",
        "【奖励类型】" + npc_info.get("reward_type", "clue"),
        "【豆豆背包】" + inv,
    ]
    if dlg["turns"] >= 4:
        context_lines.append("【注意】已超" + str(dlg["turns"]) + "轮，必须dialogue_end:true！")

    context_msg = "\n".join(context_lines)
    history = npc_info.get("dialogue_history", [])

    # 构建多轮消息
    messages = [{"role": "system", "content": system}]
    if not history:
        action_msg = context_msg + "\n\n豆豆刚靠近" + npc_name + "，对话开始，" + npc_name + "先说话。"
        if use_item:
            action_msg += "\n（豆豆准备使用道具：" + use_item + "）"
        messages.append({"role": "user", "content": action_msg})
    else:
        # 把历史直接写进 user 消息，避免伪造 assistant 轮次
        hist_text = "\n".join(
            "豆豆：" + h["player"] + "\n" + npc_name + "：" + h["npc_response"]
            for h in history
        )
        action = choice or ""
        if use_item:
            action += "（使用道具：" + use_item + "）"
        full_msg = context_msg + "\n\n以下是已发生的对话记录：\n" + hist_text + "\n\n豆豆的新行为：" + action
        messages.append({"role": "user", "content": full_msg})

    resp = get_client().chat.completions.create(model=MODEL, max_tokens=600, messages=messages)
    result = _parse_json(resp.choices[0].message.content)

    # 追加历史
    if choice or use_item:
        action_record = choice or ("使用道具：" + use_item)
        npc_info.setdefault("dialogue_history", []).append({
            "player":       action_record,
            "npc_response": result.get("npc_dialogue", ""),
        })
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ██  游戏状态  ██
# ══════════════════════════════════════════════════════════════════════════════

def make_initial_state():
    return {
        "scene_idx":    0,
        "inventory":    [],
        "clues":        [],
        "companions":   [],
        "events":       [],
        "turn":         0,
        "history":      [],
        "over":         False,
        "last_intent":  None,
        # 当前回合的主行动选项
        "action_options": [],
        # UI模式: explore | dialogue | item_action
        "ui_mode":      "explore",
        # 当前选中的NPC（查看详情用）
        "selected_npc": None,
        # 对话状态
        "dialogue": {
            "active":   False,
            "npc_name": None,
            "npc_info": {},
            "trust":    0,
            "turns":    0,
        },
        # 当前选中的物品（物品操作模式用）
        "selected_item": None,
        # 场景NPC列表
        "scene_npcs":   [],
        # 待处理标志
        "pending_advance":    False,
        "pending_npc_start":  False,
        "pending_npc_end":    False,
    }

def apply_narrative_result(gs, result, is_init=False):
    sc = result.get("stateChanges") or {}
    if not is_init:
        gs["turn"] += 1

    if sc.get("addItem") and sc["addItem"] not in gs["inventory"]:
        gs["inventory"].append(sc["addItem"])
    if sc.get("removeItem"):
        gs["inventory"] = [x for x in gs["inventory"] if x != sc["removeItem"]]
    if sc.get("addClue") and sc["addClue"] not in gs["clues"]:
        gs["clues"].append(sc["addClue"])

    # 从叙事结果中提取意图（无需单独 NLU 调用）
    if not is_init:
        intent = str(result.get("intent", "OBSERVE")).upper()
        gs["last_intent"] = intent if intent in INTENT_LIST else "OBSERVE"

    gs["history"].append(("narrator", result["narrative"]))
    gs["action_options"] = result.get("options", [])

    npc_target = result.get("npc_target")
    npc_info   = next((n for n in gs["scene_npcs"] if n["name"] == npc_target), None)

    if npc_target and npc_info and not is_init:
        # 选项B是和某个NPC互动——但现在只是预设，玩家还要点按钮确认
        gs["pending_npc_name"] = npc_target
    else:
        gs["pending_npc_name"] = None

    if sc.get("canLeave") and not is_init:
        # Python层守卫：没有任何线索不允许离开，防止玩家摸鱼通关
        if len(gs["clues"]) == 0:
            gs["history"].append(("narrator", "（还没找到任何线索，豆豆觉得现在离开太早了……）"))
            gs["pending_advance"] = False
        else:
            gs["events"].append("可以前往下一场景")
            gs["pending_advance"] = True
    else:
        gs["pending_advance"] = False

    gs["ui_mode"] = "explore"
    return gs

def apply_dialogue_result(gs, result):
    dlg = gs["dialogue"]
    dlg["turns"] += 1
    dlg["trust"] = max(0, dlg["trust"] + result.get("trust_change", 0))

    for n in gs["scene_npcs"]:
        if n["name"] == dlg["npc_name"]:
            n["trust"] = dlg["trust"]
            break

    gs["history"].append(("npc", "💬 " + dlg["npc_name"] + "：" + result.get("npc_dialogue", "")))

    rtype  = result.get("reward_type")
    reward = result.get("reward")
    npc_info = dlg.get("npc_info", {})
    trust_needed    = npc_info.get("trust_needed", 2)
    trust_sufficient = dlg["trust"] >= trust_needed
    timed_out = (dlg["turns"] >= 4) and not result.get("dialogue_end")

    if result.get("dialogue_end") or dlg["turns"] >= 4:
        if timed_out and not trust_sufficient:
            # 超时且好感不够——失败，不给奖励
            gs["history"].append(("narrator",
                "😔 对话结束了，还没能获得 " + dlg["npc_name"] + " 的信任……也许下次再试试。"))
        else:
            # 正常结束或好感达标——给奖励
            if rtype == "item" and reward:
                # 模糊去重：名字互相包含则视为同一物品
                already = any(reward in x or x in reward for x in gs["inventory"])
                if not already:
                    gs["inventory"].append(reward)
                    gs["history"].append(("narrator", "🎒 获得道具：" + reward))
            elif rtype == "clue" and reward and reward not in gs["clues"]:
                gs["clues"].append(reward)
                gs["history"].append(("narrator", "🔑 获得线索：" + reward))
            elif rtype == "companion" and reward and reward not in gs["companions"]:
                gs["companions"].append(reward)
                gs["history"].append(("narrator", "🐾 " + reward + "决定一起走！"))
            elif rtype == "shortcut" and reward:
                gs["history"].append(("narrator", "🗺️ " + reward))
            elif rtype == "danger":
                # 危险NPC：真实惩罚——额外消耗2回合
                penalty = 2
                gs["turn"] += penalty
                gs["history"].append(("narrator",
                    "😅 " + (reward or "豆豆被耽误了！") + "（浪费了 " + str(penalty) + " 回合……）"))

        for n in gs["scene_npcs"]:
            if n["name"] == dlg["npc_name"]:
                n["helped"] = True
                break

        dlg["active"] = False
        dlg["npc_name"] = None
        gs["dialogue"]  = dlg
        gs["ui_mode"]   = "explore"
        gs["pending_npc_end"] = True
    else:
        gs["action_options"] = result.get("options", [])
        gs["dialogue"] = dlg
        gs["ui_mode"]  = "dialogue"
        gs["pending_npc_end"] = False

    return gs

def advance_scene(gs):
    gs["scene_idx"]        += 1
    gs["turn"]              = 0
    gs["events"]            = []
    gs["last_intent"]       = None
    gs["scene_npcs"]        = []
    gs["selected_npc"]      = None
    gs["selected_item"]     = None
    gs["pending_advance"]   = False
    gs["pending_npc_end"]   = False
    gs["pending_npc_start"] = False
    gs["ui_mode"]           = "explore"
    gs["dialogue"] = {"active": False, "npc_name": None, "npc_info": {}, "trust": 0, "turns": 0}
    if gs["scene_idx"] >= len(SCENES) - 1:
        gs["over"] = True
    return gs


# ══════════════════════════════════════════════════════════════════════════════
# ██  UI 渲染  ██
# ══════════════════════════════════════════════════════════════════════════════

def render_progress(gs):
    parts = []
    for i, s in enumerate(SCENES):
        if i < gs["scene_idx"]:
            parts.append("<span style='color:#16a34a;font-weight:600'>" + s["emoji"] + " " + s["name"] + "</span>")
        elif i == gs["scene_idx"]:
            parts.append("<span style='color:#d97706;font-weight:700;border-bottom:2px solid #d97706'>" + s["emoji"] + " " + s["name"] + "</span>")
        else:
            parts.append("<span style='color:#9ca3af'>" + s["emoji"] + " " + s["name"] + "</span>")
    return "<div style='font-size:13px;padding:6px 0'>" + " → ".join(parts) + "</div>"

def render_npc_panel(gs):
    npcs = gs.get("scene_npcs", [])
    if not npcs:
        return "<div style='color:#9ca3af;font-size:13px;padding:6px'>NPC生成中...</div>"

    selected = gs.get("selected_npc")
    html = []

    # 只渲染详情部分（卡片行已由 gr.Button 展示）
    if not selected:
        return ""

    # 选中NPC详情
    if selected:
        trust   = selected.get("trust", 0)
        needed  = selected.get("trust_needed", 2)
        bar     = "❤️" * trust + "🤍" * max(0, needed - trust)
        t_label = "🐾 动物（豆豆可以直接说话）" if selected.get("type") == "animal" else "👤 人类（只听到汪汪）"
        danger_blk = "<div style='margin-top:5px;padding:3px 7px;background:#fef2f2;border-radius:5px;font-size:11px;color:#dc2626'>⚠️ 危险NPC，互动可能耽误时间</div>" if selected.get("is_danger") else ""

        # 对 LLM 生成的字段转义，防止 HTML 注入
        esc_name        = html_mod.escape(selected.get("name", ""))
        esc_personality = html_mod.escape(selected.get("personality", ""))
        esc_likes       = html_mod.escape(selected.get("likes", ""))
        esc_hint        = html_mod.escape(selected.get("hint", ""))

        hist = selected.get("dialogue_history", [])
        hist_html = ""
        if hist:
            hist_html = "<div style='margin-top:8px;padding-top:6px;border-top:1px solid #e4e2da;font-size:11px'>"
            hist_html += "<div style='color:#6b7280;font-weight:600;margin-bottom:4px'>📜 对话记录</div>"
            for h in hist[-3:]:
                p_text = html_mod.escape(h["player"][:28] + ("…" if len(h["player"]) > 28 else ""))
                n_text = html_mod.escape(h["npc_response"][:28] + ("…" if len(h["npc_response"]) > 28 else ""))
                hist_html += (
                    "<div style='margin-bottom:3px'><span style='color:#d97706'>豆豆：</span>" + p_text + "</div>"
                    "<div style='margin-bottom:5px'><span style='color:#0369a1'>" + esc_name + "：</span>" + n_text + "</div>"
                )
            hist_html += "</div>"

        html.append(
            "<div style='background:#fff;border:1px solid #e4e2da;border-radius:10px;padding:11px;font-size:12px'>"
            "<div style='font-size:18px;margin-bottom:4px'>" + selected["emoji"] + " <b>" + esc_name + "</b></div>"
            "<div style='color:#6b7280;font-size:11px;margin-bottom:7px'>" + t_label + "</div>"
            "<div style='margin-bottom:4px'><b>性格：</b>" + esc_personality + "</div>"
            "<div style='margin-bottom:4px'><b>喜好：</b><span style='color:#d97706'>" + esc_likes + "</span></div>"
            "<div style='margin-bottom:4px'><b>提示：</b><span style='color:#0369a1'>" + esc_hint + "</span></div>"
            "<div style='margin-bottom:3px'><b>好感度：</b>" + bar + " <span style='color:#6b7280'>(" + str(trust) + "/" + str(needed) + ")</span></div>"
            + danger_blk + hist_html +
            "</div>"
        )
    else:
        html.append("<div style='color:#9ca3af;font-size:12px'>👇 下方选择NPC查看详情</div>")

    return "".join(html)

def render_inventory(gs):
    inv = gs.get("inventory", [])
    selected_item = gs.get("selected_item")
    if not inv:
        return "<div style='color:#9ca3af;font-size:13px;padding:4px'>背包空空如也</div>"
    html = ["<div style='display:flex;gap:6px;flex-wrap:wrap'>"]
    for item in inv:
        is_sel = item == selected_item
        bg     = "#fef3c7" if is_sel else "#f9f8f5"
        border = "2px solid #d97706" if is_sel else "1px solid #e4e2da"
        html.append(
            "<div style='padding:5px 9px;background:" + bg + ";border:" + border + ";"
            "border-radius:8px;font-size:12px;font-weight:500;color:#1f2937'>"
            + item + "</div>"
        )
    html.append("</div>")
    if selected_item:
        html.append(
            "<div style='margin-top:6px;font-size:11px;color:#6b7280'>"
            "已选中 <b>" + selected_item + "</b>——下方选择操作</div>"
        )
    return "".join(html)

def render_status(gs):
    clues  = "<br>".join("• " + c for c in gs["clues"])    if gs["clues"]    else "还没有线索"
    comp   = "、".join(gs["companions"])                    if gs["companions"] else "无"
    events = "<br>".join("• " + e for e in gs["events"])   if gs["events"]   else ""
    ev_blk = "<div style='margin-top:8px;padding:5px 8px;background:#f0fdf4;border-radius:6px;font-size:11px;color:#16a34a'>" + events + "</div>" if events else ""

    intent_str = ""
    if gs.get("last_intent") and gs["last_intent"] in INTENT_LABELS:
        intent_str = "<div style='font-size:11px;color:#6b7280;margin-top:3px'>意图: " + INTENT_LABELS[gs["last_intent"]] + "</div>"

    mode_colors = {"explore": "#0369a1", "dialogue": "#d97706", "item_action": "#7c3aed"}
    mode_labels = {"explore": "🗺️ 探索模式", "dialogue": "💬 对话模式", "item_action": "🎒 物品模式"}
    mode = gs.get("ui_mode", "explore")
    mode_color = mode_colors.get(mode, "#6b7280")
    mode_label = mode_labels.get(mode, mode)

    return (
        "<div style='font-size:12px'>"
        "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px'>"
        "<b>回合 " + str(gs["turn"]) + "</b>"
        "<span style='font-size:11px;padding:2px 7px;background:" + mode_color + "20;color:" + mode_color + ";border-radius:12px;font-weight:600'>" + mode_label + "</span>"
        "</div>"
        + intent_str +
        "<div style='margin-top:7px'><b>🐾 同伴：</b>" + comp + "</div>"
        "<div style='margin-top:5px'><b>🔑 线索：</b><br>" + clues + "</div>"
        + ev_blk +
        "</div>"
    )

def render_history(history):
    msgs = []
    for role, text in history:
        if role == "narrator":
            msgs.append((None, "📖 " + text))
        elif role == "npc":
            msgs.append((None, text))
        elif role == "player":
            msgs.append(("▶ " + text, None))
    return msgs

def render_end(gs):
    n_clue = len(gs["clues"])
    n_comp = len(gs["companions"])
    n_item = len(gs["inventory"])

    # 检查是否包含关键线索（阳光小区 / 3号楼）
    clue_text = " ".join(gs["clues"])
    has_key_clue = any(kw in clue_text for kw in ["阳光小区", "3号楼", "三号楼", "小区名", "李叔叔"])

    if has_key_clue and n_clue >= 3 and n_comp >= 1:
        ending = "🏆 最佳结局：主人抱着豆豆泪流满面，同伴们在旁边欢呼！"
    elif has_key_clue and n_clue >= 2:
        ending = "😊 普通结局：豆豆摇着尾巴扑向主人，主人眼眶都红了。"
    elif n_clue >= 1:
        ending = "😅 曲折结局：天都黑了豆豆才找到家，主人又急又心疼。"
    else:
        ending = "😰 迷途结局：豆豆几乎凭运气找到了家，主人心疼地抱了好久……"

    summary = (
        "📊 " + str(n_clue) + "条线索 / "
        + str(n_item) + "件道具 / "
        + str(n_comp) + "位同伴 / "
        + str(gs["turn"]) + "回合"
    )
    return [(None, "🎉 **豆豆找到主人了！**\n\n" + ending + "\n\n" + summary)]


# ══════════════════════════════════════════════════════════════════════════════
# ██  行动按钮更新  ██
# ══════════════════════════════════════════════════════════════════════════════

def _action_updates(gs):
    """根据ui_mode动态更新底部3个行动按钮"""
    mode    = gs.get("ui_mode", "explore")
    options = gs.get("action_options", [])
    updates = []

    if mode == "item_action":
        item = gs.get("selected_item", "道具")
        sel_npc = gs.get("selected_npc")
        labels = [
            "直接使用 " + item,
            "把" + item + "给" + (sel_npc["name"] if sel_npc else "NPC"),
            "❌ 取消",
        ]
        for lbl in labels:
            updates.append(gr.update(value=lbl, visible=True, interactive=True, variant="secondary"))

    elif mode == "explore" and gs.get("pending_advance"):
        # 场景解锁——显示确认离开的按钮，玩家可选择继续探索
        next_idx = min(gs["scene_idx"] + 1, len(SCENES) - 1)
        next_name = SCENES[next_idx]["name"]
        labels = [
            "🐾 前往" + next_name + " →",
            "💡 再和NPC聊聊",
            "🔍 继续探索周围",
        ]
        variants = ["primary", "secondary", "secondary"]
        for lbl, var in zip(labels, variants):
            updates.append(gr.update(value=lbl, visible=True, interactive=True, variant=var))

    else:
        for i in range(3):
            if i < len(options):
                variant = "primary" if mode == "dialogue" and i == 1 else "secondary"
                updates.append(gr.update(value=options[i], visible=True, interactive=True, variant=variant))
            else:
                updates.append(gr.update(value="...", visible=False, interactive=False, variant="secondary"))

    return updates

def _npc_btn_updates(gs):
    """更新NPC交互按钮（最多MAX_NPCS个）"""
    npcs        = gs.get("scene_npcs", [])
    selected    = gs.get("selected_npc")
    dlg         = gs.get("dialogue", {})
    in_dlg      = dlg.get("npc_name") if dlg.get("active") else None
    recommended = gs.get("pending_npc_name")   # 叙事引擎推荐的 NPC
    updates     = []
    for i in range(MAX_NPCS):
        if i < len(npcs):
            npc       = npcs[i]
            helped    = " ✅" if npc.get("helped") else ""
            danger    = " ⚠️" if npc.get("is_danger") else ""
            suggest   = " ⭐" if npc["name"] == recommended and not npc.get("helped") else ""
            type_str  = "🐾动物" if npc["type"] == "animal" else "👤人类"
            hist_cnt  = len(npc.get("dialogue_history", []))
            hist_str  = " 💬" + str(hist_cnt) + "轮" if hist_cnt else ""
            talk_str  = " 对话中" if npc["name"] == in_dlg else ""
            label     = npc["emoji"] + " " + npc["name"] + helped + danger + suggest + "\n" + type_str + hist_str + talk_str
            updates.append(gr.update(value=label, visible=True, interactive=True))
        else:
            updates.append(gr.update(value="", visible=False, interactive=False))
    return updates

def _item_btn_updates(gs):
    """更新背包物品按钮"""
    inv = gs.get("inventory", [])
    updates = []
    for i in range(MAX_ITEMS):
        if i < len(inv):
            updates.append(gr.update(value=inv[i], visible=True, interactive=True))
        else:
            updates.append(gr.update(value="", visible=False, interactive=False))
    return updates

def _all_updates(gs):
    npc_html = render_npc_panel(gs)
    return (
        gs,
        render_progress(gs),
        render_history(gs["history"]),
        gr.update(value=npc_html, visible=bool(gs.get("selected_npc"))),
        render_inventory(gs),
        render_status(gs),
        *_action_updates(gs),
        *_npc_btn_updates(gs),
        *_item_btn_updates(gs),
    )


# ══════════════════════════════════════════════════════════════════════════════
# ██  事件处理  ██
# ══════════════════════════════════════════════════════════════════════════════

def _run_narrative(gs, choice):
    """执行一轮叙事并处理结果"""
    result = call_narrative(gs, choice)
    gs = apply_narrative_result(gs, result, is_init=(choice is None))
    return gs

def _run_npc_dialogue(gs, choice, use_item=None):
    """执行一轮NPC对话并处理结果"""
    result = call_npc_dialogue(gs, choice, use_item=use_item)
    gs = apply_dialogue_result(gs, result)
    return gs

def _enter_npc_dialogue(gs, npc_name):
    """进入与某个NPC的对话"""
    npc_info = next((n for n in gs["scene_npcs"] if n["name"] == npc_name), None)
    if not npc_info:
        return gs
    gs["dialogue"] = {
        "active":   True,
        "npc_name": npc_name,
        "npc_info": npc_info,
        "trust":    npc_info.get("trust", 0),
        "turns":    0,
    }
    gs["ui_mode"] = "dialogue"
    # 触发NPC开场白
    result = call_npc_dialogue(gs, None)
    gs = apply_dialogue_result(gs, result)
    return gs

def _handle_advance(gs):
    """处理场景推进"""
    next_idx = min(gs["scene_idx"] + 1, len(SCENES) - 1)
    gs["history"].append(("narrator", "🐾 豆豆迈开脚步，向" + SCENES[next_idx]["name"] + "走去..."))
    gs = advance_scene(gs)
    if gs["over"]:
        gs["history"] += render_end(gs)
    else:
        scene = SCENES[gs["scene_idx"]]
        gs["scene_npcs"] = generate_npcs(
            scene["id"], scene["name"],
            gs["inventory"], gs["clues"], gs["companions"]
        )
        gs = _run_narrative(gs, None)
    return gs


def on_start(state):
    gs = make_initial_state()
    try:
        scene = SCENES[gs["scene_idx"]]
        gs["scene_npcs"] = generate_npcs(
            scene["id"], scene["name"],
            gs["inventory"], gs["clues"], gs["companions"]
        )
        gs = _run_narrative(gs, None)
    except Exception as e:
        gs["history"].append(("narrator", "⚠️ 启动失败：" + str(e)))
        gs["action_options"] = ["环顾四周，仔细观察", "低头嗅嗅气味", "安静等待"]
    return _all_updates(gs)


def on_action(choice_text, state):
    """玩家点击主行动按钮"""
    gs = state
    if gs.get("over") or not choice_text or choice_text == "...":
        return _all_updates(gs)

    mode = gs.get("ui_mode", "explore")

    try:
        # 物品操作模式
        if mode == "item_action":
            item = gs.get("selected_item", "")
            if "取消" in choice_text:
                gs["selected_item"] = None
                gs["ui_mode"] = "explore"
            elif "直接使用" in choice_text:
                gs["inventory"] = [x for x in gs["inventory"] if x != item]
                gs["history"].append(("player", "使用了" + item))
                gs["selected_item"] = None
                gs = _run_narrative(gs, "豆豆使用了道具：" + item)
                # pending_advance 若被设置，等玩家确认按钮
            elif "给" in choice_text:
                # 用物品和NPC交互
                sel_npc = gs.get("selected_npc")
                npc_name = sel_npc["name"] if sel_npc else None
                if npc_name:
                    gs["history"].append(("player", "把" + item + "给了" + npc_name))
                    # 如果当前在对话中直接用
                    if gs["dialogue"].get("active") and gs["dialogue"]["npc_name"] == npc_name:
                        gs = _run_npc_dialogue(gs, "把" + item + "给你", use_item=item)
                    else:
                        gs = _enter_npc_dialogue(gs, npc_name)
                        if gs["dialogue"].get("active"):
                            gs = _run_npc_dialogue(gs, None, use_item=item)
                    if gs.get("pending_npc_end"):
                        gs["pending_npc_end"] = False
                        gs = _run_narrative(gs, "刚刚结束了NPC对话")
                    gs["inventory"] = [x for x in gs["inventory"] if x != item]
                gs["selected_item"] = None
                gs["ui_mode"] = "dialogue" if gs["dialogue"].get("active") else "explore"

        # 对话模式
        elif mode == "dialogue":
            gs["history"].append(("player", choice_text))
            gs = _run_npc_dialogue(gs, choice_text)
            if gs.get("pending_npc_end"):
                gs["pending_npc_end"] = False
                gs = _run_narrative(gs, "刚刚结束了NPC对话")
                # pending_advance 若被设置，等玩家确认按钮，不自动推进

        # 探索模式
        else:
            # 玩家点击了"前往下一场景"确认按钮
            if gs.get("pending_advance") and choice_text.startswith("🐾 前往"):
                gs = _handle_advance(gs)
            elif gs.get("pending_advance") and ("再和NPC" in choice_text or "继续探索" in choice_text):
                # 玩家选择留下继续探索——清掉 pending_advance，回到正常叙事
                gs["pending_advance"] = False
                gs["history"].append(("player", choice_text))
                gs = _run_narrative(gs, choice_text)
            else:
                gs["history"].append(("player", choice_text))
                gs = _run_narrative(gs, choice_text)
                # 叙事结果里 canLeave=true 时，pending_advance 已在 apply_narrative_result 设好
                # 不在此处自动推进，等玩家点确认按钮

    except Exception as e:
        gs["history"].append(("narrator", "⚠️ 出错了：" + str(e) + "，请重试。"))
        gs["action_options"] = ["再试一次", "等待观察", "四处张望"]
        gs["ui_mode"] = "explore"

    return _all_updates(gs)


def on_npc_btn(npc_label, state):
    """点击NPC按钮——切换选中 / 进入对话"""
    gs = state
    if not npc_label:
        return _all_updates(gs)
    # 标签格式: "emoji 名字\n类型 状态"，取第一行再去掉emoji
    first_line = npc_label.split("\n")[0].strip()
    # 从所有NPC名字里匹配（比字符串解析更可靠）
    npc_info = next((n for n in gs["scene_npcs"] if n["name"] in first_line), None)
    if not npc_info:
        return _all_updates(gs)
    npc_name = npc_info["name"]

    # 如果已选中同一个NPC → 进入对话
    if gs.get("selected_npc") and gs["selected_npc"].get("name") == npc_name:
        if not gs["dialogue"].get("active"):
            try:
                gs["history"].append(("player", "走向" + npc_name + "，打算互动"))
                gs = _enter_npc_dialogue(gs, npc_name)
                if gs.get("pending_npc_end"):
                    gs["pending_npc_end"] = False
                    gs = _run_narrative(gs, "刚刚结束了NPC对话")
            except Exception as e:
                gs["history"].append(("narrator", "⚠️ " + str(e)))
    else:
        # 第一次点击：只展示详情
        gs["selected_npc"] = npc_info

    return _all_updates(gs)


def on_item_btn(item_name, state):
    """点击背包物品——选中进入物品操作模式"""
    gs = state
    if item_name not in gs["inventory"]:
        return _all_updates(gs)
    if gs.get("selected_item") == item_name:
        # 再次点击取消选中
        gs["selected_item"] = None
        gs["ui_mode"] = "dialogue" if gs["dialogue"].get("active") else "explore"
    else:
        gs["selected_item"] = item_name
        gs["ui_mode"] = "item_action"
    return _all_updates(gs)


# ══════════════════════════════════════════════════════════════════════════════
# ██  Gradio UI  ██
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
.action-row button {
    min-height: 64px !important;
    font-size: 13px !important;
    white-space: normal !important;
    line-height: 1.5 !important;
}
.item-btn button {
    font-size: 12px !important;
    min-height: 34px !important;
}
.npc-btn button {
    font-size: 12px !important;
    min-height: 56px !important;
    white-space: normal !important;
    line-height: 1.4 !important;
}
"""

with gr.Blocks(title="豆豆回家记", theme=gr.themes.Soft(), css=CSS) as demo:

    gr.Markdown("# 🐾 豆豆回家记 · StoryWeaver v4.0")
    progress_html = gr.HTML()

    with gr.Row(equal_height=False):
        # ── 左：故事 + 行动 ────────────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="故事", height=460, bubble_full_width=False)
            with gr.Group():
                gr.HTML("<div style='font-size:12px;font-weight:600;color:#6b7280;padding:6px 4px 2px'>行动</div>")
                with gr.Row(elem_classes="action-row"):
                    act0 = gr.Button("...", variant="secondary", visible=False)
                    act1 = gr.Button("...", variant="secondary", visible=False)
                    act2 = gr.Button("...", variant="secondary", visible=False)

        # ── 右：NPC按钮 + 背包 + 状态（不含NPC详情）──────────────────────────
        with gr.Column(scale=1):

            with gr.Group():
                gr.HTML("<div style='font-size:12px;font-weight:600;color:#6b7280;padding:6px 4px 4px'>场景 NPC <span style='font-weight:400;color:#9ca3af'>（点击查看/收起详情）</span></div>")
                with gr.Row(elem_classes="npc-btn"):
                    npc0 = gr.Button("", visible=False)
                    npc1 = gr.Button("", visible=False)
                    npc2 = gr.Button("", visible=False)

            with gr.Group():
                gr.HTML("<div style='font-size:12px;font-weight:600;color:#6b7280;padding:6px 4px 4px'>🎒 背包 <span style='font-weight:400;color:#9ca3af'>（点击物品选择操作）</span></div>")
                with gr.Row(elem_classes="item-btn"):
                    itm0 = gr.Button("", visible=False)
                    itm1 = gr.Button("", visible=False)
                    itm2 = gr.Button("", visible=False)
                with gr.Row(elem_classes="item-btn"):
                    itm3 = gr.Button("", visible=False)
                    itm4 = gr.Button("", visible=False)
                    itm5 = gr.Button("", visible=False)
                inv_panel = gr.HTML(
                    "<div style='color:#9ca3af;font-size:12px;padding:4px'>背包空空如也</div>"
                )

            with gr.Group():
                gr.HTML("<div style='font-size:12px;font-weight:600;color:#6b7280;padding:6px 4px 4px'>状态</div>")
                status_html = gr.HTML()

            start_btn = gr.Button("🔄 开始 / 重置", variant="primary")

    # ── NPC详情：独立一行，有选中时才显示，不影响上方布局 ─────────────────────
    npc_panel = gr.HTML(visible=False)

    # ── 状态 & 输出定义 ────────────────────────────────────────────────────────
    game_state = gr.State(make_initial_state())

    main_outs = [
        game_state, progress_html, chatbot,
        npc_panel, inv_panel, status_html,
        act0, act1, act2,
        npc0, npc1, npc2,
        itm0, itm1, itm2, itm3, itm4, itm5,
    ]

    # ── 事件绑定 ───────────────────────────────────────────────────────────────
    start_btn.click(on_start, inputs=[game_state], outputs=main_outs)
    demo.load(on_start,       inputs=[game_state], outputs=main_outs)

    for btn in [act0, act1, act2]:
        btn.click(on_action, inputs=[btn, game_state], outputs=main_outs)

    for btn in [npc0, npc1, npc2]:
        btn.click(on_npc_btn, inputs=[btn, game_state], outputs=main_outs)

    for btn in [itm0, itm1, itm2, itm3, itm4, itm5]:
        btn.click(on_item_btn, inputs=[btn, game_state], outputs=main_outs)


if __name__ == "__main__":
    demo.launch(share=False)