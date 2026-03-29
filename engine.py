"""
游戏引擎
========
全新版本：三条主线 × 双场景 × 好感度 × 搞砸机制
"""
import random
from typing import Tuple, List, Dict, Any, Optional

import config
from game_state import (
    GameState, normalize_entity, ALL_NPCS, ENDING_META,
    SCENE_META, SCENE_MAINLINE, NPC_NEXT_SCENE, NPC_SCENE2_THRESHOLD,
    SCENE2_FAIL_THRESHOLD,
)
from nlu import DeepSeekNLU, NLUResult
from nlg import DeepSeekNLG, NPC_DISPLAY
import memory as mem
from quests import (
    get_current_quest, check_quest_trigger, advance_quest,
    get_quest_display, NPC_QUESTS,
)

_nlu = DeepSeekNLU()
_nlg = DeepSeekNLG()

_EXIT_PHRASES = ["离开", "再见", "走了", "不说了", "算了", "拜拜", "结束", "走开",
                 "退出", "结束对话", "goodbye", "bye", "quit"]

# 场景2 NPC approach → 设置探索标记（不推进任务，只设flag）
_NPC_EXPLORE_FLAGS = {
    "owl":          ("squirrel", "owl_talked"),
    "hedgehog":     ("squirrel", "hedgehog_visited"),
    "fish_vendor":  ("grandma",  "met_fish_vendor"),
    "noodle_lady":  ("grandma",  "visited_noodle"),
    "old_cat":      ("vendor",   "befriended_cat"),
    "zhang":        ("vendor",   "met_zhang"),
}

OPENING_NARRATION = """🐾 傍晚的阳光斜斜地打在公园的石板路上。

小饼从长椅下面钻出来，抖了抖身上的灰。又是普通的一天。肚子有点饿，身上有点脏，但公园里的阳光暖暖的。

大树下面，一只【小松鼠】在忙着什么，不时回头看它一眼。喷泉边的长椅上，一个【老奶奶】正在晒太阳，膝上放着布包。公园小路旁，一个【卖气球的大爷】靠在气球车上发呆，五颜六色的气球在风里晃。

小饼站起来，摇了摇尾巴。今天会有什么不一样的事发生吗？"""


# =====================================================
# 游戏入口
# =====================================================

def new_game() -> Tuple[GameState, List[str]]:
    state = GameState()
    return state, get_suggestions(state)


def process_turn(
    player_input: str,
    state: GameState,
) -> Tuple[str, GameState, List[str], Dict[str, Any]]:
    log: Dict[str, Any] = {
        "turn": state.total_turn_count + 1,
        "scene": state.current_scene,
        "player_input": player_input,
        "nlu": {},
        "intent_handled": "",
        "state_changes": [],
        "response_preview": "",
    }

    if state.game_over:
        return "游戏已经结束了，请点击「重新开始」再玩一次！", state, [], log

    # 回合推进
    state.total_turn_count += 1
    state.scene_turn_count += 1

    response = ""

    # ===== 优先级 1：观察模式结果 =====
    if state.observe_target:
        target = state.observe_target
        state.observe_target = None
        nlu_result = _nlu.classify(player_input, _scene_context(state))
        log["nlu"] = _nlu_log(nlu_result)
        response, state = _handle_observe_result(target, state, log)
        log["intent_handled"] = f"OBSERVE_RESULT:{target}"

    # ===== 优先级 2：对话模式 =====
    elif state.in_dialogue:
        if _wants_exit(player_input):
            npc = state.dialogue_target
            state.in_dialogue = False
            state.dialogue_target = None
            display = NPC_DISPLAY.get(npc, npc)
            response = f"小饼向【{display}】点点头，转身走开了。"
            log["intent_handled"] = "EXIT_DIALOGUE"
        else:
            nlu_result = _nlu.classify(player_input, _scene_context(state))
            log["nlu"] = _nlu_log(nlu_result)
            response, state = _handle_dialogue(player_input, nlu_result, state, log)
            log["intent_handled"] = f"DIALOGUE:{state.dialogue_target}"

    # ===== 优先级 3：普通意图 =====
    else:
        nlu_result = _nlu.classify(player_input, _scene_context(state))
        log["nlu"] = _nlu_log(nlu_result)
        state.last_nlu_log = log["nlu"]
        response, state = _dispatch_intent(nlu_result, player_input, state, log)
        log["intent_handled"] = nlu_result.intent

    # ===== 场景2搞砸检测 =====
    if not state.game_over and state.current_scene != "park":
        fail_threshold = SCENE2_FAIL_THRESHOLD.get(state.current_scene, 0)
        mainline_aff = state.get_mainline_affinity()
        if mainline_aff < fail_threshold:
            scene = state.current_scene
            label = f"{scene}_kicked_out"
            kick_text = _nlg.generate_narration(label, player_input, state)
            response += f"\n\n{'─'*40}\n\n{kick_text}"
            state.ending = "stray"
            state.game_over = True
            ending_text = _nlg.generate_ending("stray", state)
            response += f"\n\n{ending_text}"
            log["state_changes"].append(f"SCENE2_FAIL: affinity={mainline_aff} < {fail_threshold}")

    # ===== 公园轮数耗尽检测 =====
    if not state.game_over and state.current_scene == "park":
        max_turns = SCENE_META["park"]["max_turns"]
        if state.scene_turn_count >= max_turns:
            # 找好感度最高且达标的NPC
            best_npc = _find_best_npc(state)
            if best_npc:
                # 触发场景切换
                transition_resp = _do_scene_transition(best_npc, state, log)
                response += f"\n\n{'─'*40}\n\n{transition_resp}"
            else:
                # 无人达标 → 流浪结局
                state.ending = "stray"
                state.game_over = True
                ending_text = _nlg.generate_ending("stray", state)
                response += f"\n\n{'─'*40}\n\n{ending_text}"
                log["state_changes"].append("PARK_TIMEOUT: stray ending")

    # ===== 场景2轮数耗尽检测 =====
    if not state.game_over and state.current_scene != "park":
        max_turns = SCENE_META.get(state.current_scene, {}).get("max_turns", 12)
        if state.scene_turn_count >= max_turns:
            # 检查是否完成了最终任务
            mainline_npc = state.get_mainline_npc()
            quest = get_current_quest(mainline_npc, state)
            if quest and quest.get("ending"):
                # 差一步就完成了，给个机会自动触发
                pass
            else:
                state.ending = "stray"
                state.game_over = True
                ending_text = _nlg.generate_ending("stray", state)
                response += f"\n\n{'─'*40}\n\n{ending_text}"
                log["state_changes"].append("SCENE2_TIMEOUT: stray ending")

    log["response_preview"] = response[:120].replace("\n", " ")
    return response, state, get_suggestions(state), log


# =====================================================
# 意图分发
# =====================================================

def _dispatch_intent(nlu: NLUResult, raw: str, state: GameState,
                     log: Dict) -> Tuple[str, GameState]:
    intent = nlu.intent
    entities = nlu.entities
    target_raw = entities.get("target", "") or entities.get("location", "")
    target = normalize_entity(target_raw)

    if nlu.confidence < config.NLU_CONFIDENCE_THRESHOLD and intent != "UNKNOWN":
        intent = "UNKNOWN"

    # ===== 搞砸行为关键词检测（NLU可能识别不出来） =====
    mischief = _check_mischief(raw, state, log)
    if mischief:
        return mischief

    # NPC目标 → 检查是否在当前场景
    scene_npcs = SCENE_META.get(state.current_scene, {}).get("npcs", [])
    # 主线NPC在场景2也存在（老奶奶在菜市场、大爷在废墟等）
    mainline_npc = state.get_mainline_npc()
    available_npcs = list(scene_npcs)
    if mainline_npc and mainline_npc not in available_npcs:
        available_npcs.append(mainline_npc)

    if target in ALL_NPCS:
        if target not in available_npcs:
            return _nlg.generate_narration("npc_not_here", raw, state), state
        if intent in ("EXPLORE", "APPROACH", "FOLLOW", "BARK"):
            if intent == "BARK":
                return _handle_bark(target, raw, state, log)
            return _enter_npc_dialogue(target, raw, state, log)
        if intent == "GIVE":
            return _handle_give(target, raw, state, log)

    handlers = {
        "SNIFF":    _handle_sniff,
        "EXPLORE":  _handle_explore,
        "EXAMINE":  _handle_examine,
        "APPROACH": _handle_approach,
        "BARK":     _handle_bark,
        "FOLLOW":   _handle_follow,
        "HIDE":     _handle_hide,
        "WAIT":     _handle_wait,
        "OBSERVE":  _handle_observe,
        "GIVE":     _handle_give,
        "UNKNOWN":  _handle_unknown,
    }
    return handlers.get(intent, _handle_unknown)(target, raw, state, log)


# =====================================================
# 进入NPC对话
# =====================================================

def _enter_npc_dialogue(npc: str, raw: str, state: GameState,
                        log: Dict) -> Tuple[str, GameState]:
    changes = []

    # 好感度：首次靠近
    mainline = _get_mainline_for_npc(npc)
    if mainline and not state.is_explored(f"approach_{npc}"):
        state.mark_explored(f"approach_{npc}")
        delta = state.adjust_affinity(mainline, "approach")
        if delta:
            changes.append(f"affinity[{mainline}] += {delta}")

    # 任务推进：approach 触发（公园主线NPC）
    _try_quest_advance(mainline, f"approach_{npc}", state, changes)

    # 松鼠线：背包有松果 → 自动触发给松果
    _pine_nut_transition = None
    if npc == "squirrel" and "pine_nuts" in state.inventory:
        if check_quest_trigger("squirrel", "give_pine_nuts", state):
            state.inventory.remove("pine_nuts")
            delta = state.adjust_affinity("squirrel", "give_pine_nuts")
            if delta:
                changes.append(f"affinity[squirrel] += {delta}")
            effects = advance_quest("squirrel", state)
            changes.append(f"quest[squirrel] → step {effects.get('step')}")
            changes.append("inventory -= pine_nuts")
            if effects.get("transitions_to"):
                _pine_nut_transition = True

    # 场景2 NPC：设置探索标记 + 好感度
    if npc in _NPC_EXPLORE_FLAGS:
        explore_mainline, explore_flag = _NPC_EXPLORE_FLAGS[npc]
        if not state.is_explored(explore_flag):
            state.mark_explored(explore_flag)
            changes.append(f"explored: {explore_flag}")
            # 特定NPC首次互动给额外好感
            affinity_actions = {
                "old_cat":   ("vendor",   "befriend_cat"),
                "zhang":     ("vendor",   "zhang_visit"),
                "hedgehog":  ("squirrel", "visit_hedgehog"),
            }
            if npc in affinity_actions:
                m, a = affinity_actions[npc]
                delta = state.adjust_affinity(m, a)
                if delta:
                    changes.append(f"affinity[{m}] += {delta}")

    state.in_dialogue = True
    state.dialogue_target = npc
    npc_resp, flags = _nlg.generate_npc_response(npc, raw, state)
    quest_response = _process_quest_flags(npc, flags, state, changes)
    mem.append_npc_history(npc, "user", raw, state)
    mem.append_npc_history(npc, "assistant", npc_resp, state)

    if flags.get("end_chat"):
        state.in_dialogue = False
        state.dialogue_target = None

    log["state_changes"].extend(changes)
    display = NPC_DISPLAY.get(npc, npc)
    result = f"💬 【{display}】{npc_resp}"

    # 如果任务推进触发了场景切换或结局
    if quest_response:
        result += f"\n\n{quest_response}"

    # 松鼠线：给完松果后的场景切换（对话结束后单独成段）
    if _pine_nut_transition:
        state.in_dialogue = False
        state.dialogue_target = None
        transition_resp = _do_scene_transition("squirrel", state, log)
        result += f"\n\n{'─'*40}\n\n{transition_resp}"

    return result, state


# =====================================================
# 意图处理函数
# =====================================================

def _handle_sniff(target: str, raw: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    changes = []

    if target == "bushes" and state.current_scene == "park":
        # 松鼠任务：找松果 → 加入背包（需要回去找松鼠给它）
        mainline = "squirrel"
        if check_quest_trigger(mainline, "find_pine_nuts", state):
            effects = advance_quest(mainline, state)
            changes.append(f"quest[squirrel] → step {effects.get('step')}")
            state.inventory.append("pine_nuts")
            changes.append("inventory += pine_nuts")
            log["state_changes"].extend(changes)
            return _nlg.generate_narration("find_pine_nuts", raw, state), state

        if not state.is_explored("bushes_sniff"):
            state.mark_explored("bushes_sniff")
            log["state_changes"].extend(changes)
            return _nlg.generate_narration("sniff_bushes_new", raw, state), state
        else:
            return _nlg.generate_narration("sniff_bushes_again", raw, state), state

    if target == "stash" and state.current_scene == "forest":
        # 偷吃存粮 → 搞砸
        delta = state.adjust_affinity("squirrel", "steal_stash")
        if delta:
            changes.append(f"affinity[squirrel] += {delta}")
        log["state_changes"].extend(changes)
        return _nlg.generate_narration("forest_steal_stash", raw, state), state

    label = f"sniff_{target}" if target else "unknown_action"
    state.mark_explored(target)
    log["state_changes"].extend(changes)
    return _nlg.generate_narration(label, raw, state), state


def _handle_explore(target: str, raw: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    changes = []
    scene = state.current_scene

    # 场景感知标签映射
    label_map = {
        ("park", "tree"):       "explore_tree",
        ("park", "bushes"):     "explore_bushes",
        ("park", "fountain"):   "explore_fountain",
        ("park", "bench"):      "explore_bench",
        ("park", "path"):       "explore_path",
        ("forest", "oak_tree"):  "forest_explore_oak",
        ("forest", "clearing"):  "forest_explore_clearing",
        ("forest", "burrow"):    "forest_explore_burrow",
        ("forest", "stash"):     "forest_explore_stash",
        ("market", "fish_stall"): "market_explore_fish",
        ("market", "veggie_stall"): "market_explore_veggie",
        ("market", "noodle_shop"):  "market_explore_noodle",
        ("market", "cart"):      "market_explore_cart",
        ("ruins", "iron_gate"):  "ruins_explore_gate",
        ("ruins", "room"):       "ruins_explore_room",
        ("ruins", "doorstep"):   "ruins_explore_doorstep",
    }

    label = label_map.get((scene, target), f"explore_{target}" if target else "unknown_action")

    # 公园草丛：也能触发找松果 → 加入背包
    if scene == "park" and target == "bushes":
        if check_quest_trigger("squirrel", "find_pine_nuts", state):
            effects = advance_quest("squirrel", state)
            changes.append(f"quest[squirrel] → step {effects.get('step')}")
            state.inventory.append("pine_nuts")
            changes.append("inventory += pine_nuts")
            log["state_changes"].extend(changes)
            return _nlg.generate_narration("find_pine_nuts", raw, state), state

    # NPC目标 → 进入对话
    if target in ALL_NPCS:
        scene_npcs = SCENE_META.get(scene, {}).get("npcs", [])
        mainline_npc = state.get_mainline_npc()
        available = list(scene_npcs) + ([mainline_npc] if mainline_npc else [])
        if target in available:
            return _enter_npc_dialogue(target, raw, state, log)
        else:
            return _nlg.generate_narration("npc_not_here", raw, state), state

    # 森林存粮处：可触发赶野猫（需要前置探索完成）
    if scene == "forest" and target == "stash":
        # 探索存粮处也可以触发赶野猫（如果任务到了这步）
        if check_quest_trigger("squirrel", "chase_wildcat", state):
            delta = state.adjust_affinity("squirrel", "chase_wildcat")
            if delta:
                changes.append(f"affinity[squirrel] += {delta}")
            effects = advance_quest("squirrel", state)
            changes.append(f"quest[squirrel] → step {effects.get('step')}")
            log["state_changes"].extend(changes)
            narration = _nlg.generate_narration("forest_chase_wildcat", raw, state)
            if effects.get("ending"):
                aff = state.npc_affinity.get("squirrel", 0)
                fail_threshold = SCENE2_FAIL_THRESHOLD.get("forest", 20)
                if aff >= fail_threshold:
                    state.ending = effects["ending"]
                    state.game_over = True
                    ending_text = _nlg.generate_ending(effects["ending"], state)
                    narration += f"\n\n{'─'*40}\n\n{ending_text}"
            return narration, state

    state.mark_explored(target)
    log["state_changes"].extend(changes)
    return _nlg.generate_narration(label, raw, state), state


def _handle_examine(target: str, raw: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    label = f"examine_{target}" if target else "unknown_action"
    state.mark_explored(target)
    return _nlg.generate_narration(label, raw, state), state


def _handle_approach(target: str, raw: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    if target in ALL_NPCS:
        return _enter_npc_dialogue(target, raw, state, log)
    return _nlg.generate_narration(f"approach_{target}", raw, state), state


def _handle_bark(target: str, raw: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    changes = []
    mainline = _get_mainline_for_npc(target) if target in ALL_NPCS else None

    # 吠叫通常是负面行为
    if target in ALL_NPCS and mainline:
        action = "bark_at" if state.current_scene == "park" else f"bark_{target}"
        # 场景2特殊负面标签
        bark_actions = {
            ("forest", "owl"):    ("squirrel", "bark_owl"),
            ("market", ""):       ("grandma", "bark_passerby"),
            ("ruins", "zhang"):   ("vendor", "bark_zhang"),
        }
        for (s, t), (m, a) in bark_actions.items():
            if state.current_scene == s and (t == "" or t == target):
                mainline, action = m, a
                break

        delta = state.adjust_affinity(mainline, action)
        if delta:
            changes.append(f"affinity[{mainline}] += {delta}")

    scene = state.current_scene
    bark_labels = {
        ("forest", "owl"):    "forest_bark_owl",
        ("ruins", "zhang"):   "ruins_bark_zhang",
    }
    label = bark_labels.get((scene, target), f"bark_{target}")

    log["state_changes"].extend(changes)
    return _nlg.generate_narration(label, raw, state), state


def _handle_follow(target: str, raw: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    if target in ALL_NPCS:
        scene_npcs = SCENE_META.get(state.current_scene, {}).get("npcs", [])
        mainline_npc = state.get_mainline_npc()
        available = list(scene_npcs) + ([mainline_npc] if mainline_npc else [])
        if target in available:
            # 菜市场：跟着老奶奶 = 帮提菜
            if state.current_scene == "market" and target == "grandma":
                changes = []
                _try_quest_advance("grandma", "carry_groceries", state, changes)
                log["state_changes"].extend(changes)
            return _enter_npc_dialogue(target, raw, state, log)
    return _nlg.generate_narration(f"follow_{target}", raw, state), state


def _handle_hide(target: str, raw: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    changes = []
    # 森林：蹲守存粮处 = 赶野猫
    if state.current_scene == "forest" and target in ("stash", "burrow", ""):
        if check_quest_trigger("squirrel", "chase_wildcat", state):
            delta = state.adjust_affinity("squirrel", "chase_wildcat")
            if delta:
                changes.append(f"affinity[squirrel] += {delta}")
            effects = advance_quest("squirrel", state)
            changes.append(f"quest[squirrel] → step {effects.get('step')}")
            log["state_changes"].extend(changes)

            narration = _nlg.generate_narration("forest_chase_wildcat", raw, state)
            # 检查是否触发结局
            if effects.get("ending"):
                ending = effects["ending"]
                aff = state.npc_affinity.get("squirrel", 0)
                fail_threshold = SCENE2_FAIL_THRESHOLD.get("forest", 20)
                if aff >= fail_threshold:
                    state.ending = ending
                    state.game_over = True
                    ending_text = _nlg.generate_ending(ending, state)
                    narration += f"\n\n{'─'*40}\n\n{ending_text}"
                    changes.append(f"ENDING: {ending}")
            return narration, state

    return _nlg.generate_narration(f"hide_{target}", raw, state), state


def _handle_wait(target: str, raw: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    changes = []
    scene = state.current_scene

    # 菜市场：看推车 = 守推车（设探索标记）—— target可以是cart、grandma或空
    if scene == "market" and target in ("cart", "grandma", ""):
        if not state.is_explored("guarded_cart"):
            state.mark_explored("guarded_cart")
            delta = state.adjust_affinity("grandma", "guard_cart")
            if delta:
                changes.append(f"affinity[grandma] += {delta}")
            changes.append("explored: guarded_cart")
            log["state_changes"].extend(changes)
            return _nlg.generate_narration("market_guard_cart", raw, state), state

    # 废墟：夜晚安静陪伴
    if scene == "ruins" and target in ("", "doorstep", "vendor"):
        if check_quest_trigger("vendor", "night_company", state):
            delta = state.adjust_affinity("vendor", "night_company")
            if delta:
                changes.append(f"affinity[vendor] += {delta}")
            effects = advance_quest("vendor", state)
            changes.append(f"quest[vendor] → step {effects.get('step')}")
            log["state_changes"].extend(changes)

            narration = _nlg.generate_narration("ruins_night", raw, state)
            if effects.get("ending"):
                ending = effects["ending"]
                aff = state.npc_affinity.get("vendor", 0)
                fail_threshold = SCENE2_FAIL_THRESHOLD.get("ruins", 20)
                if aff >= fail_threshold:
                    state.ending = ending
                    state.game_over = True
                    ending_text = _nlg.generate_ending(ending, state)
                    narration += f"\n\n{'─'*40}\n\n{ending_text}"
                    changes.append(f"ENDING: {ending}")
            return narration, state

    # 废墟：不安分
    if scene == "ruins":
        delta = state.adjust_affinity("vendor", "restless_night")
        if delta:
            changes.append(f"affinity[vendor] += {delta}")
            log["state_changes"].extend(changes)
            return _nlg.generate_narration("ruins_restless", raw, state), state

    log["state_changes"].extend(changes)
    return _nlg.generate_narration("wait_rest" if not target else f"wait_{target}", raw, state), state


def _handle_observe(target: str, raw: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    # 菜市场：观察推车/老奶奶 → 当作守推车处理
    if state.current_scene == "market" and target in ("cart", "grandma", ""):
        if not state.is_explored("guarded_cart"):
            return _handle_wait(target, raw, state, log)

    # 废墟：观察门口/大爷 → 当作安静陪伴
    if state.current_scene == "ruins" and target in ("doorstep", "vendor", ""):
        return _handle_wait(target, raw, state, log)

    state.observe_target = target or raw
    log["state_changes"].append(f"observe_target={state.observe_target}")
    return _nlg.generate_narration(f"observe_{target}", raw, state), state


def _handle_observe_result(target: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    return _nlg.generate_narration(f"observe_result_{target}", f"观察{target}", state), state


def _handle_give(target: str, raw: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    # 给松鼠松果：进入对话触发
    if target == "squirrel" and "pine_nuts" in state.inventory:
        return _enter_npc_dialogue("squirrel", raw, state, log)
    if target in ALL_NPCS:
        return _enter_npc_dialogue(target, raw, state, log)
    return _nlg.generate_narration("unknown_action", raw, state), state


def _handle_unknown(target: str, raw: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    return _nlg.generate_narration("unknown_action", raw, state), state


# =====================================================
# 对话模式
# =====================================================

def _handle_dialogue(raw: str, nlu: NLUResult, state: GameState,
                     log: Dict) -> Tuple[str, GameState]:
    npc = state.dialogue_target
    if not npc:
        state.in_dialogue = False
        return "小饼四处张望了一下。", state

    # EXPLORE 指向非NPC → 退出对话
    target = normalize_entity(nlu.entities.get("target", ""))
    if nlu.intent in ("EXPLORE", "HIDE") and target not in ALL_NPCS:
        state.in_dialogue = False
        state.dialogue_target = None
        display = NPC_DISPLAY.get(npc, npc)
        return f"小饼向【{display}】点点头，转身走开了。", state

    # GIVE 意图在对话中
    if nlu.intent == "GIVE":
        # 松鼠对话中给松果：移除背包，让正常对话流程处理（NPC会生成[QUEST_COMPLETE]）
        if npc == "squirrel" and "pine_nuts" in state.inventory:
            state.inventory.remove("pine_nuts")
            # 继续走正常对话流程
        else:
            give_target = target if target in ALL_NPCS else npc
            state.in_dialogue = False
            state.dialogue_target = None
            return _handle_give(give_target, raw, state, log)

    changes = []

    # 对话中的友好互动给好感度（摇尾巴、蹭腿、安静趴着等）
    mainline = _get_mainline_for_npc(npc)
    if mainline and nlu.intent in ("APPROACH", "WAIT", "UNKNOWN"):
        delta = state.adjust_affinity(mainline, "dialogue")
        if delta:
            changes.append(f"affinity[{mainline}] += {delta}")

    npc_resp, flags = _nlg.generate_npc_response(npc, raw, state)
    quest_response = _process_quest_flags(npc, flags, state, changes)
    mem.append_npc_history(npc, "user", raw, state)
    mem.append_npc_history(npc, "assistant", npc_resp, state)

    if flags.get("end_chat"):
        state.in_dialogue = False
        state.dialogue_target = None

    log["state_changes"].extend(changes)
    display = NPC_DISPLAY.get(npc, npc)
    result = f"💬 【{display}】{npc_resp}"

    if quest_response:
        result += f"\n\n{quest_response}"

    return result, state


# =====================================================
# 任务flag处理
# =====================================================

def _process_quest_flags(npc: str, flags: Dict, state: GameState,
                         changes: List[str]) -> str:
    """处理NPC对话中的任务标记，返回额外叙述"""
    mainline = _get_mainline_for_npc(npc)
    if not mainline:
        return ""

    extra = ""

    # 松鼠：想吃松果
    if flags.get("wants_nuts"):
        _try_quest_advance(mainline, "squirrel_dialogue_hungry", state, changes)

    # 老奶奶：听故事
    if flags.get("listened_story"):
        delta = state.adjust_affinity("grandma", "listen_story")
        if delta:
            changes.append(f"affinity[grandma] += {delta}")
        _try_quest_advance("grandma", "listened_story", state, changes)

    # 老奶奶：吃饼干
    if flags.get("ate_snack"):
        delta = state.adjust_affinity("grandma", "eat_snack")
        if delta:
            changes.append(f"affinity[grandma] += {delta}")
        _try_quest_advance("grandma", "ate_snack", state, changes)

    # 老奶奶：咬裤脚
    if flags.get("bite_pants"):
        delta = state.adjust_affinity("grandma", "bite_pants")
        if delta:
            changes.append(f"affinity[grandma] += {delta}")

    # 大爷：对话推进（主动邀请帮卖气球）
    if flags.get("vendor_dialogue"):
        _try_quest_advance("vendor", "vendor_dialogue", state, changes)

    # 大爷：帮卖气球
    if flags.get("help_sell"):
        delta = state.adjust_affinity("vendor", "help_sell")
        if delta:
            changes.append(f"affinity[vendor] += {delta}")

    # 大爷：卖出气球
    if flags.get("sell_success"):
        delta = state.adjust_affinity("vendor", "sell_success")
        if delta:
            changes.append(f"affinity[vendor] += {delta}")
        _try_quest_advance("vendor", "sell_success", state, changes)

    # 通用：QUEST_COMPLETE
    if flags.get("quest_complete"):
        # 尝试各种可能的触发
        for trigger in ["give_pine_nuts", "grandma_invite", "vendor_invite",
                        "carry_groceries", "night_company"]:
            if check_quest_trigger(mainline, trigger, state):
                # 给松果时加好感度
                if trigger == "give_pine_nuts":
                    delta = state.adjust_affinity("squirrel", "give_pine_nuts")
                    if delta:
                        changes.append(f"affinity[squirrel] += {delta}")
                effects = advance_quest(mainline, state)
                changes.append(f"quest[{mainline}] → step {effects.get('step')}")

                if effects.get("transitions_to"):
                    extra = f"{'─'*40}\n\n" + _do_scene_transition(mainline, state, {"state_changes": changes})

                if effects.get("ending"):
                    ending = effects["ending"]
                    npc_key = mainline
                    aff = state.npc_affinity.get(npc_key, 0)
                    scene = state.current_scene
                    fail_threshold = SCENE2_FAIL_THRESHOLD.get(scene, 0)
                    if aff >= fail_threshold:
                        state.ending = ending
                        state.game_over = True
                        state.in_dialogue = False
                        state.dialogue_target = None
                        ending_text = _nlg.generate_ending(ending, state)
                        extra = f"{'─'*40}\n\n{ending_text}"
                        changes.append(f"ENDING: {ending}")
                break

    return extra


# =====================================================
# 场景切换
# =====================================================

def _do_scene_transition(npc: str, state: GameState, log: Dict) -> str:
    from_scene = state.current_scene
    to_scene = NPC_NEXT_SCENE.get(npc, "")
    if not to_scene:
        return ""

    transition_text = _nlg.generate_scene_transition(from_scene, to_scene, npc, state)

    # 切换场景
    state.current_scene = to_scene
    state.scene_turn_count = 0
    state.in_dialogue = False
    state.dialogue_target = None
    state.observe_target = None

    if isinstance(log, dict) and "state_changes" in log:
        log["state_changes"].append(f"SCENE: {from_scene} → {to_scene}")

    # 场景2到达叙述
    arrive_label = f"{to_scene}_arrive"
    arrive_text = _nlg.generate_narration(arrive_label, "", state)

    # 用【】列出新场景可探索的NPC和地点
    explore_hint = _build_explore_hint(to_scene, npc)

    return f"{transition_text}\n\n{arrive_text}\n\n{explore_hint}"


def _build_explore_hint(scene: str, mainline_npc: str) -> str:
    """生成新场景的可探索内容提示，用【】括出NPC和地点"""
    meta = SCENE_META.get(scene, {})
    npcs = meta.get("npcs", [])
    locations = meta.get("locations", [])

    loc_names = {
        "oak_tree": "老橡树", "clearing": "空地", "burrow": "树根窝", "stash": "存粮处",
        "fish_stall": "鱼摊", "veggie_stall": "蔬菜摊", "noodle_shop": "面馆", "cart": "推车",
        "iron_gate": "铁门", "room": "屋子", "doorstep": "门口",
    }

    parts = []
    # 主线NPC也在场景2中
    mainline_display = NPC_DISPLAY.get(mainline_npc, mainline_npc)
    parts.append(f"【{mainline_display}】")
    for npc_id in npcs:
        display = NPC_DISPLAY.get(npc_id, npc_id)
        parts.append(f"【{display}】")
    for loc in locations:
        name = loc_names.get(loc, loc)
        parts.append(f"【{name}】")

    return f"小饼环顾四周，这里有{'、'.join(parts)}可以探索。"


def _find_best_npc(state: GameState) -> Optional[str]:
    """公园轮数耗尽时，找好感最高且达标的NPC"""
    best = None
    best_aff = 0
    for npc, threshold in NPC_SCENE2_THRESHOLD.items():
        aff = state.npc_affinity.get(npc, 0)
        if aff >= threshold and aff > best_aff:
            best = npc
            best_aff = aff
    return best


# =====================================================
# 推荐选项
# =====================================================

def get_suggestions(state: GameState) -> List[str]:
    if state.game_over:
        return []

    scene = state.current_scene
    suggestions = []

    # 对话中的选项
    if state.in_dialogue:
        npc = state.dialogue_target
        return _dialogue_suggestions(npc, state)

    # === 优先级1：任务相关 ===
    for mainline_npc in ["squirrel", "grandma", "vendor"]:
        quest = get_current_quest(mainline_npc, state)
        if not quest or quest.get("scene") != scene:
            continue
        step = quest["step"]
        task_sugg = _quest_suggestion(mainline_npc, step, state)
        if task_sugg:
            suggestions.append(task_sugg)

    # === 优先级2：未探索目标 ===
    scene_meta = SCENE_META.get(scene, {})
    for npc_id in scene_meta.get("npcs", []):
        if not state.is_explored(f"approach_{npc_id}") and len(suggestions) < 3:
            display = NPC_DISPLAY.get(npc_id, npc_id)
            suggestions.append(f"走向{display}")

    for loc in scene_meta.get("locations", []):
        if not state.is_explored(loc) and len(suggestions) < 3:
            loc_names = {
                "tree": "大树", "bushes": "草丛", "fountain": "喷泉",
                "bench": "长椅", "path": "小路",
                "oak_tree": "老橡树", "clearing": "空地",
                "burrow": "树根窝", "stash": "存粮处",
                "fish_stall": "鱼摊", "noodle_shop": "面馆",
                "cart": "推车", "iron_gate": "铁门",
                "room": "屋里看看", "doorstep": "门口坐坐",
            }
            name = loc_names.get(loc, loc)
            suggestions.append(f"去{name}看看")

    # === 优先级3：搞砸选项（场景2必有，场景1随机） ===
    mischief = _mischief_suggestion(scene, state)
    if mischief and len(suggestions) < 4:
        suggestions.append(mischief)

    # 填充
    fillers = {
        "park":   ["四处闻闻", "摇摇尾巴", "趴下来晒太阳"],
        "forest": ["在林子里走走", "闻闻空气", "找个地方趴着"],
        "market": ["在市场里逛逛", "闻闻空气里的香味"],
        "ruins":  ["四处闻闻", "趴下来歇会"],
    }
    for f in fillers.get(scene, ["四处看看"]):
        if len(suggestions) >= 4:
            break
        if f not in suggestions:
            suggestions.append(f)

    return suggestions[:4]


def _dialogue_suggestions(npc: str, state: GameState) -> List[str]:
    """对话中的推荐选项"""
    mainline = _get_mainline_for_npc(npc)
    quest = get_current_quest(mainline, state) if mainline else None
    step = quest["step"] if quest else 0

    base = {
        "squirrel": ["吱吱叫回应它", "蹭蹭它", "问它草丛里有什么"],
        "grandma":  ["安静蹭蹭她的腿", "摇摇尾巴", "乖乖趴着听她说"],
        "vendor":   ["摇摇尾巴", "蹭蹭他的腿", "乖乖趴在旁边"],
        "owl":      ["安静听它说", "点点头"],
        "hedgehog": ["蹭蹭刺猬妈妈", "帮它看着蘑菇"],
        "fish_vendor": ["摇摇尾巴", "闻闻鱼"],
        "noodle_lady":  ["乖乖趴着", "摇尾巴"],
        "old_cat":  ["蹭蹭阿黄", "趴在它旁边"],
        "zhang":    ["摇摇尾巴", "乖乖坐着"],
    }

    opts = list(base.get(npc, ["摇摇尾巴"]))[:3]

    # 任务相关选项
    if npc == "squirrel" and step in (2, 3):
        opts.insert(0, "去草丛帮它找松果")
    if npc == "grandma" and step == 3:
        opts.insert(0, "吃她的饼干")
    if npc == "vendor" and step == 3:
        opts.insert(0, "帮大爷招揽客人")

    # 搞砸选项
    mischief = {
        "squirrel": "对它汪汪叫",
        "grandma":  "咬她的裤脚",
        "vendor":   "去戳气球",
        "owl":      "对它叫两声",
        "hedgehog": "偷偷尝一口蘑菇",
        "old_cat":  "追着阿黄跑",
        "zhang":    "对他汪汪叫",
    }
    if npc in mischief and len(opts) < 4:
        opts.append(mischief[npc])

    return opts[:4]


def _quest_suggestion(npc: str, step: int, state: GameState) -> Optional[str]:
    """根据任务步骤生成推荐"""
    # 公园步骤（静态）
    park_sugg = {
        ("squirrel", 1): "走向大树下的松鼠",
        ("squirrel", 2): None,  # 在对话中
        ("squirrel", 3): "去草丛找松果",
        ("squirrel", 4): "把松果带给松鼠",
        ("grandma", 1):  "走向喷泉边的老奶奶",
        ("grandma", 2):  None,
        ("grandma", 3):  None,
        ("grandma", 4):  None,
        ("vendor", 1):   "走向卖气球的大爷",
        ("vendor", 2):   None,
        ("vendor", 3):   None,
        ("vendor", 4):   None,
    }
    if (npc, step) in park_sugg:
        return park_sugg[(npc, step)]

    # 场景2（动态：根据探索标记推荐下一步）
    if npc == "squirrel" and step == 5:
        if not state.is_explored("owl_talked"):
            return "去老橡树找猫头鹰"
        if not state.is_explored("hedgehog_visited"):
            return "去树根窝找刺猬一家"
        return "在存粮处蹲守"

    if npc == "grandma" and step == 5:
        if not state.is_explored("guarded_cart"):
            return "帮奶奶看着推车"
        return "跟着老奶奶走"

    if npc == "vendor" and step == 5:
        if not state.is_explored("befriended_cat"):
            return "走向阿黄"
        return "安静地趴在大爷脚边"

    return None


def _mischief_suggestion(scene: str, state: GameState) -> Optional[str]:
    """搞砸选项"""
    options = {
        "park": [
            "对松鼠汪汪叫" if not state.is_explored("approach_squirrel") else None,
            "去戳气球" if state.is_explored("approach_vendor") else None,
        ],
        "forest": [
            "偷偷尝一口蘑菇",
            "对猫头鹰叫两声",
            "蹲守到一半跑掉",
        ],
        "market": [
            "偷一条鱼尝尝",
            "在菜市场里乱跑",
            "对路人汪汪叫",
        ],
        "ruins": [
            "追着阿黄跑",
            "把毯子咬了玩",
            "对张叔汪汪叫",
        ],
    }
    candidates = [o for o in options.get(scene, []) if o]
    return random.choice(candidates) if candidates else None


# =====================================================
# 辅助函数
# =====================================================

def _check_mischief(raw: str, state: GameState, log: Dict) -> Optional[Tuple[str, GameState]]:
    """关键词检测搞砸行为 + 守推车快捷触发，返回 (response, state) 或 None"""
    scene = state.current_scene

    # 菜市场："帮看推车"/"守推车" 关键词 → 直接触发守推车
    if scene == "market" and any(k in raw for k in ["看着推车", "守推车", "帮看", "守着", "看推车"]):
        if not state.is_explored("guarded_cart"):
            return _handle_wait("cart", raw, state, log)

    # 菜市场搞砸
    if scene == "market":
        if any(k in raw for k in ["偷鱼", "偷一条", "叼鱼", "抢鱼"]):
            delta = state.adjust_affinity("grandma", "steal_fish")
            if delta:
                log["state_changes"].append(f"affinity[grandma] += {delta}")
            return _nlg.generate_narration("market_steal_fish", raw, state), state
        if any(k in raw for k in ["乱跑", "乱窜", "撞翻", "捣乱"]):
            delta = state.adjust_affinity("grandma", "rampage_stall")
            if delta:
                log["state_changes"].append(f"affinity[grandma] += {delta}")
            return _nlg.generate_narration("market_rampage", raw, state), state
        if any(k in raw for k in ["不看了", "不守了", "跑开", "离开推车"]):
            delta = state.adjust_affinity("grandma", "abandon_cart")
            if delta:
                log["state_changes"].append(f"affinity[grandma] += {delta}")
            return _nlg.generate_narration("market_abandon_cart", raw, state), state

    # 废墟搞砸
    if scene == "ruins":
        if any(k in raw for k in ["追猫", "追着阿黄", "追阿黄", "欺负猫"]):
            delta = state.adjust_affinity("vendor", "bully_cat")
            if delta:
                log["state_changes"].append(f"affinity[vendor] += {delta}")
            return _nlg.generate_narration("ruins_bully_cat", raw, state), state
        if any(k in raw for k in ["咬毯子", "咬了", "撕", "破坏"]):
            delta = state.adjust_affinity("vendor", "destroy_stuff")
            if delta:
                log["state_changes"].append(f"affinity[vendor] += {delta}")
            return _nlg.generate_narration("ruins_destroy", raw, state), state

    return None


def _get_mainline_for_npc(npc: str) -> Optional[str]:
    """获取NPC所属的主线NPC"""
    from memory import NPC_CONTEXT
    ctx = NPC_CONTEXT.get(npc)
    return ctx["mainline"] if ctx else None


def _try_quest_advance(mainline: str, trigger: str, state: GameState,
                       changes: List[str]):
    """尝试推进任务"""
    if mainline and check_quest_trigger(mainline, trigger, state):
        effects = advance_quest(mainline, state)
        changes.append(f"quest[{mainline}] → step {effects.get('step')}")
        return effects
    return {}


def _wants_exit(text: str) -> bool:
    return any(p in text for p in _EXIT_PHRASES)


def _scene_context(state: GameState) -> str:
    scene_name = SCENE_META.get(state.current_scene, {}).get("name", "")
    return f"{scene_name}，流浪小狗小饼在探索"


def _nlu_log(nlu: NLUResult) -> Dict:
    return {
        "intent": nlu.intent,
        "entities": nlu.entities,
        "confidence": round(nlu.confidence, 3),
    }
