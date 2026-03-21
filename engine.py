"""
游戏引擎
========
修复：
  1. EXPLORE/ASK 指向 NPC/动物 → 自动提升为对话入口
  2. 场景标签精确传递给 NLG，不让 NLG 自行推断
  3. 入夜先显示过渡叙事，下一回合才触发结局
  4. 小动物完整对话支持
  5. 退出对话覆盖所有 6 个角色
"""
import random
from typing import Tuple, List, Dict, Any, Optional

import config
from game_state import GameState, normalize_entity, ALL_NPCS, CLUE_META, ENDING_META, PHASE_DESC
from nlu import DeepSeekNLU, NLUResult
from nlg import DeepSeekNLG, NPC_DISPLAY
import memory as mem

_nlu = DeepSeekNLU()
_nlg = DeepSeekNLG()

_EXIT_PHRASES = ["离开", "再见", "走了", "不说了", "算了", "拜拜", "结束", "走开",
                 "退出", "结束对话", "goodbye", "bye", "quit"]

OPENING_NARRATION = """🐾 傍晚的阳光斜斜地打在公园的石板路上。

小饼慢慢睁开眼睛，发现自己趴在一棵大树下的草地上——主人小明不见了。空气里有割草的气味，还有一点点熟悉的、属于小明的味道，从不远处飘来。

公园里还有些人。【长椅】就在几步之外，上面的气味最浓。【老奶奶】坐在喷泉边上，她的泰迪正绕着她转圈。【气球摊主】推着小车停在喷泉旁边，耳朵里塞着耳机。角落里有一丛【草丛】，一只【松鼠】刚从那边蹿出来，回头看了小饼一眼。远处的跑道上，【慢跑者】正戴着耳机绕圈跑步。公园入口附近，一群【鸽子】在地上抢食。废弃的【角落】里，有什么东西在阴影里动了一下……

小饼站起来，抖了抖身上的草屑。要去哪里呢？"""


def new_game() -> Tuple[GameState, List[str]]:
    state = GameState()
    return state, get_suggestions(state)


def process_turn(
    player_input: str,
    state: GameState,
) -> Tuple[str, GameState, List[str], Dict[str, Any]]:
    log: Dict[str, Any] = {
        "turn": state.turn_count + 1,
        "player_input": player_input,
        "nlu": {},
        "intent_handled": "",
        "state_changes": [],
        "response_preview": "",
    }

    if state.game_over:
        return "游戏已经结束了，请点击「重新开始」再玩一次！", state, [], log

    # ===== 回合推进 & 阶段变化 =====
    state.turn_count += 1
    old_phase = state.phase
    if state.turn_count > config.PHASE_2_END:
        state.phase = 3
    elif state.turn_count > config.PHASE_1_END:
        state.phase = 2

    if state.phase != old_phase:
        log["state_changes"].append(f"phase: {old_phase} → {state.phase}")

    response = ""

    # ===== 优先级 1：观察模式结果（上一轮 OBSERVE） =====
    if state.observe_target:
        target = state.observe_target
        state.observe_target = None
        response, state = _handle_observe_result(target, state, log)
        log["intent_handled"] = f"OBSERVE_RESULT:{target}"

    # ===== 优先级 2：危险自救模式 =====
    elif state.flag_jogger_danger and not state.flag_danger_resolved and not state.flag_rescue_failed:
        nlu_result = _nlu.classify(player_input, _scene_context(state))
        log["nlu"] = _nlu_log(nlu_result)
        response, state = _handle_rescue(player_input, nlu_result, state, log)
        log["intent_handled"] = "RESCUE"

    # ===== 优先级 3：对话模式 =====
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

    # ===== 优先级 4：普通意图 =====
    else:
        nlu_result = _nlu.classify(player_input, _scene_context(state))
        log["nlu"] = _nlu_log(nlu_result)
        state.last_nlu_log = log["nlu"]
        response, state = _dispatch_intent(nlu_result, player_input, state, log)
        log["intent_handled"] = nlu_result.intent

    # ===== 入夜处理（两步：先过渡叙事，再触发结局） =====
    if state.phase == 3 and not state.game_over:
        if not state.flag_night_transition_shown:
            # 第一次进入入夜：显示过渡叙事，不触发结局
            state.flag_night_transition_shown = True
            transition = _nlg.generate_transition(state)
            response += f"\n\n{'─'*40}\n\n🌃 {transition}"
            log["state_changes"].append("night_transition_shown")
        else:
            # 已显示过渡叙事：检测结局
            ending = _auto_ending(state)
            if ending:
                state.ending = ending
                state.game_over = True
                ending_text = _nlg.generate_ending(ending, state)
                response += f"\n\n{'─'*40}\n\n{ending_text}"
                log["state_changes"].append(f"AUTO_ENDING: {ending}")

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

    # ===== NPC/动物目标 → 自动进入对话（EXPLORE/ASK/APPROACH 统一处理） =====
    if target in ALL_NPCS and intent in ("EXPLORE", "ASK", "APPROACH", "FOLLOW", "BARK"):
        if intent == "BARK" and target == "jogger":
            return _handle_bark(target, raw, state, log)
        if intent == "BARK" and target == "grandma":
            return _handle_bark(target, raw, state, log)
        # 其余情况：靠近 NPC/动物 → 进入对话
        return _enter_npc_dialogue(target, raw, state, log)

    handlers = {
        "SNIFF":   _handle_sniff,
        "EXPLORE": _handle_explore,
        "EXAMINE": _handle_examine,
        "APPROACH":_handle_approach,
        "ASK":     _handle_ask,
        "BARK":    _handle_bark,
        "FOLLOW":  _handle_follow,
        "HIDE":    _handle_hide,
        "WAIT":    _handle_wait,
        "OBSERVE": _handle_observe,
        "UNKNOWN": _handle_unknown,
    }
    return handlers.get(intent, _handle_unknown)(target, raw, state, log)


# =====================================================
# 进入 NPC/动物对话的统一入口
# =====================================================

def _enter_npc_dialogue(npc: str, raw: str, state: GameState,
                        log: Dict) -> Tuple[str, GameState]:
    """统一的进入对话逻辑，适用于所有 6 个角色"""
    changes = []

    # 关系进度更新
    if npc == "grandma" and not state.flag_grandma_bonded:
        state.grandma_bond_progress = min(state.grandma_bond_progress + 1, 2)
        if state.grandma_bond_progress >= 2:
            state.flag_grandma_bonded = True
            changes.append("flag_grandma_bonded=True")

    elif npc == "cat" and not state.flag_cat_ally:
        state.cat_approach_progress = min(state.cat_approach_progress + 1, 1)
        if state.cat_approach_progress >= 1:
            state.flag_cat_ally = True
            changes.append("flag_cat_ally=True")

    elif npc == "jogger":
        # 慢跑者有自己的接触逻辑
        return _approach_jogger(raw, state, log)

    state.in_dialogue = True
    state.dialogue_target = npc
    npc_resp, flags = _nlg.generate_npc_response(npc, raw, state)
    _post_npc_flags(npc, flags, state, log)
    mem.append_npc_history(npc, "user", raw, state)
    mem.append_npc_history(npc, "assistant", npc_resp, state)

    if flags.get("chat_ended"):
        state.in_dialogue = False
        state.dialogue_target = None

    log["state_changes"].extend(changes)
    display = NPC_DISPLAY.get(npc, npc)
    return f"💬 【{display}】{npc_resp}", state


def _approach_jogger(raw: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    """慢跑者接触逻辑"""
    if state.flag_danger_resolved:
        # 已自救成功，可以正常对话
        state.in_dialogue = True
        state.dialogue_target = "jogger"
        npc_resp, flags = _nlg.generate_npc_response("jogger", raw, state)
        _post_npc_flags("jogger", flags, state, log)
        mem.append_npc_history("jogger", "user", raw, state)
        mem.append_npc_history("jogger", "assistant", npc_resp, state)
        if flags.get("chat_ended"):
            state.in_dialogue = False
            state.dialogue_target = None
        return f"💬 【慢跑者】{npc_resp}", state

    elif not state.flag_jogger_warned:
        # 第一次靠近 → 惊扰警告
        state.flag_jogger_warned = True
        log["state_changes"].append("flag_jogger_warned=True")
        narration = _nlg.generate_narration("jogger_first_approach", raw, state)
        return narration + "\n\n⚠️ 慢跑者好像有点不高兴……", state

    elif not state.flag_jogger_danger:
        # 已警告，再次靠近 → 触发危险
        state.flag_jogger_danger = True
        log["state_changes"].append("flag_jogger_danger=True")
        narration = _nlg.generate_narration("jogger_warned_approach", raw, state)
        return narration + "\n\n⚠️⚠️ **危险！** 小饼感觉情况不对劲，必须马上做出决定！", state

    # 危险进行中
    return _nlg.generate_narration("jogger_danger_triggered", raw, state), state


# =====================================================
# 各意图处理函数
# =====================================================

def _handle_sniff(target: str, raw: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    changes = []
    label = "unknown_action"

    if target == "bench":
        if not state.is_explored("bench_sniff"):
            state.mark_explored("bench_sniff")
            if state.add_clue("bench_scent"):
                changes.append("clue: bench_scent")
            label = "sniff_bench_new"
        else:
            label = "sniff_bench_again"

    elif target == "bushes":
        if not state.is_explored("bushes_sniff"):
            state.mark_explored("bushes_sniff")
            state.mark_explored("bushes")
            if state.add_clue("key_holder"):
                changes.append("clue: key_holder")
            label = "sniff_bushes_new"
        else:
            label = "sniff_bushes_again"

    elif target == "corner" and state.flag_cat_ally:
        if not state.is_explored("corner_sniff"):
            state.mark_explored("corner_sniff")
            state.mark_explored("outfit")
            if state.add_clue("cat_outfit_card"):
                state.has_membership_card = True
                changes.append("clue: cat_outfit_card")
            label = "sniff_corner_cat"
        else:
            label = "already_explored"
    else:
        label = f"sniff_{target}" if target else "unknown_action"

    log["state_changes"].extend(changes)
    return _nlg.generate_narration(label, raw, state), state


def _handle_explore(target: str, raw: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    # 触发主动结局
    if target == "entrance" or any(kw in raw for kw in ["门口", "全家", "便利店", "出去", "离开公园"]):
        ending = _check_active_ending(state)
        if ending:
            state.ending = ending
            state.game_over = True
            log["state_changes"].append(f"ENDING_TRIGGERED: {ending}")
            return _nlg.generate_ending(ending, state), state

    label_map = {
        "bushes":   "explore_bushes_new",
        "corner":   "explore_corner_new",
        "fountain": "explore_fountain",
        "entrance": "explore_entrance",
        "track":    "jogger_first_appear",
    }
    label = label_map.get(target, f"explore_{target}" if target else "unknown_action")

    # 慢跑者/跑道首次亮相
    if target in ("track", "jogger") and not state.is_explored("jogger_first"):
        state.mark_explored("jogger_first")
        label = "jogger_first_appear"

    state.mark_explored(target)
    return _nlg.generate_narration(label, raw, state), state


def _handle_examine(target: str, raw: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    changes = []
    label = f"examine_{target}" if target else "unknown_action"

    if target == "outfit" and state.flag_cat_ally:
        if not state.is_explored("outfit"):
            state.mark_explored("outfit")
            if state.add_clue("cat_outfit_card"):
                state.has_membership_card = True
                changes.append("clue: cat_outfit_card")
            label = "examine_outfit"

    elif target == "membership_card" and state.has_membership_card:
        label = "examine_card"

    elif target == "key_holder" and "key_holder" in state.clues_found:
        label = "examine_keyholder"

    log["state_changes"].extend(changes)
    state.mark_explored(target)
    return _nlg.generate_narration(label, raw, state), state


def _handle_approach(target: str, raw: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    if target in ALL_NPCS:
        return _enter_npc_dialogue(target, raw, state, log)
    return _nlg.generate_narration(f"approach_{target}", raw, state), state


def _handle_ask(target: str, raw: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    if target in ALL_NPCS:
        return _enter_npc_dialogue(target, raw, state, log)
    return _nlg.generate_narration("unknown_action", raw, state), state


def _handle_bark(target: str, raw: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    if target == "grandma":
        # 对老奶奶叫 → 负面，不建立关系
        return _nlg.generate_narration("grandma_approach_rude", raw, state), state

    elif target == "jogger":
        if not state.flag_jogger_danger:
            if not state.flag_jogger_warned:
                state.flag_jogger_warned = True
                log["state_changes"].append("flag_jogger_warned=True ← BARK")
                narration = _nlg.generate_narration("jogger_first_approach", raw, state)
                return narration + "\n\n⚠️ 慢跑者好像有点不高兴……", state
            else:
                state.flag_jogger_danger = True
                log["state_changes"].append("flag_jogger_danger=True ← BARK")
                narration = _nlg.generate_narration("jogger_danger_bark", raw, state)
                return narration + "\n\n⚠️⚠️ **危险！** 必须马上做出决定！", state

    return _nlg.generate_narration(f"bark_{target}", raw, state), state


def _handle_follow(target: str, raw: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    changes = []

    if target == "squirrel":
        if not state.is_explored("bushes"):
            state.mark_explored("bushes")
            state.mark_explored("squirrel")
            if state.add_clue("key_holder"):
                changes.append("clue: key_holder ← squirrel")
            log["state_changes"].extend(changes)
            return _nlg.generate_narration("follow_squirrel", raw, state), state

    elif target == "cat":
        if state.flag_cat_ally and not state.is_explored("corner"):
            state.mark_explored("corner")
            if state.add_clue("cat_outfit_card"):
                state.has_membership_card = True
                changes.append("clue: cat_outfit_card ← follow_cat")
            log["state_changes"].extend(changes)
            return _nlg.generate_narration("follow_cat_to_corner", raw, state), state
        elif not state.flag_cat_ally:
            # 跟猫但还没建立信任 → 进入对话
            return _enter_npc_dialogue("cat", raw, state, log)

    elif target == "pigeons":
        state.mark_explored("pigeons")
        if not state.flag_jogger_warned and not state.flag_jogger_danger:
            state.flag_jogger_warned = True
            changes.append("flag_jogger_warned=True ← pigeons_trap")
            log["state_changes"].extend(changes)
            narration = _nlg.generate_narration("follow_pigeons_trap", raw, state)
            return narration + "\n\n⚠️ 慢跑者好像有点不高兴……", state

    elif target == "grandma" and state.phase == 3 and state.flag_grandma_bonded:
        state.ending = "adopted_grandma"
        state.game_over = True
        log["state_changes"].append("ENDING: adopted_grandma")
        return _nlg.generate_ending("adopted_grandma", state), state

    log["state_changes"].extend(changes)
    return _nlg.generate_narration(f"follow_{target}", raw, state), state


def _handle_hide(target: str, raw: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    if state.flag_jogger_danger and not state.flag_danger_resolved:
        state.flag_danger_resolved = True
        log["state_changes"].append("flag_danger_resolved=True ← HIDE")
        return _nlg.generate_narration("rescue_hide_success", raw, state), state
    return _nlg.generate_narration(f"hide_{target}", raw, state), state


def _handle_wait(target: str, raw: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    if state.flag_jogger_danger and not state.flag_danger_resolved:
        success = random.random() > 0.45
        if success:
            state.flag_danger_resolved = True
            log["state_changes"].append("flag_danger_resolved=True ← WAIT cute")
            return _nlg.generate_narration("rescue_cute_success", raw, state), state
        else:
            state.flag_rescue_failed = True
            state.game_over = True
            log["state_changes"].append("flag_rescue_failed=True ← WAIT fail")
            narration = _nlg.generate_narration("rescue_cute_fail", raw, state)
            return narration + "\n\n" + _nlg.generate_ending("bad", state), state

    return _nlg.generate_narration("wait_rest", raw, state), state


def _handle_observe(target: str, raw: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    state.observe_target = target or raw
    log["state_changes"].append(f"observe_target={state.observe_target}")
    return _nlg.generate_narration(f"observe_{target}", raw, state), state


def _handle_observe_result(target: str, state: GameState, log: Dict) -> Tuple[str, GameState]:
    if target in ("jogger", "慢跑者"):
        if state.flag_jogger_warned and not state.flag_jogger_danger:
            state.flag_jogger_danger = True
            log["state_changes"].append("flag_jogger_danger=True ← observe result")
            narration = _nlg.generate_narration("jogger_warned_approach", f"等待{target}", state)
            return narration + "\n\n⚠️⚠️ **危险！** 必须马上做出决定！", state
        elif not state.flag_jogger_warned:
            state.flag_jogger_warned = True
            log["state_changes"].append("flag_jogger_warned=True ← observe")
    return _nlg.generate_narration(f"observe_result_{target}", f"等待{target}", state), state


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

    # EXPLORE 指向非当前 NPC → 退出对话
    target = normalize_entity(nlu.entities.get("target", ""))
    if nlu.intent in ("EXPLORE", "HIDE") and target not in ALL_NPCS:
        state.in_dialogue = False
        state.dialogue_target = None
        display = NPC_DISPLAY.get(npc, npc)
        return f"小饼向【{display}】点点头，转身走开了。", state

    npc_resp, flags = _nlg.generate_npc_response(npc, raw, state)
    _post_npc_flags(npc, flags, state, log)
    mem.append_npc_history(npc, "user", raw, state)
    mem.append_npc_history(npc, "assistant", npc_resp, state)

    if flags.get("chat_ended"):
        state.in_dialogue = False
        state.dialogue_target = None

    display = NPC_DISPLAY.get(npc, npc)
    return f"💬 【{display}】{npc_resp}", state


# =====================================================
# 危险自救
# =====================================================

def _handle_rescue(raw: str, nlu: NLUResult, state: GameState,
                   log: Dict) -> Tuple[str, GameState]:
    intent = nlu.intent
    target = normalize_entity(nlu.entities.get("target", ""))

    if intent == "HIDE" or target == "bushes":
        state.flag_danger_resolved = True
        log["state_changes"].append("flag_danger_resolved=True ← HIDE")
        return _nlg.generate_narration("rescue_hide_success", raw, state), state

    elif intent == "APPROACH" and target == "grandma":
        if state.flag_grandma_bonded:
            state.flag_danger_resolved = True
            log["state_changes"].append("flag_danger_resolved=True ← grandma rescue")
            return _nlg.generate_narration("rescue_grandma_success", raw, state), state
        else:
            state.flag_rescue_failed = True
            state.game_over = True
            log["state_changes"].append("flag_rescue_failed=True ← grandma not bonded")
            narration = _nlg.generate_narration("rescue_grandma_fail", raw, state)
            return narration + "\n\n" + _nlg.generate_ending("bad", state), state

    elif intent == "WAIT" or any(kw in raw for kw in ["趴下", "摇尾巴", "装可怜"]):
        return _handle_wait("", raw, state, log)

    else:
        state.flag_rescue_failed = True
        state.game_over = True
        log["state_changes"].append("flag_rescue_failed=True ← no rescue")
        narration = _nlg.generate_narration("rescue_fail_generic", raw, state)
        return narration + "\n\n" + _nlg.generate_ending("bad", state), state


# =====================================================
# 结局检测
# =====================================================

def _check_active_ending(state: GameState) -> Optional[str]:
    if state.flag_direction_known and len(state.clues_found) >= 3:
        return "happy"
    if len(state.clues_found) >= 3 and state.has_membership_card and not state.flag_direction_known:
        return "solo"
    return None


def _auto_ending(state: GameState) -> Optional[str]:
    if state.flag_rescue_failed:
        return "bad"
    if state.flag_direction_known and len(state.clues_found) >= 3:
        return "happy"
    if len(state.clues_found) >= 3 and state.has_membership_card:
        return "solo"
    if state.flag_grandma_bonded:
        return "adopted_grandma"
    if state.flag_cat_ally:
        return "adopted_cat"
    return "stray"


# =====================================================
# 建议选项
# =====================================================

def get_suggestions(state: GameState) -> List[str]:
    if state.game_over:
        return []

    # 危险自救
    if state.flag_jogger_danger and not state.flag_danger_resolved and not state.flag_rescue_failed:
        return ["躲进旁边的草丛", "跑向老奶奶身边", "趴下来摇摇尾巴"]

    # 对话中
    if state.in_dialogue:
        npc = state.dialogue_target
        dialogue_opts = {
            "grandma":  ["撒娇蹭蹭她", "问她有没有看到小明", "问全家便利店怎么走"],
            "vendor":   ["问他有没有看到人", "追问细节"],
            "jogger":   ["温柔地蹭蹭他", "问他手机里的照片"] if state.flag_danger_resolved
                        else ["温柔地蹭蹭他"],
            "cat":      ["喵呜一声打招呼", "跟着它走", "蹭蹭它的脸"],
            "squirrel": ["吱吱叫回应它", "问它草丛里有什么", "追着它玩"],
            "pigeons":  ["汪一声问问它们", "跟着鸽子走", "假装不在意它们"],
        }
        return dialogue_opts.get(npc, ["继续聊"])[:3]

    suggestions = []

    # 主动离开
    if state.flag_direction_known and len(state.clues_found) >= 3:
        suggestions.append("往公园门口对面的全家跑去！")
    elif len(state.clues_found) >= 3 and state.has_membership_card and not state.flag_direction_known:
        suggestions.append("悄悄从公园侧门溜出去")

    # 未探索热点
    if not state.is_explored("bench_sniff") and len(suggestions) < 4:
        suggestions.append("闻闻那张长椅")
    if not state.is_explored("bushes") and len(suggestions) < 4:
        suggestions.append("去草丛看看")
    if not state.flag_grandma_bonded and len(suggestions) < 4:
        suggestions.append("慢慢走向老奶奶")
    if not state.flag_cat_ally and len(suggestions) < 4:
        suggestions.append("去废弃角落找找")
    if not state.is_explored("vendor") and len(suggestions) < 4:
        suggestions.append("靠近气球摊主")
    if state.flag_cat_ally and not state.is_explored("corner") and len(suggestions) < 4:
        suggestions.append("跟着流浪猫走")
    if not state.is_explored("jogger_first") and len(suggestions) < 4:
        suggestions.append("观察那个慢跑者")
    if not state.is_explored("squirrel") and len(suggestions) < 4:
        suggestions.append("跟着松鼠跑")
    if not state.is_explored("pigeons") and len(suggestions) < 4:
        suggestions.append("走近那群鸽子")

    fillers = ["四处嗅嗅看", "抬头张望一下", "在公园里走走"]
    for f in fillers:
        if len(suggestions) >= 4:
            break
        suggestions.append(f)

    return suggestions[:4]


# =====================================================
# 辅助函数
# =====================================================

def _post_npc_flags(npc: str, flags: Dict, state: GameState, log: Dict):
    if flags.get("key_info_revealed") and not state.flag_direction_known:
        state.flag_direction_known = True
        new = state.add_clue("grandma_witness")
        log["state_changes"].append(
            f"flag_direction_known=True {'+ clue:grandma_witness' if new else ''}"
        )
    if flags.get("photo_shown") and not state.flag_jogger_photo_revealed:
        state.flag_jogger_photo_revealed = True
        state.add_clue("jogger_photo")
        log["state_changes"].append("flag_jogger_photo_revealed=True + clue:jogger_photo")
    if flags.get("hint_bushes") and not state.is_explored("squirrel_hint"):
        state.mark_explored("squirrel_hint")
        log["state_changes"].append("squirrel hinted bushes")


def _wants_exit(text: str) -> bool:
    return any(p in text for p in _EXIT_PHRASES)


def _scene_context(state: GameState) -> str:
    phase_name = PHASE_DESC.get(state.phase, ("傍晚", ""))[0]
    return f"{phase_name}的城市公园，小饼在寻找主人小明"


def _nlu_log(nlu: NLUResult) -> Dict:
    return {
        "intent": nlu.intent,
        "entities": nlu.entities,
        "confidence": round(nlu.confidence, 3),
    }
