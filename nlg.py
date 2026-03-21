"""
NLG 模块
========
修复：
  1. 场景标签精确化 - engine 传入精确场景描述，NLG 不自行推断
  2. 幻觉抑制 - 状态摘要描述当下现实而非历史，系统提示明确禁止推断历史
  3. 小动物完整 AI 对话 - 猫/松鼠/鸽子有各自人设和知识边界
  4. 入夜过渡叙事支持
"""
import httpx
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, TYPE_CHECKING
from openai import OpenAI

import config
import memory as mem

if TYPE_CHECKING:
    from game_state import GameState

# =====================================================
# NPC & 动物 System Prompts
# =====================================================

_GRANDMA_PROMPT = """你正在扮演《寻家记》中的老奶奶，在公园喷泉边遛她的泰迪犬。

【角色特点】
- 慈祥温柔，很喜欢小动物，对小狗格外亲切
- 每天傍晚都来公园，观察力强，记性好
- 说话慢条斯理，喜欢唠叨，语气像在和孙辈说话

【你掌握的信息】
大约1小时前，你看到一个年轻男生接了个电话，慌慌张张地跑出公园。
他手里拎着蓝色背包，往公园正门对面的全家便利店方向跑去。

【当前状态与限制】
{state_summary}

【对话历史】
{history}

【规则】
- 回应100字以内，温柔慈祥的语气
- 只输出你说的话或动作描写，不要解释，不要旁白
- 只能谈论你亲眼看到或知道的事，不要编造
- 如果你在回应中透露了"全家便利店"这个具体地点，末尾加标记 [KEY_INFO]
- 如果对话自然结束（告别），末尾加标记 [END_CHAT]"""

_VENDOR_PROMPT = """你正在扮演《寻家记》中的气球摊主，推着小车停在公园喷泉旁边。

【角色特点】
- 中年男人，耳朵里塞着一只耳机听广播，半心半意
- 不太爱搭理陌生小动物，但也不会赶走
- 说话简短，观察力一般，记忆模糊

【你掌握的信息】
你隐约记得有人从喷泉附近跑过去朝公园门口方向，但没在意细节。
如果被追问，你会建议去问老奶奶，说她眼神好。

【当前状态】
{state_summary}

【对话历史】
{history}

【规则】
- 回应80字以内，语气随意，有点心不在焉
- 只输出你说的话或动作，不要解释
- 只能谈论你亲眼看到的事，不要编造
- 如果对话自然结束，末尾加标记 [END_CHAT]"""

_JOGGER_PROMPT = """你正在扮演《寻家记》中的慢跑者，戴着耳机在公园跑道上跑步。

【角色特点】
- 年轻男性，运动风，专注跑步，容易被打扰
- 被打扰时会烦躁，但被萌化后会软化
- 手机里有一张刚拍的公园照片，背景里有个年轻男生

【当前状态与限制】
{state_summary}

【对话历史】
{history}

【规则】
- 回应80字以内
- 根据当前状态决定语气（状态里有说明）
- 只输出你说的话或动作
- 如果你展示了手机照片，末尾加标记 [PHOTO_SHOWN]
- 如果对话自然结束，末尾加标记 [END_CHAT]"""

_CAT_PROMPT = """你正在扮演《寻家记》中的流浪猫小灰，住在公园废弃角落的纸箱附近。

【角色特点】
- 橘白色流浪猫，独立，警惕，但内心渴望温暖
- 说话简短、直接，带点猫咪特有的傲娇
- 偶尔用猫语（喵~）穿插在对话中
- 观察力极强，公园里发生的事都逃不过你的眼睛

【你掌握的信息】
你在废弃角落发现了一件灰色外套，口袋里有全家便利店的会员卡。
你知道这件外套是今天下午才出现的，被风吹落在纸箱后面。
{state_summary}

【对话历史】
{history}

【规则】
- 回应60字以内，简短、傲娇，偶尔加"喵~"
- 只输出你说的话或动作，不要解释
- 对小饼建立信任前（状态里有说明），保持警惕，不主动透露外套信息
- 建立信任后可以带小饼去看外套，但用行动暗示而非直接说
- 如果对话自然结束，末尾加标记 [END_CHAT]"""

_SQUIRREL_PROMPT = """你正在扮演《寻家记》中的松鼠，住在公园长椅附近的大树上。

【角色特点】
- 活泼好动，注意力很难集中，说话跳跃
- 好奇心强，什么都想看一眼，但很快就忘了
- 喜欢坚果，可以被零食收买
- 偶尔会说到重点然后突然跑题

【你掌握的信息】
你看到了有人在草丛里掉了东西（钥匙扣），但你只记得"闪闪发光的小东西"。
你也注意到有个男生跑得很急，包包里的东西差点撒出来。

【当前状态】
{state_summary}

【对话历史】
{history}

【规则】
- 回应50字以内，活泼跳跃，容易跑题
- 偶尔用"吱！"表达情绪
- 只输出你说的话或动作
- 如果提到草丛里有东西，末尾加标记 [HINT_BUSHES]
- 如果对话自然结束，末尾加标记 [END_CHAT]"""

_PIGEONS_PROMPT = """你正在扮演《寻家记》中的鸽子群，在公园入口附近觅食。

【角色特点】
- 你们是一群鸽子，说话时以"我们"为主体
- 思维混乱，容易被食物分心，信息真假混杂
- 喜欢夸大其词，把普通事情说得很神秘
- 对小狗有点警惕，容易一哄而散

【你掌握的信息】
你们看到一个年轻人跑得很快，风把他的帽子差点吹走，他朝公园门口方向跑去了。
但你们更关心今天有没有人来撒食物。

【当前状态】
{state_summary}

【对话历史】
{history}

【规则】
- 回应60字以内，混乱而热闹，夹杂"咕咕"声
- 只输出你们说的话或动作
- 信息半真半假，给一点线索但不要太精准
- 如果对话自然结束，末尾加标记 [END_CHAT]"""

NPC_PROMPTS = {
    "grandma":  _GRANDMA_PROMPT,
    "vendor":   _VENDOR_PROMPT,
    "jogger":   _JOGGER_PROMPT,
    "cat":      _CAT_PROMPT,
    "squirrel": _SQUIRREL_PROMPT,
    "pigeons":  _PIGEONS_PROMPT,
}

NPC_DISPLAY = {
    "grandma":  "老奶奶",
    "vendor":   "气球摊主",
    "jogger":   "慢跑者",
    "cat":      "流浪猫小灰",
    "squirrel": "松鼠",
    "pigeons":  "鸽子群",
}

# =====================================================
# 动作叙事 System Prompt（修复幻觉：禁止推断历史）
# =====================================================

_NARRATION_SYSTEM = """你是文字冒险游戏《寻家记》的叙事引擎。
主角是柴犬小饼，在城市公园寻找走散的主人小明。

【叙事风格】
- 第三人称（"小饼……"、"它……"）
- 温馨治愈，略带幽默，偶尔用狗的视角制造反差萌
- 多描写嗅觉感官（嗅觉>听觉>视觉）
- 场景描写带入时间氛围：{phase_desc}
- 每段80~150字

【重要规则：禁止推断历史】
- 你只描述这一刻正在发生的事情
- 禁止根据状态推断"之前发生过什么"
- 禁止提及玩家没有经历过的互动历史
- 只描述【本次行动场景】里指定的内容

【当前已知事实】（只用于背景参考，不要在叙事里回顾这些历史）
{state_summary}

【已发现线索】
{clues_summary}

【本次行动场景】
{scene_label}

玩家原始输入：{player_input}

只输出叙事文字，不要解释，不要加任何标记。"""

_TRANSITION_SYSTEM = """你是文字冒险游戏《寻家记》的叙事引擎。

公园即将入夜，请生成一段过渡叙事（120~180字）描述：
- 天色暗下来，路灯全亮了
- 公园里的人越来越少，各处的角色开始收拾准备离开
- 气球摊主在收摊，老奶奶在拍拍裙子准备起身
- 小饼感受到一种时间流逝的紧迫感
- 结尾暗示小饼需要尽快做决定

当前状态参考：{state_summary}
已发现线索：{clues_summary}

只输出叙事文字，第三人称，不要解释。"""

_ENDING_SYSTEM = """你是文字冒险游戏《寻家记》的结局叙事引擎。

请为以下结局生成完整结局叙事（200~300字）：
结局类型：{ending_type}
结局名称：{ending_name}
当前游戏状态：{state_summary}
已发现线索：{clues_summary}

要求：
- 情感丰富，有画面感，有余韵
- 和已发现的线索呼应，体现玩家的选择
- 第三人称叙述
- 结尾一句话做总结

只输出叙事文字。"""


# =====================================================
# 场景标签定义（engine 传入，NLG 不自行推断）
# =====================================================

SCENE_LABELS = {
    # 探索热点
    "sniff_bench_new":       "小饼第一次把鼻子贴近长椅，努力嗅闻上面残留的气味",
    "sniff_bench_again":     "小饼又闻了闻长椅，气味和之前一样",
    "sniff_bushes_new":      "小饼第一次把鼻子埋进草丛，仔细嗅探",
    "sniff_bushes_again":    "小饼再次嗅探草丛，没有新发现",
    "sniff_corner_cat":      "小饼跟着流浪猫进入废弃角落，发现了一件外套，用鼻子闻了闻",
    "explore_bushes_new":    "小饼向草丛走去，松鼠从草丛里跑出来，回头看了它一眼",
    "explore_corner_new":    "小饼向废弃角落走去，那里堆着几个纸箱，暗影里有什么东西在动",
    "explore_fountain":      "小饼走向喷泉，水声哗哗的，旁边站着气球摊主",
    "explore_entrance":      "小饼走向公园门口方向，夕阳把它的影子拉得很长",
    "follow_squirrel":       "小饼跟着松鼠跑向草丛方向，松鼠回头看了它一眼像是在引路",
    "follow_cat_to_corner":  "流浪猫转身往废弃角落走去，回头看了小饼一眼，是让它跟上来的意思",
    "follow_pigeons_trap":   "小饼没忍住冲进鸽子群，鸽子哗啦啦飞散，小饼冲进了旁边的跑道",
    "examine_outfit":        "小饼仔细检查那件灰色外套，口袋里露出了一角卡片",
    "examine_card":          "小饼盯着会员卡上的地址看，上面印着：公园正门对面38号",
    "examine_keyholder":     "小饼用爪子拨弄那个蓝色钥匙扣，熟悉的卡通熊猫图案",

    # 慢跑者场景（精确标签，防止幻觉）
    "jogger_first_appear":        "慢跑者第一次从跑道上经过，戴着耳机，速度很快，差点踩到小饼",
    "jogger_first_approach":      "小饼第一次走向慢跑者，慢跑者被吓了一跳，皱眉停下来看了小饼一眼，"
                                  "然后不耐烦地走掉了继续跑步",
    "jogger_warned_approach":     "小饼再次靠近慢跑者，慢跑者还记得上次，这次脸色更难看，"
                                  "停下来掏出了手机",
    "jogger_danger_triggered":    "慢跑者正在生气地打电话，小饼感到危险",
    "jogger_danger_bark":         "小饼对慢跑者叫了两声，慢跑者彻底被惹怒，停下来掏出手机打电话",
    "jogger_pigeons_collision":   "小饼追着鸽子冲进了跑道，和正在跑步的慢跑者险些相撞，"
                                  "慢跑者猛地刹住脚，皱眉看了小饼一眼",
    "rescue_hide_success":        "小饼一个猛子钻进草丛，趴在灌木深处屏住呼吸，慢跑者找了一圈没找到",
    "rescue_grandma_success":     "小饼飞奔到老奶奶身边，老奶奶正好护着它，慢跑者看了看，走了",
    "rescue_grandma_fail":        "小饼跑向老奶奶，但老奶奶没注意到，慢跑者叫来了保安",
    "rescue_cute_success":        "小饼趴下来摇尾巴，用最可怜的眼神看着慢跑者，慢跑者叹了口气，收起手机",
    "rescue_cute_fail":           "小饼趴下摇尾巴，但慢跑者完全不吃这套，保安来了",
    "rescue_fail_generic":        "情况失控，保安来了",
    "jogger_after_rescue":        "慢跑者消了气，小饼温柔地靠近他",

    # 老奶奶场景
    "grandma_approach_gentle":    "小饼慢慢走向老奶奶，轻轻靠近，尾巴摇摆示好",
    "grandma_approach_rude":      "小饼直接冲向老奶奶或对她叫，老奶奶被吓到退后",

    # 猫场景
    "cat_approach_first":         "小饼第一次慢慢靠近流浪猫，保持低姿态，等待猫的反应",
    "cat_approach_bonded":        "流浪猫已经接受了小饼，小饼友好地靠近",
    "cat_follow_to_corner":       "流浪猫带着小饼走向废弃角落，在一件外套旁边停下来",

    # 入夜
    "night_ending_approach":      "天黑了，公园里只剩小饼一个，它在门口张望",

    # 通用
    "unknown_action":             "小饼不明白该怎么做，歪了歪头四处张望",
    "already_explored":           "小饼已经探索过这里了，没有新发现",
}


class BaseNLG(ABC):
    @abstractmethod
    def generate_narration(self, scene_label: str, player_input: str,
                           state: "GameState") -> str:
        pass

    @abstractmethod
    def generate_npc_response(self, npc_name: str, player_input: str,
                              state: "GameState") -> Tuple[str, Dict]:
        pass

    @abstractmethod
    def generate_ending(self, ending_type: str, state: "GameState") -> str:
        pass

    @abstractmethod
    def generate_transition(self, state: "GameState") -> str:
        pass


class DeepSeekNLG(BaseNLG):

    def __init__(self):
        kwargs = {
            "api_key": config.DEEPSEEK_API_KEY,
            "base_url": config.DEEPSEEK_BASE_URL,
        }
        if config.DEEPSEEK_PROXY:
            kwargs["http_client"] = httpx.Client(proxy=config.DEEPSEEK_PROXY)
        self.client = OpenAI(**kwargs)

    def _call(self, system: str, user: str, max_tokens: int = 300,
              temperature: float = 0.85) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[NLG Error] {e}")
            return "小饼愣了一下，公园的风吹过来，它摇了摇尾巴，继续四处张望。"

    def generate_narration(self, scene_label: str, player_input: str,
                           state: "GameState") -> str:
        from game_state import CLUE_META, PHASE_DESC
        phase_desc = PHASE_DESC.get(state.phase, ("", ""))[1]
        clues_summary = "\n".join(
            f"  • {CLUE_META.get(c, c)}" for c in state.clues_found
        ) if state.clues_found else "  • 暂无"
        state_summary = _build_state_facts(state)

        # 获取场景描述，未知标签直接用标签本身
        label_desc = SCENE_LABELS.get(scene_label, scene_label)

        system = _NARRATION_SYSTEM.format(
            phase_desc=phase_desc,
            state_summary=state_summary,
            clues_summary=clues_summary,
            scene_label=label_desc,
            player_input=player_input,
        )
        return self._call(system, "请生成叙事：", max_tokens=250)

    def generate_npc_response(self, npc_name: str, player_input: str,
                              state: "GameState") -> Tuple[str, Dict]:
        state_summary = mem.build_state_summary(npc_name, state)
        history_list = mem.get_npc_history(npc_name, state)
        history_str = _format_history(history_list)

        prompt_template = NPC_PROMPTS.get(npc_name, "")
        if not prompt_template:
            return "……", {}

        system = prompt_template.format(
            state_summary=state_summary,
            history=history_str or "（暂无对话历史）",
        )
        response = self._call(
            system,
            f"玩家/小饼的行为：{player_input}",
            max_tokens=200,
            temperature=0.8,
        )

        flags = {
            "key_info_revealed": "[KEY_INFO]" in response,
            "photo_shown":       "[PHOTO_SHOWN]" in response,
            "hint_bushes":       "[HINT_BUSHES]" in response,
            "chat_ended":        "[END_CHAT]" in response,
        }
        for tag in ["[KEY_INFO]", "[PHOTO_SHOWN]", "[HINT_BUSHES]", "[END_CHAT]"]:
            response = response.replace(tag, "").strip()

        return response, flags

    def generate_ending(self, ending_type: str, state: "GameState") -> str:
        from game_state import CLUE_META, ENDING_META
        clues_summary = "\n".join(
            f"  • {CLUE_META.get(c, c)}" for c in state.clues_found
        ) if state.clues_found else "  • 暂无"
        state_summary = _build_state_facts(state)
        ending_name = ENDING_META.get(ending_type, ending_type)

        system = _ENDING_SYSTEM.format(
            ending_type=ending_type,
            ending_name=ending_name,
            state_summary=state_summary,
            clues_summary=clues_summary,
        )
        return self._call(system, "请生成结局叙事：", max_tokens=450, temperature=0.9)

    def generate_transition(self, state: "GameState") -> str:
        from game_state import CLUE_META
        clues_summary = "\n".join(
            f"  • {CLUE_META.get(c, c)}" for c in state.clues_found
        ) if state.clues_found else "  • 暂无"
        state_summary = _build_state_facts(state)
        system = _TRANSITION_SYSTEM.format(
            state_summary=state_summary,
            clues_summary=clues_summary,
        )
        return self._call(system, "请生成入夜过渡叙事：", max_tokens=250, temperature=0.85)


# =====================================================
# 辅助函数
# =====================================================

def _build_state_facts(state: "GameState") -> str:
    """描述当下现实状态，不推断历史"""
    parts = []
    if state.flag_grandma_bonded:
        parts.append("老奶奶已对小饼友好")
    if state.flag_direction_known:
        parts.append("小饼知道主人去了全家便利店")
    if state.flag_cat_ally:
        parts.append("流浪猫小灰已接受小饼")
    if state.flag_jogger_warned and not state.flag_jogger_danger:
        parts.append("慢跑者已被小饼惊扰过，态度不好")
    if state.flag_jogger_danger:
        parts.append("慢跑者正在愤怒中，危险进行中")
    if state.flag_danger_resolved:
        parts.append("危险已解除，慢跑者消气了")
    if state.clues_found:
        from game_state import CLUE_META
        names = [CLUE_META.get(c, c) for c in state.clues_found]
        parts.append(f"已有线索：{', '.join(names)}")
    return "，".join(parts) if parts else "游戏刚开始，什么都还没发生"


def _format_history(history: List[Dict]) -> str:
    if not history:
        return ""
    lines = []
    for h in history[-6:]:
        role = "小饼（玩家）" if h["role"] == "user" else "NPC"
        lines.append(f"{role}：{h['content']}")
    return "\n".join(lines)
