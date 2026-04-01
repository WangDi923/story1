"""
NLG 模块
========
全新版本：9个NPC + 40+场景标签 + 场景切换 + 结局叙述
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
# NPC System Prompts
# =====================================================

_SQUIRREL_PROMPT = """你正在扮演《寻家记》中的小松鼠，住在公园大树上。

【角色特点】
- 活泼好动，话多，注意力不集中，容易跑题
- 嘴馋，最爱吃松果
- 好奇心强，对小饼这只流浪狗有点好奇
- 偶尔用"吱！"表达情绪

【当前状态】
{state_summary}

【对话历史】
{history}

【规则】
- 回应50字以内，活泼跳跃
- 只输出你说的话或动作，不要旁白
- 根据状态提示决定是否加标记（[WANTS_NUTS] [QUEST_COMPLETE] [END_CHAT]）"""

_GRANDMA_PROMPT = """你正在扮演《寻家记》中的老奶奶，坐在公园喷泉边的长椅上。

【角色特点】
- 慈祥温柔，喜欢小动物，爱唠叨
- 经常来公园遛弯，年轻时养过一只很像小饼的狗
- 说话慢条斯理，语气像在和孙辈说话
- 包里总带着小饼干

【当前状态】
{state_summary}

【对话历史】
{history}

【规则】
- 回应80字以内，温柔慈祥
- 只输出你说的话或动作
- 根据状态提示决定是否加标记（[LISTENED_STORY] [ATE_SNACK] [BITE_PANTS] [QUEST_COMPLETE] [END_CHAT]）"""

_VENDOR_PROMPT = """你正在扮演《寻家记》中卖气球的大爷，推着气球车停在公园小路旁。

【角色特点】
- 有点落寞，话不多但很温和
- 一个人生活，生意不太好
- 看到小饼觉得挺有缘的——都是一个人
- 偶尔叹口气，但笑起来很温暖

【当前状态】
{state_summary}

【对话历史】
{history}

【规则】
- 回应80字以内，温和略带落寞
- 只输出你说的话或动作
- 根据状态提示决定是否加标记（[HELP_SELL] [SELL_SUCCESS] [QUEST_COMPLETE] [END_CHAT]）"""

_OWL_PROMPT = """你正在扮演《寻家记》中的猫头鹰，住在森林老橡树上。

【角色特点】
- 沉稳，话少但说得准，是森林的"情报员"
- 知道森林里所有动物的事
- 对陌生动物有点傲慢，但不坏
- 说话简短，有时闭着一只眼睛

【你知道的事】
最近森林边上来了一只野猫，总是偷刺猬一家藏在树根下的蘑菇存粮，刺猬一家很发愁。

【当前状态】
{state_summary}

【对话历史】
{history}

【规则】
- 回应50字以内，简短沉稳
- 只输出你说的话或动作
- 如果对话自然结束，末尾加 [END_CHAT]"""

_HEDGEHOG_PROMPT = """你正在扮演《寻家记》中的刺猬一家，住在森林树根下的窝里。

【角色特点】
- 你是刺猬妈妈，带着两个刺猬宝宝
- 热情好客，但最近很焦虑因为存粮被偷
- 说话带着叹气和担忧
- 对帮忙的动物非常感激

【你的困扰】
辛苦攒的蘑菇被偷了好几次，不知道是谁干的（其实是野猫）。蘑菇存在旁边的【存粮处】，你希望有人能去存粮处蹲守帮忙抓小偷。

【当前状态】
{state_summary}

【对话历史】
{history}

【规则】
- 回应60字以内，焦虑但热情
- 只输出你说的话或动作
- 如果对话自然结束，末尾加 [END_CHAT]"""

_FISH_VENDOR_PROMPT = """你正在扮演《寻家记》中卖鱼的大叔，在菜市场开鱼摊。

【角色特点】
- 嗓门大，性格豪爽
- 认识老奶奶（王奶奶），是老客户
- 总喜欢多塞两条小鱼给熟客
- 对小动物无所谓，但看小饼跟着老奶奶觉得挺有趣

【当前状态】
{state_summary}

【对话历史】
{history}

【规则】
- 回应60字以内，豪爽大嗓门
- 只输出你说的话或动作
- 如果对话自然结束，末尾加 [END_CHAT]"""

_NOODLE_LADY_PROMPT = """你正在扮演《寻家记》中面馆老板娘，在菜市场尽头开面馆。

【角色特点】
- 心善，安静，不多话
- 认识老奶奶
- 经常给流浪动物留剩饭
- 看到小饼会偷偷多给一碟肉末

【当前状态】
{state_summary}

【对话历史】
{history}

【规则】
- 回应50字以内，温柔安静
- 只输出你说的话或动作
- 如果对话自然结束，末尾加 [END_CHAT]"""

_OLD_CAT_PROMPT = """你正在扮演《寻家记》中的流浪猫阿黄，住在大爷的废墟之家。

【角色特点】
- 大爷收留的老猫，慵懒但心眼好
- 话少，动作多，喜欢蹭来蹭去
- 对新来的小饼好奇，会主动亲近
- 偶尔"喵~"一声

【当前状态】
{state_summary}

【对话历史】
{history}

【规则】
- 回应40字以内，慵懒温柔，多用动作描写
- 只输出你的动作或偶尔的叫声
- 如果对话自然结束，末尾加 [END_CHAT]"""

_ZHANG_PROMPT = """你正在扮演《寻家记》中收废品的张叔，住在大爷隔壁的棚子里。

【角色特点】
- 朴实热心，大爷的老邻居
- 说话直来直去，带点粗犷
- 偶尔过来串门，会带点吃的分享
- 看到小饼觉得大爷终于有个伴了

【当前状态】
{state_summary}

【对话历史】
{history}

【规则】
- 回应60字以内，朴实直接
- 只输出你说的话或动作
- 如果对话自然结束，末尾加 [END_CHAT]"""

NPC_PROMPTS = {
    "squirrel":    _SQUIRREL_PROMPT,
    "grandma":     _GRANDMA_PROMPT,
    "vendor":      _VENDOR_PROMPT,
    "owl":         _OWL_PROMPT,
    "hedgehog":    _HEDGEHOG_PROMPT,
    "fish_vendor": _FISH_VENDOR_PROMPT,
    "noodle_lady": _NOODLE_LADY_PROMPT,
    "old_cat":     _OLD_CAT_PROMPT,
    "zhang":       _ZHANG_PROMPT,
}

NPC_DISPLAY = {
    "squirrel":    "小松鼠",
    "grandma":     "老奶奶",
    "vendor":      "卖气球的大爷",
    "owl":         "猫头鹰",
    "hedgehog":    "刺猬一家",
    "fish_vendor": "卖鱼大叔",
    "noodle_lady": "面馆老板娘",
    "old_cat":     "阿黄",
    "zhang":       "张叔",
}

# =====================================================
# 场景标签
# =====================================================

SCENE_LABELS = {
    # === 公园 - 探索 ===
    "explore_tree":         "小饼走到大树下，抬头看了看，树上有只松鼠在啃东西",
    "explore_bushes":       "小饼凑近草丛，鼻子贴着地面嗅来嗅去",
    "explore_fountain":     "小饼走到喷泉边，水花溅到鼻子上，它打了个喷嚏",
    "explore_bench":        "小饼钻到长椅下面闻了闻，这是它常睡觉的地方",
    "explore_path":         "小饼沿着小路走了走，看到远处有个推着气球车的人",
    "sniff_bushes_new":     "小饼把鼻子埋进草丛，拨开落叶仔细嗅探",
    "sniff_bushes_again":   "小饼再次嗅探草丛，没有新发现",

    # === 公园 - 松鼠任务 ===
    "squirrel_first_look":      "大树上的松鼠歪着头看了小饼一眼，尾巴一抖",
    "find_pine_nuts":           "小饼在草丛深处用鼻子拱开落叶，发现了几颗松果！它小心翼翼地叼起来，也许可以带回去给松鼠",

    # === 公园 - 老奶奶任务 ===
    "grandma_first_look":   "喷泉边的长椅上，老奶奶正在晒太阳，膝上放着一个布包",
    "grandma_pants_bite":   "小饼咬住了老奶奶的裤脚，老奶奶吓了一跳往后缩",

    # === 公园 - 大爷任务 ===
    "vendor_first_look":    "小路旁停着一辆气球车，五颜六色的气球在风里晃。大爷靠在车上发呆",
    "vendor_help_sell":     "小饼站在气球车旁边摇尾巴，一个小朋友被吸引过来了",
    "vendor_sell_success":  "小朋友买了一个红色气球，大爷的脸上终于有了笑容",

    # === 场景切换 ===
    "transition_to_forest": "松鼠跳到小饼面前，回头望了望公园后面的小路——'跟我来！'",
    "transition_to_market": "老奶奶站起来拍拍裙子，拉着小推车向小饼招招手：'走吧，跟奶奶去买菜。'",
    "transition_to_ruins":  "大爷收起气球车，低头看看小饼：'走吧，跟我回去。'铁轮在石板路上咯咯响",

    # === 森林 ===
    "forest_arrive":        "穿过一条长满杂草的小路，眼前豁然开朗——高大的树木，洒下的阳光，鸟叫虫鸣。松鼠跳上树枝回头招呼：'吱！先带你去认识猫头鹰大叔，它住在老橡树上！'",
    "forest_explore_oak":   "小饼走到老橡树下，抬头往上看，树枝上有一双圆圆的眼睛在盯着它",
    "forest_explore_clearing": "林间空地上长着蘑菇，空气里有松针的味道",
    "forest_explore_burrow":"树根下有个小洞，洞口堆着干草，里面传来窸窸窣窣的声音",
    "forest_explore_stash": "一堆蘑菇整齐地码在树根旁，但有的被扒拉得乱七八糟。突然，一个黑影从灌木丛里蹑手蹑脚地走出来——是只野猫！它贪婪地盯着蘑菇堆，做出要偷吃的样子。",
    "forest_discover_wildcat": "一堆蘑菇整齐地码在树根旁。突然，一个黑影从灌木丛里蹑手蹑脚地走出来——是只野猫！它贪婪地盯着蘑菇堆，做出要偷吃的样子。现在该怎么办呢？",
    "forest_hide_waiting": "小饼在蘑菇堆附近蹲下来，屏住呼吸等待。不一会儿，一只野猫从灌木丛边缘探头探脑，眼睛在黑暗中闪着幽绿的光。它发现了蘑菇堆，开始慢慢地往这边靠近……",
    "forest_chase_wildcat": "小饼猛地站起来汪了两声，野猫被这突然的吠叫吓了一大跳！它尖叫一声转身就跑，一溜烟消失在树林深处",
    "forest_steal_stash":   "小饼忍不住凑近蘑菇堆闻了闻，嘴巴拱了一下——刺猬妈妈急了",
    "forest_bark_owl":      "小饼对着树上的猫头鹰叫了两声，猫头鹰皱起眉头，翅膀一扇飞到更高的树枝上",
    "forest_abandon_guard": "小饼蹲了一会儿就跑开了，没有坚持守住存粮",

    # === 菜市场 ===
    "market_arrive":        "熙熙攘攘的菜市场，叫卖声此起彼伏。鱼腥味、青菜味、包子香混在一起，小饼的鼻子忙不过来。老奶奶把小推车停在路边，弯下腰摸摸小饼的头：'乖孩子，奶奶去挑菜，你帮奶奶看着推车好不好？'",
    "market_explore_fish":  "鱼摊上摆着各种鲜鱼，水箱里的鱼还在蹦跶。卖鱼大叔正在吆喝",
    "market_explore_veggie":"菜摊上堆着绿油油的蔬菜，老奶奶正在认真挑选",
    "market_explore_noodle":"面馆里飘出一阵香味，老板娘正在煮面",
    "market_explore_cart":  "老奶奶的小推车停在路边，里面已经放了一些菜",
    "market_guard_cart":    "老奶奶去挑菜了，小饼蹲在推车旁边守着。一个小孩跑过来伸手想拿水果，小饼轻轻叫了一声",
    "market_guard_success": "小孩被吓跑了，老奶奶回来看到东西都在，笑着摸了摸小饼的头",
    "market_abandon_cart":  "小饼没守住，自己跑去闻鱼摊了。老奶奶回来发现推车被人动过",
    "market_steal_fish":    "小饼没忍住，叼了一条鱼就跑。卖鱼大叔大喊一声，老奶奶脸都红了",
    "market_rampage":       "小饼在菜市场里乱窜，撞翻了一个菜筐，周围人纷纷躲开",
    "market_bark_people":   "小饼对着路人汪汪叫，有人被吓到了，老奶奶赶紧拉住它",

    # === 废墟 ===
    "ruins_arrive":         "城市边缘，一片旧厂房改的住所。大爷推开生锈的铁门，里面出乎意料地干净。门口挂着易拉罐做的风铃。大爷指了指角落里一只老猫：'来，我给你介绍一下阿黄，它也住这儿。'",
    "ruins_explore_gate":   "铁门上锈迹斑斑，但门框被擦得很干净。风铃在晚风里叮叮当当",
    "ruins_explore_room":   "不大的房间，一张床，一个柜子，墙上贴着旧报纸。角落里有一条叠好的旧毯子",
    "ruins_explore_doorstep":"门口放着两个小马扎，大爷平时就坐这里看天。远处是城市的灯光",
    "ruins_befriend_cat":   "阿黄慢悠悠地走过来，蹭了蹭小饼的脸。小饼愣了一下，然后也蹭了回去",
    "ruins_bully_cat":      "小饼追着阿黄跑，阿黄吓得钻到床底下。大爷皱起了眉头",
    "ruins_zhang_arrive":   "隔壁棚子里传来脚步声，一个提着袋子的中年人推门进来——'老李，我来了'",
    "ruins_bark_zhang":     "小饼对着张叔汪汪叫，张叔退了一步。大爷赶紧拍拍小饼",
    "ruins_destroy":        "小饼把大爷叠好的毯子拖下来咬了几口，大爷叹了口气",
    "ruins_night":          "夜深了，大爷坐在门口的小马扎上，远处城市的灯光一点一点亮起来",
    "ruins_restless":       "小饼在门口转来转去，好像想出去。大爷看了看它，没说话",

    # === 搞砸后被劝退 ===
    "forest_kicked_out":    "松鼠低着头，小声说'要不……你还是回公园吧。'猫头鹰转过头去不看小饼",
    "market_kicked_out":    "老奶奶叹了口气，'这孩子太野了，奶奶管不住啊……'她慢慢走远了",
    "ruins_kicked_out":     "大爷打开门，沉默了一会儿，说'你走吧，我也不勉强你。'",

    # === 通用 ===
    "unknown_action":       "小饼歪了歪头，不太明白该怎么做",
    "already_explored":     "小饼已经看过这里了，没什么新发现",
    "npc_not_here":         "小饼四处张望了一下，这里没有这个角色",
}


# =====================================================
# 叙事系统提示
# =====================================================

_NARRATION_SYSTEM = """你是文字冒险游戏《寻家记》的叙事引擎。
主角是流浪小狗小饼，它没有主人，在城市里流浪。

【叙事风格】
- 第三人称（"小饼……"、"它……"）
- 温馨治愈，略带幽默，偶尔用狗的视角制造反差萌
- 多描写嗅觉感官（嗅觉>听觉>视觉）
- 场景氛围：{scene_desc}
- 每段80~150字

【重要规则】
- 只描述这一刻正在发生的事情
- 禁止推断之前发生过什么
- 只描述【本次行动场景】里指定的内容

【当前状态】（只做参考，不要在叙事里回顾）
{state_summary}

【本次行动场景】
{scene_label}

玩家原始输入：{player_input}

只输出叙事文字，不要解释，不要加任何标记。"""

_SCENE_DESC = {
    "park":   "傍晚的城市公园，阳光斜照，暖洋洋的。长椅、喷泉、草丛、大树，还有来来往往的人",
    "forest": "午后的城郊森林，树木高大，阳光从叶缝洒下。鸟叫虫鸣，空气里有泥土和松针的味道",
    "market": "热闹的露天菜市场，叫卖声此起彼伏。鱼腥味、青菜味、包子香混在一起，地上湿漉漉的",
    "ruins":  "城市边缘的旧厂房，墙皮斑驳，铁门生锈。但里面收拾得干干净净，门口挂着易拉罐风铃",
}

_ENDING_SYSTEM = """你是文字冒险游戏《寻家记》的结局叙事引擎。
小饼是一只流浪小狗，它没有主人，一直在城市里独自流浪。

请为以下结局生成完整结局叙事（200~300字）：
结局类型：{ending_type}
结局名称：{ending_name}
当前状态：{state_summary}

要求：
- 情感丰富，有画面感，有余韵
- 体现小饼从流浪到找到归属的变化
- 第三人称叙述
- 结尾一句话做总结

只输出叙事文字。"""

_TRANSITION_SYSTEM = """你是文字冒险游戏《寻家记》的叙事引擎。

小饼是一只流浪小狗，它正要从{from_scene}去{to_scene}。
{escort_desc}

请生成一段场景切换叙事（120~180字）描述：
- 离开当前场景的画面
- 路上的感受（嗅觉为主）
- 到达新场景的第一印象

只输出叙事文字，第三人称，不要解释。"""


# =====================================================
# NLG 实现
# =====================================================

class BaseNLG(ABC):
    @abstractmethod
    def generate_narration(self, scene_label: str, player_input: str,
                           state: "GameState") -> str: ...

    @abstractmethod
    def generate_npc_response(self, npc_name: str, player_input: str,
                              state: "GameState") -> Tuple[str, Dict]: ...

    @abstractmethod
    def generate_ending(self, ending_type: str, state: "GameState") -> str: ...

    @abstractmethod
    def generate_scene_transition(self, from_scene: str, to_scene: str,
                                  escort_npc: str, state: "GameState") -> str: ...


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
            return "小饼愣了一下，摇了摇尾巴，继续四处张望。"

    def generate_narration(self, scene_label: str, player_input: str,
                           state: "GameState") -> str:
        scene_desc = _SCENE_DESC.get(state.current_scene, "")
        state_summary = _build_state_facts(state)
        label_desc = SCENE_LABELS.get(scene_label, scene_label)

        system = _NARRATION_SYSTEM.format(
            scene_desc=scene_desc,
            state_summary=state_summary,
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
            f"小饼的行为：{player_input}",
            max_tokens=200,
            temperature=0.8,
        )

        # 提取所有标记
        all_tags = [
            "[WANTS_NUTS]", "[QUEST_COMPLETE]", "[END_CHAT]",
            "[LISTENED_STORY]", "[ATE_SNACK]", "[BITE_PANTS]",
            "[HELP_SELL]", "[SELL_SUCCESS]", "[VENDOR_DIALOGUE]",
        ]
        flags = {}
        for tag in all_tags:
            key = tag.strip("[]").lower()
            if tag in response:
                flags[key] = True
                response = response.replace(tag, "").strip()

        return response, flags

    def generate_ending(self, ending_type: str, state: "GameState") -> str:
        from game_state import ENDING_META
        ending_name = ENDING_META.get(ending_type, ending_type)
        state_summary = _build_state_facts(state)

        system = _ENDING_SYSTEM.format(
            ending_type=ending_type,
            ending_name=ending_name,
            state_summary=state_summary,
        )
        return self._call(system, "请生成结局叙事：", max_tokens=450, temperature=0.9)

    def generate_scene_transition(self, from_scene: str, to_scene: str,
                                  escort_npc: str, state: "GameState") -> str:
        from game_state import SCENE_META
        from_name = SCENE_META.get(from_scene, {}).get("name", from_scene)
        to_name = SCENE_META.get(to_scene, {}).get("name", to_scene)
        escort_display = NPC_DISPLAY.get(escort_npc, escort_npc)
        escort_desc = f"{escort_display}带着小饼一起走。"

        system = _TRANSITION_SYSTEM.format(
            from_scene=from_name,
            to_scene=to_name,
            escort_desc=escort_desc,
        )
        return self._call(system, "请生成场景切换叙事：", max_tokens=250, temperature=0.85)


# =====================================================
# 辅助函数
# =====================================================

def _build_state_facts(state: "GameState") -> str:
    parts = []
    from game_state import SCENE_META
    scene_name = SCENE_META.get(state.current_scene, {}).get("name", state.current_scene)
    parts.append(f"当前在{scene_name}")

    for npc, aff in state.npc_affinity.items():
        if aff > 0:
            display = NPC_DISPLAY.get(npc, npc)
            parts.append(f"{display}好感{aff}")

    if state.inventory:
        parts.append(f"背包：{'、'.join(state.inventory)}")

    return "，".join(parts) if parts else "游戏刚开始"


def _format_history(history: List[Dict]) -> str:
    if not history:
        return ""
    lines = []
    for h in history[-6:]:
        role = "小饼（玩家）" if h["role"] == "user" else "NPC"
        lines.append(f"{role}：{h['content']}")
    return "\n".join(lines)
