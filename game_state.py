"""
游戏状态管理
============
小动物和 NPC 统一处理，均可进入对话模式。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

CLUE_META = {
    "bench_scent":      "长椅上的气味——小明不久前在这里待过",
    "key_holder":       "草丛里的钥匙扣——小明走得很急",
    "grandma_witness":  "老奶奶目击——小明跑向公园对面的全家便利店",
    "vendor_memory":    "摊主的记忆——有人跑向公园门口方向",
    "cat_outfit_card":  "小明的外套和全家会员卡（门店：公园正门对面38号）",
    "jogger_photo":     "🌟慢跑者手机里的照片——拍到了小明的背影（彩蛋）",
}

PHASE_DESC = {
    1: ("🌅 傍晚", "公园里还有不少人，阳光斜斜地打在石板路上。"),
    2: ("🌆 黄昏", "路灯开始亮起，公园里的人少了许多。"),
    3: ("🌃 入夜", "公园的灯全亮了，夜风带着凉意。"),
}

ENDING_META = {
    "happy":           "🎉 圆满结局",
    "solo":            "🌙 独自出走结局",
    "adopted_grandma": "🏠 被老奶奶收养结局",
    "adopted_cat":     "🐱 跟流浪猫留下结局",
    "stray":           "🌌 流浪结局",
    "bad":             "💀 坏结局",
}

# 所有可对话角色
ALL_NPCS = ["grandma", "vendor", "jogger", "cat", "squirrel", "pigeons"]

ENTITY_MAP = {
    "grandma":         ["老奶奶", "奶奶", "老太太", "老人家", "老人"],
    "vendor":          ["摊主", "气球摊主", "卖气球的", "卖气球"],
    "jogger":          ["慢跑者", "跑步的", "跑步的人", "戴耳机的"],
    "cat":             ["流浪猫", "猫", "小灰", "猫咪", "橘猫", "花猫"],
    "squirrel":        ["松鼠"],
    "pigeons":         ["鸽子", "鸽子群", "一群鸽子", "鸽子们"],
    "bench":           ["长椅", "椅子", "木椅", "长凳"],
    "bushes":          ["草丛", "灌木", "草地", "草"],
    "fountain":        ["喷泉", "水池"],
    "entrance":        ["公园门口", "公园入口", "大门", "门口", "全家", "便利店", "出口"],
    "corner":          ["废弃角落", "角落", "纸箱", "废弃的角落"],
    "track":           ["跑道", "跑步道"],
    "outfit":          ["外套", "衣服", "大衣"],
    "key_holder":      ["钥匙扣", "钥匙"],
    "membership_card": ["会员卡", "卡片", "小卡片"],
}


def normalize_entity(raw: str) -> str:
    if not raw:
        return ""
    for key, aliases in ENTITY_MAP.items():
        if any(alias in raw for alias in aliases):
            return key
    return raw


@dataclass
class GameState:
    turn_count: int = 0
    phase: int = 1

    # 关系 flag
    grandma_bond_progress: int = 0
    flag_grandma_bonded: bool = False
    cat_approach_progress: int = 0
    flag_cat_ally: bool = False

    # 危险 flag
    flag_jogger_warned: bool = False
    flag_jogger_danger: bool = False
    flag_danger_resolved: bool = False
    flag_rescue_failed: bool = False

    # 剧情 flag
    flag_direction_known: bool = False
    flag_jogger_photo_revealed: bool = False
    flag_night_transition_shown: bool = False  # 入夜过渡叙事已显示

    # 线索
    clues_found: List[str] = field(default_factory=list)
    has_membership_card: bool = False

    # 探索记录
    explored: Dict[str, int] = field(default_factory=dict)

    # 对话状态（所有角色统一）
    in_dialogue: bool = False
    dialogue_target: Optional[str] = None

    # 观察状态
    observe_target: Optional[str] = None

    # NPC & 动物对话历史（统一管理）
    npc_memories: Dict[str, List[Dict[str, str]]] = field(default_factory=lambda: {
        "grandma":  [],
        "vendor":   [],
        "jogger":   [],
        "cat":      [],
        "squirrel": [],
        "pigeons":  [],
    })

    game_over: bool = False
    ending: Optional[str] = None
    last_nlu_log: Dict[str, Any] = field(default_factory=dict)

    def add_clue(self, clue_id: str) -> bool:
        if clue_id not in self.clues_found:
            self.clues_found.append(clue_id)
            return True
        return False

    def mark_explored(self, entity: str):
        self.explored[entity] = self.explored.get(entity, 0) + 1

    def is_explored(self, entity: str) -> bool:
        return self.explored.get(entity, 0) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn": self.turn_count,
            "phase": self.phase,
            "flags": {
                "grandma_bonded": self.flag_grandma_bonded,
                "cat_ally": self.flag_cat_ally,
                "jogger_warned": self.flag_jogger_warned,
                "jogger_danger": self.flag_jogger_danger,
                "danger_resolved": self.flag_danger_resolved,
                "rescue_failed": self.flag_rescue_failed,
                "direction_known": self.flag_direction_known,
                "jogger_photo": self.flag_jogger_photo_revealed,
                "night_transition": self.flag_night_transition_shown,
            },
            "clues": self.clues_found,
            "has_membership_card": self.has_membership_card,
            "explored": self.explored,
            "in_dialogue": self.in_dialogue,
            "dialogue_target": self.dialogue_target,
            "game_over": self.game_over,
            "ending": self.ending,
        }
