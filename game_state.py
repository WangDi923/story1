"""
游戏状态管理
============
流浪小狗小饼的冒险：三条主线 × 双场景 × 好感度系统
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


# =====================================================
# 场景元数据
# =====================================================

SCENE_META = {
    "park": {
        "name": "城市公园",
        "max_turns": 10,
        "npcs": ["squirrel", "grandma", "vendor"],
        "locations": ["tree", "bushes", "fountain", "bench", "path"],
    },
    "forest": {
        "name": "城郊森林",
        "max_turns": 12,
        "npcs": ["owl", "hedgehog"],
        "locations": ["oak_tree", "clearing", "burrow", "stash"],
    },
    "market": {
        "name": "菜市场",
        "max_turns": 12,
        "npcs": ["fish_vendor", "noodle_lady"],
        "locations": ["fish_stall", "veggie_stall", "noodle_shop", "cart"],
    },
    "ruins": {
        "name": "废墟之家",
        "max_turns": 12,
        "npcs": ["old_cat", "zhang"],
        "locations": ["iron_gate", "room", "doorstep"],
    },
}

# 场景2对应的主线NPC（进入该场景时跟随的NPC）
SCENE_MAINLINE = {
    "park": None,
    "forest": "squirrel",
    "market": "grandma",
    "ruins": "vendor",
}

# 公园NPC → 对应场景2
NPC_NEXT_SCENE = {
    "squirrel": "forest",
    "grandma": "market",
    "vendor": "ruins",
}

# 公园NPC进入场景2的好感阈值
NPC_SCENE2_THRESHOLD = {
    "squirrel": 40,
    "grandma": 30,
    "vendor": 40,
}

# 场景2搞砸阈值（好感低于此 → 流浪结局）
SCENE2_FAIL_THRESHOLD = {
    "forest": 20,
    "market": 15,
    "ruins": 20,
}


# =====================================================
# 好感度行为映射
# =====================================================

AFFINITY_ACTIONS: Dict[tuple, int] = {
    # === 公园 - 松鼠 ===
    ("squirrel", "approach"):       +10,
    ("squirrel", "dialogue"):       +5,
    ("squirrel", "give_pine_nuts"): +30,
    ("squirrel", "bark_at"):        -15,

    # === 公园 - 老奶奶 ===
    ("grandma", "approach"):        +10,
    ("grandma", "listen_story"):    +15,
    ("grandma", "eat_snack"):       +15,
    ("grandma", "bite_pants"):      -20,
    ("grandma", "bark_at"):         -20,

    # === 公园 - 卖气球大爷 ===
    ("vendor", "approach"):         +10,
    ("vendor", "dialogue"):         +5,
    ("vendor", "help_sell"):        +15,
    ("vendor", "sell_success"):     +10,
    ("vendor", "pop_balloon"):      -15,

    # === 森林（挂在松鼠主线下） ===
    ("squirrel", "owl_dialogue"):       +5,
    ("squirrel", "visit_hedgehog"):     +5,
    ("squirrel", "chase_wildcat"):      +20,
    ("squirrel", "bark_owl"):           -15,
    ("squirrel", "steal_stash"):        -25,
    ("squirrel", "abandon_guard"):      -20,

    # === 菜市场（挂在老奶奶主线下） ===
    ("grandma", "meet_fish_vendor"):    +5,
    ("grandma", "guard_cart"):          +15,
    ("grandma", "visit_noodle_shop"):   +10,
    ("grandma", "rampage_stall"):       -15,
    ("grandma", "steal_fish"):          -20,
    ("grandma", "abandon_cart"):        -20,
    ("grandma", "bark_passerby"):       -15,

    # === 废墟（挂在大爷主线下） ===
    ("vendor", "befriend_cat"):     +5,
    ("vendor", "zhang_visit"):      +5,
    ("vendor", "night_company"):    +10,
    ("vendor", "bully_cat"):        -20,
    ("vendor", "destroy_stuff"):    -15,
    ("vendor", "bark_zhang"):       -15,
    ("vendor", "restless_night"):   -10,
}


# =====================================================
# 结局
# =====================================================

ENDING_META = {
    "forest_home":  "🌲 森林之家",
    "grandma_home": "🏠 奶奶的家",
    "ruins_home":   "🎈 废墟里的家",
    "stray":        "🌙 流浪",
}


# =====================================================
# 实体映射
# =====================================================

ALL_NPCS = [
    "squirrel", "grandma", "vendor",          # 公园
    "owl", "hedgehog",                         # 森林
    "fish_vendor", "noodle_lady",              # 菜市场
    "old_cat", "zhang",                        # 废墟
]

ENTITY_MAP = {
    # 公园NPC
    "squirrel":     ["松鼠", "小松鼠"],
    "grandma":      ["老奶奶", "奶奶", "老太太", "老人家"],
    "vendor":       ["大爷", "卖气球的", "卖气球大爷", "气球大爷", "老大爷"],
    # 森林NPC
    "owl":          ["猫头鹰", "夜猫子"],
    "hedgehog":     ["刺猬", "刺猬一家", "刺猬妈妈"],
    # 菜市场NPC
    "fish_vendor":  ["卖鱼的", "卖鱼大叔", "大叔", "鱼贩"],
    "noodle_lady":  ["老板娘", "面馆老板娘", "面馆"],
    # 废墟NPC
    "old_cat":      ["阿黄", "老猫", "猫", "猫咪"],
    "zhang":        ["张叔", "收废品的", "邻居"],
    # 公园地点
    "tree":         ["大树", "树", "树下"],
    "bushes":       ["草丛", "灌木", "草地"],
    "fountain":     ["喷泉", "水池"],
    "bench":        ["长椅", "椅子", "凳子"],
    "path":         ["小路", "路", "公园小路"],
    # 森林地点
    "oak_tree":     ["老橡树", "橡树", "大橡树"],
    "clearing":     ["空地", "林间空地"],
    "burrow":       ["树根窝", "树根", "刺猬窝", "窝"],
    "stash":        ["存粮处", "存粮", "蘑菇堆", "洞口"],
    # 菜市场地点
    "fish_stall":   ["鱼摊", "鱼"],
    "veggie_stall": ["菜摊", "蔬菜摊"],
    "noodle_shop":  ["面馆", "面店"],
    "cart":         ["推车", "小推车"],
    # 废墟地点
    "iron_gate":    ["铁门", "大门", "门"],
    "room":         ["房间", "屋里", "大爷的房间"],
    "doorstep":     ["门口", "门前"],
    # 物品
    "pine_nuts":    ["松果", "松子"],
}


def normalize_entity(raw: str) -> str:
    if not raw:
        return ""
    for key, aliases in ENTITY_MAP.items():
        if any(alias in raw for alias in aliases):
            return key
    return raw


# =====================================================
# 游戏状态
# =====================================================

@dataclass
class GameState:
    # 场景
    current_scene: str = "park"
    scene_turn_count: int = 0
    total_turn_count: int = 0

    # 好感度（0-100，只有三个主线NPC有好感度）
    npc_affinity: Dict[str, int] = field(default_factory=lambda: {
        "squirrel": 0,
        "grandma": 0,
        "vendor": 0,
    })

    # 任务进度（每个主线NPC的当前步骤）
    npc_quest_step: Dict[str, int] = field(default_factory=lambda: {
        "squirrel": 0,
        "grandma": 0,
        "vendor": 0,
    })

    # 背包
    inventory: List[str] = field(default_factory=list)

    # 探索记录（按场景分区）
    explored: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {"park": {}, "forest": {}, "market": {}, "ruins": {}}
    )

    # 对话状态
    in_dialogue: bool = False
    dialogue_target: Optional[str] = None

    # 观察状态
    observe_target: Optional[str] = None

    # NPC 对话历史
    npc_memories: Dict[str, List[Dict[str, str]]] = field(default_factory=lambda: {
        npc: [] for npc in ALL_NPCS
    })

    # 结局
    game_over: bool = False
    ending: Optional[str] = None

    # 日志
    last_nlu_log: Dict[str, Any] = field(default_factory=dict)

    # ---------- 辅助方法 ----------

    def adjust_affinity(self, npc: str, action_tag: str) -> int:
        """调整好感度，返回变化量"""
        delta = AFFINITY_ACTIONS.get((npc, action_tag), 0)
        if delta == 0:
            return 0
        old = self.npc_affinity.get(npc, 0)
        self.npc_affinity[npc] = max(0, min(100, old + delta))
        return delta

    def get_mainline_npc(self) -> Optional[str]:
        """当前场景的主线NPC"""
        return SCENE_MAINLINE.get(self.current_scene)

    def get_mainline_affinity(self) -> int:
        """当前场景主线好感度"""
        npc = self.get_mainline_npc()
        if npc:
            return self.npc_affinity.get(npc, 0)
        return 0

    def mark_explored(self, entity: str):
        scene = self.current_scene
        self.explored[scene][entity] = self.explored[scene].get(entity, 0) + 1

    def is_explored(self, entity: str) -> bool:
        scene = self.current_scene
        return self.explored.get(scene, {}).get(entity, 0) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene": self.current_scene,
            "scene_turn": self.scene_turn_count,
            "total_turn": self.total_turn_count,
            "affinity": dict(self.npc_affinity),
            "quest_step": dict(self.npc_quest_step),
            "inventory": list(self.inventory),
            "explored": {k: dict(v) for k, v in self.explored.items()},
            "in_dialogue": self.in_dialogue,
            "dialogue_target": self.dialogue_target,
            "game_over": self.game_over,
            "ending": self.ending,
        }
