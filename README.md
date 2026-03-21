---
title: storyweaver
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
python_version: '3.8'
app_file: app.py
pinned: false
---
# 🐾 巴蒂的回家路 — 交互式文本冒险游戏

> NLP 课程设计 · Python 3.8+ · DeepSeek API · Gradio 4.x

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY

# 3. 启动游戏
python main.py
# 浏览器访问 http://localhost:7860
```

---

## 项目结构

```
bati_game/
├── api_client.py      # DeepSeek API 统一封装（NLU/NLG/校验三种调用类型）
├── nlu_module.py      # NLU 意图识别 + 实体抽取（5意图×3实体，置信度机制）
├── nlg_module.py      # NLG 文本生成（NPC回复/剧情/结局/一致性校验）
├── memory_manager.py  # 三层 Memory 架构（静态/动态/轨迹）
├── plot_engine.py     # 剧情引擎（完整游戏循环 + 道具逻辑 + 结局触发）
├── gradio_ui.py       # Gradio 前端界面（暖色系，含技术展示区）
├── main.py            # 主程序入口
├── test_cases.py      # 独立测试套件（3个完整用例）
├── requirements.txt
├── .env.example
└── README.md
```

---

## 核心 NLP 设计

### 1. 意图识别（NLU）

| 意图 | 说明 | 示例输入 |
|------|------|----------|
| EXPLORE | 探索/移动 | 「往街边巷弄走」 |
| INTERACT | 与NPC互动 | 「和路人小姐姐说话」 |
| GET | 获取道具 | 「把手帕拿出来用」 |
| AVOID | 避险 | 「快跑！流浪猫来了」 |
| ASK | 询问信息 | 「怎么回幸福小区」 |

- **置信度机制**：低于 0.7 时调用 NLG 生成澄清反问，不强行猜测意图
- **预留接口**：`extract_intent_entity(text)` 签名不变，可直接替换为自研模型

### 2. 文本生成（NLG）

- **上下文感知**：场景/道具/线索/NPC子记忆全部动态注入 Prompt
- **风格延续**：NPC 个性细节首次生成后锁定，同轮保持一致
- **一致性校验**：语义级（非关键词匹配），不通过时强化约束重试最多 2 次

### 3. 三层 Memory

| 层级 | 类型 | 作用 |
|------|------|------|
| 全局静态 Memory | 只读写死 | NPC配置/道具规则/剧情解锁规则 |
| 会话动态 Memory | 实时更新 | 场景/道具/线索/NPC子记忆 |
| 分支轨迹 Memory | 全程记录 | 每轮输入/NLU/输出/Memory变化 |

---

## 测试用例

```bash
# 运行全部测试（需要有效API Key）
python test_cases.py

# 只运行NLU快速测试
python test_cases.py --nlu-only

# 只运行指定用例
python test_cases.py --case 1   # 核心NPC交互
python test_cases.py --case 2   # 道具使用
python test_cases.py --case 3   # 剧情推进全流程
```

| 用例 | 覆盖内容 |
|------|----------|
| 用例1 | 路人小姐姐交互 / NLG生成 / 一致性校验 / Memory更新 |
| 用例2 | 火腿肠换门禁卡 / 消耗品管理 / 永久品保留 / 错误场景拦截 |
| 用例3 | 全场景推进 / 剧情解锁规则 / E2结局触发 |

---

## 剧情路线

```
超市门口 ──→ 街边巷弄 ──→ 城市公园 / 老街入口 ──→ 老街集市 ──→ 🏠 幸福小区
（初始）    （需持有手帕）  （需路人指路线索）     （需门禁卡+招牌线索）
```

### 三种结局

- 🌟 **完美结局 E1**：集齐手帕+小铃铛+门禁卡，与全部4个核心NPC互动
- 🐕 **暖心结局 E3**：与公园小狗互动，解锁「公园小狗线索」
- 🏠 **普通结局 E2**：到达幸福小区即触发（兜底结局）

---

## 扩展说明

**替换为自研意图识别模型**：仅需修改 `nlu_module.py` 中的 `extract_intent_entity` 函数体，保持返回结构不变即可，`plot_engine.py` 等上层模块无需改动。

**替换为本地生成模型**：仅需修改 `api_client.py` 中的 `call_deepseek_api` 函数，将 HTTP 请求替换为本地模型推理调用，返回 `{"result": str, "error": None}` 格式即可。