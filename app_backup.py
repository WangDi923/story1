"""
StoryWeaver Gradio UI
=====================
全新版本：场景动态切换 + 好感度面板 + 任务追踪
"""
import gradio as gr
from typing import List, Tuple, Any

from game_state import GameState, ENDING_META, SCENE_META, NPC_SCENE2_THRESHOLD
from engine import new_game, process_turn, get_suggestions, OPENING_NARRATION
from nlg import NPC_DISPLAY
from quests import get_quest_display
import logger as glogger


# =====================================================
# 状态面板渲染
# =====================================================

def render_scene_panel(state: GameState) -> str:
    scene = SCENE_META.get(state.current_scene, {})
    scene_name = scene.get("name", state.current_scene)
    lines = [f"## 📍 {scene_name}", ""]

    def _segmented_bar(current: int, total: int, segments: int = 12) -> tuple[str, int]:
        if total <= 0:
            return "▱▱▱▱ ▱▱▱▱ ▱▱▱▱", 0
        ratio = max(0.0, min(1.0, current / total))
        filled = int(round(ratio * segments))
        blocks = ["▰" if i < filled else "▱" for i in range(segments)]
        grouped = ["".join(blocks[i:i + 4]) for i in range(0, segments, 4)]
        return " ".join(grouped), int(ratio * 100)

    # 可互动角色
    npc_ids = list(scene.get("npcs", []))
    mainline = state.get_mainline_npc()
    if mainline and mainline not in npc_ids:
        npc_ids.insert(0, mainline)
    npc_names = [f"【{NPC_DISPLAY.get(n, n)}】" for n in npc_ids]
    lines.append("**可互动角色**")
    lines.append("  ".join(npc_names))
    lines.append("")

    # 可探索地点
    loc_names = {
        "tree": "大树", "bushes": "草丛", "fountain": "喷泉",
        "bench": "长椅", "path": "小路",
        "oak_tree": "老橡树", "clearing": "空地",
        "burrow": "树根窝", "stash": "存粮处",
        "fish_stall": "鱼摊", "veggie_stall": "菜摊",
        "noodle_shop": "面馆", "cart": "推车",
        "iron_gate": "铁门", "room": "房间", "doorstep": "门口",
    }
    locations = [f"【{loc_names.get(l, l)}】" for l in scene.get("locations", [])]
    if locations:
        lines.append("**可探索地点**")
        lines.append("  ".join(locations))
        lines.append("")

    # 场景时间进度
    lines.append("---")
    max_turns = scene.get("max_turns", 20)
    turn = min(state.scene_turn_count, max_turns)
    scene_bar, scene_pct = _segmented_bar(turn, max_turns, segments=12)
    lines.append(f"**🕐 场景进度** ({turn}/{max_turns}) · {scene_pct}%")
    lines.append(f"`{scene_bar}`")
    lines.append("")

    # 本局总进度（公园 + 场景2）
    park_max = SCENE_META.get("park", {}).get("max_turns", 10)
    non_park_max = [m.get("max_turns", 12) for s, m in SCENE_META.items() if s != "park"]
    scene2_max = max(non_park_max) if non_park_max else 12
    total_max = park_max + scene2_max
    total_turn = min(state.total_turn_count, total_max)
    total_bar, total_pct = _segmented_bar(total_turn, total_max, segments=16)
    lines.append(f"**📈 总进度（本局）** ({total_turn}/{total_max}) · {total_pct}%")
    lines.append(f"`{total_bar}`")

    return "\n".join(lines)


def render_affinity_panel(state: GameState) -> str:
    lines = ["## 💛 好感度", ""]

    npc_names = {"squirrel": "小松鼠", "grandma": "老奶奶", "vendor": "大爷"}

    for npc, display in npc_names.items():
        aff = state.npc_affinity.get(npc, 0)
        filled = int(aff / 100 * 10)
        bar = "█" * filled + "░" * (10 - filled)
        threshold = NPC_SCENE2_THRESHOLD.get(npc, 0)
        marker = " ✅" if aff >= threshold else ""
        lines.append(f"**{display}**: `{bar}` {aff}/100{marker}")

    return "\n".join(lines)


def render_quest_panel(state: GameState) -> str:
    lines = ["## 📋 任务", ""]

    npc_names = {"squirrel": "🐿️ 松鼠", "grandma": "👵 老奶奶", "vendor": "🎈 大爷"}

    for npc, display in npc_names.items():
        quest_desc = get_quest_display(npc, state)
        step = state.npc_quest_step.get(npc, 0)
        if quest_desc:
            if step > 0:
                lines.append(f"🟡 {display}：{quest_desc}")
            else:
                lines.append(f"⬜ {display}：{quest_desc}")
        else:
            lines.append(f"✅ {display}：已完成")

    return "\n".join(lines)


def render_ending_panel(state: GameState) -> str:
    if not state.game_over or not state.ending:
        return ""
    lines = ["## 🎮 结局", ""]
    for eid, ename in ENDING_META.items():
        if eid == state.ending:
            lines.append(f"✅ **{ename}**")
        else:
            lines.append(f"⬜ {ename}")
    lines.append("")
    lines.append("*「小饼的故事还有其他可能……」*")
    return "\n".join(lines)


# =====================================================
# 对话状态栏
# =====================================================

def dialogue_updates(state: GameState):
    if state.in_dialogue and state.dialogue_target:
        display = NPC_DISPLAY.get(state.dialogue_target, state.dialogue_target)
        text = f"💬 正在和 【{display}】 对话中"
        return (
            gr.update(value=text, visible=True),
            gr.update(visible=True),
        )
    return (
        gr.update(value="", visible=False),
        gr.update(visible=False),
    )


# =====================================================
# Gradio 事件处理函数
# =====================================================

def on_start():
    state, suggestions = new_game()
    chat = [{"role": "assistant", "content": OPENING_NARRATION}]
    logs = []
    dlg_status, exit_visible = dialogue_updates(state)
    return (
        chat, state, logs,
        render_scene_panel(state),
        render_affinity_panel(state),
        render_quest_panel(state),
        dlg_status, exit_visible,
        render_ending_panel(state),
        *_btn_updates(suggestions),
    )


def on_submit(player_input: str, chat: list, state: GameState, logs: list):
    if not player_input or not player_input.strip():
        return ("",) + _unchanged(chat, state, logs)

    response, state, suggestions, log_entry = process_turn(player_input.strip(), state)
    glogger.write_log(log_entry)
    logs.append(log_entry)
    chat.append({"role": "user", "content": player_input})
    chat.append({"role": "assistant", "content": response})

    dlg_status, exit_visible = dialogue_updates(state)
    return (
        "",
        chat, state, logs,
        render_scene_panel(state),
        render_affinity_panel(state),
        render_quest_panel(state),
        dlg_status, exit_visible,
        render_ending_panel(state),
        glogger.format_log_for_display(logs),
        *_btn_updates(suggestions),
    )


def on_suggestion(btn_text: str, chat: list, state: GameState, logs: list):
    if not btn_text or not btn_text.strip():
        return ("",) + _unchanged(chat, state, logs)
    return on_submit(btn_text, chat, state, logs)


def on_exit_dialogue(chat: list, state: GameState, logs: list):
    if state.in_dialogue and state.dialogue_target:
        display = NPC_DISPLAY.get(state.dialogue_target, state.dialogue_target)
        state.in_dialogue = False
        state.dialogue_target = None
        chat.append({"role": "assistant", "content": f"小饼向【{display}】点点头，转身走开了。"})

    suggestions = get_suggestions(state)
    dlg_status, exit_visible = dialogue_updates(state)
    return (
        "",
        chat, state, logs,
        render_scene_panel(state),
        render_affinity_panel(state),
        render_quest_panel(state),
        dlg_status, exit_visible,
        render_ending_panel(state),
        glogger.format_log_for_display(logs),
        *_btn_updates(suggestions),
    )


def _btn_updates(suggestions: List[str]):
    padded = (suggestions + ["", "", "", ""])[:4]
    return [gr.update(value=p, visible=bool(p)) for p in padded]


def _unchanged(chat, state, logs):
    suggestions = get_suggestions(state)
    dlg_status, exit_visible = dialogue_updates(state)
    return (
        chat, state, logs,
        render_scene_panel(state),
        render_affinity_panel(state),
        render_quest_panel(state),
        dlg_status, exit_visible,
        render_ending_panel(state),
        glogger.format_log_for_display(logs),
        *_btn_updates(suggestions),
    )


# =====================================================
# UI 构建
# =====================================================

def build_ui():
    glogger.init_log_file()

    with gr.Blocks(
        title="寻家记：小饼的城市冒险",
        theme=gr.themes.Soft(
            primary_hue="amber",
            neutral_hue="stone",
            font=["ZCOOL XiaoWei", "Noto Serif SC", "serif"],
        ),
        css="""
        @import url('https://fonts.googleapis.com/css2?family=ZCOOL+XiaoWei&family=Noto+Serif+SC:wght@400;600;700&display=swap');

        :root {
            --sw-bg-1: #f6efe1;
            --sw-bg-2: #e8d9b8;
            --sw-ink: #2b2118;
            --sw-card: rgba(255, 250, 238, 0.88);
            --sw-border: rgba(84, 61, 38, 0.16);
            --sw-accent: #a45a1b;
            --sw-accent-soft: #f6d4a6;
            --sw-shadow: 0 10px 28px rgba(48, 33, 18, 0.12);
        }

        .gradio-container {
            background:
                radial-gradient(circle at 8% 6%, rgba(255, 224, 170, 0.5), transparent 36%),
                radial-gradient(circle at 92% 88%, rgba(178, 130, 80, 0.2), transparent 40%),
                linear-gradient(145deg, var(--sw-bg-1) 0%, var(--sw-bg-2) 100%);
            color: var(--sw-ink);
        }

        .gradio-container * {
            font-family: "Noto Serif SC", serif;
        }

        #story-title h1 {
            font-family: "ZCOOL XiaoWei", serif;
            letter-spacing: 1px;
            margin-bottom: 6px;
        }

        #story-subtitle {
            color: rgba(43, 33, 24, 0.82);
            margin-bottom: 8px;
        }

        #main-tabs {
            background: rgba(255, 252, 245, 0.52);
            border: 1px solid var(--sw-border);
            border-radius: 16px;
            padding: 8px;
            box-shadow: var(--sw-shadow);
            backdrop-filter: blur(3px);
        }

        #main-tabs .tab-nav {
            gap: 8px;
            border: none;
            padding: 4px;
        }

        #main-tabs .tab-nav button {
            border-radius: 12px;
            border: 1px solid var(--sw-border);
            background: rgba(255, 248, 233, 0.72);
            color: var(--sw-ink);
        }

        #main-tabs .tab-nav button.selected {
            background: linear-gradient(135deg, #d08239 0%, #b56b2a 100%);
            color: #fffaf0;
            border-color: rgba(129, 70, 22, 0.7);
        }

        #left-pane, #right-pane {
            background: var(--sw-card);
            border: 1px solid var(--sw-border);
            border-radius: 16px;
            padding: 14px;
            box-shadow: var(--sw-shadow);
            animation: riseIn 320ms ease-out;
        }

        #story-chat {
            border-radius: 14px;
            border: 1px solid rgba(120, 82, 50, 0.2);
            overflow: hidden;
        }

        #story-chat .wrap {
            background: linear-gradient(180deg, #fff8ea 0%, #fff2d9 100%);
        }

        #dialogue-bar {
            background: linear-gradient(135deg, #ffe8bf 0%, #ffd89a 100%);
            border: 1px solid rgba(160, 92, 28, 0.35);
            border-radius: 10px;
            padding: 7px 12px;
            font-size: 0.92em;
            margin-bottom: 6px;
            color: #4d2d15;
        }

        #input-box textarea {
            border-radius: 10px;
            border: 1px solid rgba(120, 82, 50, 0.24);
            background: #fffdf7;
        }

        #send-btn, #restart-btn {
            border-radius: 10px;
            font-weight: 700;
            letter-spacing: 0.3px;
        }

        #suggestions-row .gr-button {
            border-radius: 999px;
            border: 1px solid rgba(133, 85, 39, 0.28);
            background: rgba(255, 244, 219, 0.85);
        }

        .panel-card {
            border-radius: 12px;
            border: 1px solid var(--sw-border);
            background: rgba(255, 249, 238, 0.92);
            padding: 8px 10px;
            margin-bottom: 8px;
            box-shadow: 0 5px 14px rgba(69, 44, 22, 0.08);
        }

        #ending-panel {
            border: 1px solid rgba(173, 99, 25, 0.38);
            background: linear-gradient(150deg, #fff6e4 0%, #ffeac7 100%);
        }

        #log-box textarea {
            background: #2e2218;
            color: #fde7cb;
            border-radius: 10px;
            border: 1px solid rgba(255, 209, 145, 0.24);
            font-family: "Noto Serif SC", serif;
        }

        @keyframes riseIn {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @media (max-width: 900px) {
            #left-pane, #right-pane {
                padding: 10px;
            }
            #story-chat {
                min-height: 360px;
            }
        }
        """
    ) as demo:

        game_state = gr.State(GameState())
        log_data   = gr.State([])

        gr.Markdown("# 🐾 寻家记：小饼的城市冒险", elem_id="story-title")
        gr.Markdown("*你是流浪小狗小饼。在城市的角落里，也许能找到一个属于你的家。*", elem_id="story-subtitle")
        gr.Markdown("---")

        with gr.Tabs(elem_id="main-tabs"):

            # =================== 游戏主页 ===================
            with gr.Tab("🎮 游戏"):
                with gr.Row():

                    # ===== 左栏 =====
                    with gr.Column(scale=3, elem_id="left-pane"):
                        chatbot = gr.Chatbot(
                            label="冒险故事",
                            height=460,
                            elem_id="story-chat",
                        )

                        dialogue_status = gr.Markdown(
                            value="",
                            elem_id="dialogue-bar",
                            visible=False,
                        )
                        exit_btn = gr.Button(
                            "🚪 结束对话",
                            variant="secondary",
                            size="sm",
                            visible=False,
                        )

                        with gr.Row():
                            input_box = gr.Textbox(
                                placeholder="告诉小饼做什么，例如：走向松鼠 / 闻闻草丛 / 把松果给它",
                                label="",
                                lines=1,
                                scale=5,
                                show_label=False,
                                elem_id="input-box",
                            )
                            send_btn = gr.Button("发送 →", variant="primary", scale=1, elem_id="send-btn")

                        with gr.Row(elem_id="suggestions-row"):
                            btn0 = gr.Button("", variant="secondary", visible=False, size="sm")
                            btn1 = gr.Button("", variant="secondary", visible=False, size="sm")
                            btn2 = gr.Button("", variant="secondary", visible=False, size="sm")
                            btn3 = gr.Button("", variant="secondary", visible=False, size="sm")

                    # ===== 右栏 =====
                    with gr.Column(scale=2, elem_id="right-pane"):
                        scene_panel    = gr.Markdown("点击「开始游戏」", elem_classes=["panel-card"])
                        affinity_panel = gr.Markdown("", elem_classes=["panel-card"])
                        quest_panel    = gr.Markdown("", elem_classes=["panel-card"])
                        ending_panel   = gr.Markdown("", elem_classes=["panel-card"], elem_id="ending-panel")
                        restart_btn    = gr.Button("🔄 重新开始", variant="secondary", elem_id="restart-btn")

            # =================== 日志页 ===================
            with gr.Tab("📋 日志"):
                gr.Markdown("## NLU / 状态日志")
                gr.Markdown("*记录每回合意图识别结果、状态变化，供评估和调试使用。*")
                log_display = gr.Textbox(
                    label="",
                    value="暂无日志。",
                    lines=30,
                    max_lines=60,
                    interactive=False,
                    elem_id="log-box",
                )

        # ===== 统一输出结构 =====
        core_outputs = [
            chatbot, game_state, log_data,
            scene_panel, affinity_panel, quest_panel,
            dialogue_status, exit_btn,
            ending_panel,
        ]
        submit_outputs = core_outputs + [log_display, btn0, btn1, btn2, btn3]
        start_outputs  = core_outputs + [btn0, btn1, btn2, btn3]

        # ===== 事件绑定 =====
        restart_btn.click(fn=on_start, outputs=start_outputs)

        send_btn.click(
            fn=on_submit,
            inputs=[input_box, chatbot, game_state, log_data],
            outputs=[input_box] + submit_outputs,
        )
        input_box.submit(
            fn=on_submit,
            inputs=[input_box, chatbot, game_state, log_data],
            outputs=[input_box] + submit_outputs,
        )

        for btn in [btn0, btn1, btn2, btn3]:
            btn.click(
                fn=on_suggestion,
                inputs=[btn, chatbot, game_state, log_data],
                outputs=[input_box] + submit_outputs,
            )

        exit_btn.click(
            fn=on_exit_dialogue,
            inputs=[chatbot, game_state, log_data],
            outputs=[input_box] + submit_outputs,
        )

        demo.load(fn=on_start, outputs=start_outputs)

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
