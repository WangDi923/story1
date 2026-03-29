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

    # 时间进度
    lines.append("---")
    max_turns = scene.get("max_turns", 20)
    turn = min(state.scene_turn_count, max_turns)
    filled = int(turn / max_turns * 20)
    bar = "█" * filled + "░" * (20 - filled)
    lines.append(f"**🕐 时间进度** ({turn}/{max_turns})")
    lines.append(f"`{bar}`")

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
    chat = [(None, OPENING_NARRATION)]
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
    chat.append((player_input, response))

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
        chat.append((None, f"小饼向【{display}】点点头，转身走开了。"))

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
        theme=gr.themes.Soft(),
        css="""
        #dialogue-bar {
            background: #f0f7ff;
            border: 1px solid #a0c4ff;
            border-radius: 8px;
            padding: 6px 12px;
            font-size: 0.9em;
            margin-bottom: 4px;
        }
        """
    ) as demo:

        game_state = gr.State(GameState())
        log_data   = gr.State([])

        gr.Markdown("# 🐾 寻家记：小饼的城市冒险")
        gr.Markdown("*你是流浪小狗小饼。在城市的角落里，也许能找到一个属于你的家。*")
        gr.Markdown("---")

        with gr.Tabs():

            # =================== 游戏主页 ===================
            with gr.Tab("🎮 游戏"):
                with gr.Row():

                    # ===== 左栏 =====
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="冒险故事",
                            height=460,
                            bubble_full_width=False,
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
                            )
                            send_btn = gr.Button("发送 →", variant="primary", scale=1)

                        with gr.Row():
                            btn0 = gr.Button("", variant="secondary", visible=False, size="sm")
                            btn1 = gr.Button("", variant="secondary", visible=False, size="sm")
                            btn2 = gr.Button("", variant="secondary", visible=False, size="sm")
                            btn3 = gr.Button("", variant="secondary", visible=False, size="sm")

                    # ===== 右栏 =====
                    with gr.Column(scale=2):
                        scene_panel    = gr.Markdown("点击「开始游戏」")
                        affinity_panel = gr.Markdown("")
                        quest_panel    = gr.Markdown("")
                        ending_panel   = gr.Markdown("")
                        restart_btn    = gr.Button("🔄 重新开始", variant="secondary")

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
                    show_copy_button=True,
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
