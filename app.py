"""
StoryWeaver Gradio UI
=====================
修复：
  1. 退出对话按钮正确显示/隐藏（使用 gr.update）
  2. 场景面板合并 NPC 与小动物
  3. 输出列表统一，消除 dlg_text/dlg_visible 混乱
"""
import gradio as gr
from typing import List, Tuple, Any

from game_state import GameState, CLUE_META, ENDING_META, PHASE_DESC
from engine import new_game, process_turn, get_suggestions, OPENING_NARRATION
from nlg import NPC_DISPLAY
import logger as glogger


# =====================================================
# 状态面板渲染
# =====================================================

def render_scene_panel(state: GameState) -> str:
    phase_name, _ = PHASE_DESC.get(state.phase, ("傍晚", ""))
    lines = [f"## 📍 城市公园 · {phase_name}", ""]

    lines.append("**可互动角色**")
    lines.append("【老奶奶】  【气球摊主】  【慢跑者】")
    lines.append("【流浪猫小灰】  【松鼠】  【鸽子群】")
    lines.append("")
    lines.append("**可探索地点**")
    lines.append("【长椅】  【草丛】  【喷泉】  【公园入口】  【废弃角落】")

    lines.append("")
    lines.append("---")
    turn = state.turn_count
    total = 20
    filled = min(turn, total)
    bar = "█" * filled + "░" * (total - filled)
    phase_arrow = "🌅 傍晚"
    if state.phase >= 2:
        phase_arrow += " → 🌆 黄昏"
    if state.phase >= 3:
        phase_arrow += " → 🌃 入夜"
    lines.append(f"**🕐 时间进度**")
    lines.append(f"`{bar}`")
    lines.append(phase_arrow)

    return "\n".join(lines)


def render_clue_panel(state: GameState) -> str:
    lines = ["## 🔍 线索栏", ""]
    if not state.clues_found:
        lines.append("💭 小饼东张西望，还没找到任何有用的线索……")
    else:
        for clue_id in state.clues_found:
            desc = CLUE_META.get(clue_id, clue_id)
            lines.append(f"✅ {desc}")
        if not state.game_over:
            n = len(state.clues_found)
            lines.append("")
            if state.flag_direction_known and n >= 3:
                lines.append("💭 小饼感觉自己已经知道该去哪里了！")
            elif n >= 2:
                lines.append("💭 小饼感觉方向越来越清晰了……")
            else:
                lines.append("💭 小饼觉得线索还不够，公园里一定还藏着什么……")
    return "\n".join(lines)


def render_ending_panel(state: GameState) -> str:
    if not state.game_over or not state.ending:
        return ""
    lines = ["## 🎮 已解锁结局", ""]
    for eid, ename in ENDING_META.items():
        if eid == state.ending:
            lines.append(f"✅ **{ename}**")
        else:
            lines.append(f"⬜ {ename}")
    lines.append("")
    lines.append("*「小饼的故事还有其他可能……」*")
    return "\n".join(lines)


# =====================================================
# 对话状态栏（返回 gr.update 对象）
# =====================================================

def dialogue_updates(state: GameState):
    """返回 (dialogue_status update, exit_btn update)"""
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
        render_clue_panel(state),
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
        "",                              # clear input box
        chat, state, logs,
        render_scene_panel(state),
        render_clue_panel(state),
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
        render_clue_panel(state),
        dlg_status, exit_visible,
        render_ending_panel(state),
        glogger.format_log_for_display(logs),
        *_btn_updates(suggestions),
    )


def _btn_updates(suggestions: List[str]):
    padded = (suggestions + ["", "", "", ""])[:4]
    return [gr.update(value=p, visible=bool(p)) for p in padded]


def _unchanged(chat, state, logs):
    """输入为空时什么都不变，返回同样结构"""
    suggestions = get_suggestions(state)
    dlg_status, exit_visible = dialogue_updates(state)
    return (
        chat, state, logs,
        render_scene_panel(state),
        render_clue_panel(state),
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
        gr.Markdown("*你是柴犬小饼，在城市公园和主人小明走散了。"
                    "靠嗅觉和机智找到回家的路！*")
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

                        # 对话状态栏
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

                        # 输入行
                        with gr.Row():
                            input_box = gr.Textbox(
                                placeholder="告诉小饼做什么，例如：闻闻那张长椅 / 慢慢走向老奶奶",
                                label="",
                                lines=1,
                                scale=5,
                                show_label=False,
                            )
                            send_btn = gr.Button("发送 →", variant="primary", scale=1)

                        # 建议选项
                        with gr.Row():
                            btn0 = gr.Button("", variant="secondary", visible=False, size="sm")
                            btn1 = gr.Button("", variant="secondary", visible=False, size="sm")
                            btn2 = gr.Button("", variant="secondary", visible=False, size="sm")
                            btn3 = gr.Button("", variant="secondary", visible=False, size="sm")

                    # ===== 右栏 =====
                    with gr.Column(scale=2):
                        scene_panel  = gr.Markdown("点击「开始游戏」")
                        clue_panel   = gr.Markdown("")
                        ending_panel = gr.Markdown("")
                        restart_btn  = gr.Button("🔄 重新开始", variant="secondary")

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
        # 所有事件函数输出顺序：
        # ("" input_clear,) chat, state, logs,
        # scene, clues, dlg_status, exit_btn,
        # ending, log_display, btn0~3

        core_outputs = [
            chatbot, game_state, log_data,
            scene_panel, clue_panel,
            dialogue_status, exit_btn,
            ending_panel,
        ]
        submit_outputs = core_outputs + [log_display, btn0, btn1, btn2, btn3]
        start_outputs  = core_outputs + [btn0, btn1, btn2, btn3]  # no log_display update

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
