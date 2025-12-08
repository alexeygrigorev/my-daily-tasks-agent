"""Streamlit interface for chatting with the todo agent."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, List

import streamlit as st
from dotenv import load_dotenv
from pydantic_ai.messages import FunctionToolCallEvent

from mdt_agent.agent import TodoAgentConfig, TodoAgentRunner

load_dotenv()

DEFAULT_BASE_URL = os.getenv("MY_DAILY_TASKS_URL", "http://localhost:3000")
DEFAULT_MODEL = os.getenv("MY_DAILY_TASKS_MODEL", "openai:gpt-4o-mini")

st.set_page_config(page_title="My Daily Tasks Agent", page_icon=":spiral_note_pad:")


class ToolLogCollector:
    """Collect tool call data so it can be rendered in the chat stream."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.records: List[str] = []

    async def _handle_event(self, ctx: Any, event: Any) -> None:
        if hasattr(event, "__aiter__"):
            async for sub_event in event:  # type: ignore[attr-defined]
                await self._handle_event(ctx, sub_event)
            return

        if isinstance(event, FunctionToolCallEvent):
            args_json = json.dumps(event.part.args, sort_keys=True, default=str)
            self.records.append(f"{self.agent_name}: {event.part.tool_name}({args_json})")

    async def __call__(self, ctx: Any, event: Any) -> None:
        await self._handle_event(ctx, event)


def init_session_state() -> None:
    state = st.session_state
    state.setdefault("base_url", DEFAULT_BASE_URL)
    state.setdefault("model_name", DEFAULT_MODEL)
    state.setdefault("base_url_input", state["base_url"])
    state.setdefault("model_input", state["model_name"])
    state.setdefault("agent_messages", [])
    state.setdefault("chat_history", [])


def reset_conversation() -> None:
    st.session_state.agent_messages = []
    st.session_state.chat_history = []


def build_runner() -> TodoAgentRunner:
    config = TodoAgentConfig(
        model=st.session_state.model_name,
    )
    base_url = st.session_state.base_url
    tools = tools.TodoTools(base_url=base_url)
    return TodoAgentRunner(config, todo_tools=tools)


def run_agent(prompt: str, runner: TodoAgentRunner) -> tuple[str, List[str]]:
    logger = ToolLogCollector(runner.agent.name)
    result = asyncio.run(
        runner.agent.run(
            prompt,
            message_history=st.session_state.agent_messages,
            event_stream_handler=logger,
        )
    )
    st.session_state.agent_messages.extend(result.new_messages())
    return str(result.output), logger.records.copy()


def handle_user_prompt(prompt: str, runner: TodoAgentRunner) -> None:
    clean_prompt = prompt.strip()
    if not clean_prompt:
        st.warning("Please enter a prompt before sending.")
        return

    st.session_state.chat_history.append({"role": "user", "content": clean_prompt})

    try:
        response, tool_logs = run_agent(clean_prompt, runner)
    except Exception as exc:  # noqa: BLE001
        error_text = f"Unable to fulfill the request: {exc}"
        st.session_state.chat_history.append({"role": "assistant", "content": error_text})
        st.error(error_text)
        return

    for log in tool_logs:
        st.session_state.chat_history.append({"role": "tool", "content": log})

    st.session_state.chat_history.append({"role": "assistant", "content": response})


def main() -> None:
    init_session_state()

    with st.sidebar:
        st.header("Configuration")
        st.caption("Provide the Todo API endpoint and model to use.")
        st.text_input("API base URL", key="base_url_input")
        st.text_input("Model", key="model_input")
        if st.button("Reset conversation", use_container_width=True):
            reset_conversation()

    new_base_url = st.session_state.base_url_input.strip() or DEFAULT_BASE_URL
    new_model = st.session_state.model_input.strip() or DEFAULT_MODEL
    if new_base_url != st.session_state.base_url or new_model != st.session_state.model_name:
        st.session_state.base_url = new_base_url
        st.session_state.model_name = new_model
        st.session_state.base_url_input = new_base_url
        st.session_state.model_input = new_model
        reset_conversation()

    runner = build_runner()

    st.title("My Daily Tasks Agent")
    st.write("Chat with the todo agent backed by the Todo API.")
    st.caption(f"Connected to: {st.session_state.base_url}")

    if not st.session_state.chat_history:
        st.info("Start the conversation with the input box below.")
    else:
        for message in st.session_state.chat_history:
            role = message.get("role", "assistant")
            display_role = "assistant" if role == "tool" else role
            with st.chat_message(display_role):
                if role == "tool":
                    st.code(message["content"], language="text")
                else:
                    st.markdown(message["content"])

    user_prompt = st.chat_input("Ask about your todos")
    if user_prompt is not None:
        handle_user_prompt(user_prompt, runner)
        st.rerun()


if __name__ == "__main__":
    main()
