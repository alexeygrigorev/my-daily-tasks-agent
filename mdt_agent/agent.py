import inspect
import textwrap

from datetime import datetime

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from pydantic_ai import Agent
from pydantic_ai.messages import FunctionToolCallEvent

from mdt_agent import tools


def _default_instructions() -> str:
    current_date = datetime.now().date().isoformat()

    return textwrap.dedent(
        f"""
        You're a helpful todo agent

        format for displaying todos:

        - "<NAME>" due to <DUE_DATE> (tag1, tag2...)

        show all the todos returned by the API
        
        Rules:

        - if due date or tags are not present, don't display them
        - when you create a todo, reply with confirmation
        - today is {current_date}
        """
    ).strip()


def _get_public_instance_methods(instance: object) -> List[Any]:
    """Return non-dunder bound methods for an object."""
    methods: List[Any] = []
    for name, member in inspect.getmembers(instance, predicate=inspect.ismethod):
        if not name.startswith("_"):
            methods.append(member)
    return methods


@dataclass
class TodoAgentConfig:
    model: str = "openai:gpt-4o-mini"
    name: str = "todo"
    instructions: str = field(default_factory=_default_instructions)


class NamedCallback:
    """Stream handler that prints the tool calls triggered by an agent."""

    def __init__(self, agent: Agent):
        self.agent_name = agent.name

    async def _print_function_calls(self, ctx: Any, event: Any) -> None:
        # Detect nested streams
        if hasattr(event, "__aiter__"):
            async for sub_event in event:  # type: ignore[attr-defined]
                await self._print_function_calls(ctx, sub_event)
            return

        if isinstance(event, FunctionToolCallEvent):
            tool_name = event.part.tool_name
            args = event.part.args
            print(f"TOOL CALL ({self.agent_name}): {tool_name}({args})")

    async def __call__(self, ctx: Any, event: Any) -> None:
        await self._print_function_calls(ctx, event)


class TodoAgentRunner:
    """Builds and executes the todo agent without relying on module globals."""

    def __init__(self, config: TodoAgentConfig, todo_tools: tools.TodoTools):
        self._config = config
        self._todo_tools = todo_tools
        self._agent = Agent(
            name=config.name,
            instructions=config.instructions,
            tools=_get_public_instance_methods(self._todo_tools),
            model=config.model,
        )
        self._callback = NamedCallback(self._agent)

    @property
    def agent(self) -> Agent:
        """Expose the configured Agent instance."""
        return self._agent

    @property
    def callback(self) -> NamedCallback:
        """Expose the default callback used for function-call logging."""
        return self._callback

    async def run_prompt(
        self,
        prompt: str,
        *,
        message_history: Optional[Sequence[Any]] = None,
        stream_events: bool = True,
    ):
        """Execute a prompt against the agent and return the raw result."""
        event_handler = self._callback if stream_events else None
        result = await self._agent.run(
            prompt,
            message_history=list(message_history or []),
            event_stream_handler=event_handler,
        )
        return result

    async def interactive_cli(self, *, stop_word: str = "stop") -> None:
        """Run an interactive CLI conversation until the user types stop."""
        messages: List[Any] = []

        while True:
            prompt = input("You: ")
            if prompt.lower().strip() == stop_word:
                break

            result = await self.run_prompt(prompt, message_history=messages)
            print(result.output)
            messages.extend(result.new_messages())


__all__ = ["TodoAgentConfig", "TodoAgentRunner", "NamedCallback"]
