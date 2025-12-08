import pytest

from datetime import datetime, timedelta
from typing import Dict, List

from mdt_agent.agent import TodoAgentConfig, TodoAgentRunner
from tests.mock_tools import MockTodoTools
from tests.utils import get_tool_calls

def today():
    return datetime.now().replace(hour=17, minute=0, second=0, microsecond=0)


def _generate_default_todos() -> List[Dict[str, object]]:
    """Return canned todos that cover the common test scenarios."""
    today_date = today()
    tomorrow = today_date + timedelta(days=1)

    return [
        {
            "id": "1765065600010",
            "text": "Update the API spreadsheet",
            "dueDate": today.isoformat(),
            "tags": ["work", "api"],
            "completed": False,
        },
        {
            "id": "1765065600011",
            "text": "Plan engineering sync",
            "dueDate": tomorrow.isoformat(),
            "tags": ["work"],
            "completed": False,
        },
        {
            "id": "1765065600012",
            "text": "Pick up groceries",
            "dueDate": None,
            "tags": ["personal", "errand"],
            "completed": False,
        },
    ]


def create_agent(todos=None):
    config = TodoAgentConfig()

    if todos is None:
        todos = _generate_default_todos()

    tools = MockTodoTools(todos=todos)
    runner = TodoAgentRunner(config, todo_tools=tools)
    return runner


@pytest.mark.asyncio
async def test_agent_todo():
    runner = create_agent()

    prompt = "What do I have for today?"

    result = await runner.run_prompt(prompt)
    tool_calls = get_tool_calls(result)
    print(tool_calls)
    print(result.output)

    assert len(tool_calls) > 0
    assert tool_calls[0].name in ["get_todos", "get_incomplete_todos"]

    if tool_calls[0].name == 'get_todos':
        assert tool_calls[0].args == {'due_date': 'today'}


@pytest.mark.asyncio
async def test_agent_two_interactions():
    todos = [
        {
            "id": "1765065600010",
            "text": "Update the API spreadsheet",
            "dueDate": today().isoformat(),
            "tags": ["work", "api"],
            "completed": False,
        }
    ]

    runner = create_agent(todos=todos)

    prompt = "What do I have for today?"

    result = await runner.run_prompt(prompt)
    tool_calls = get_tool_calls(result)
    assert 'api' in result.output.lower() 

    assert tool_calls[0].name in ["get_todos", "get_incomplete_todos"]

    prompt_2 = "I have already updated the API spreadsheet"
    result_2 = await runner.run_prompt(
        prompt_2,
        message_history=result.new_messages()
    )

    tool_calls_2 = get_tool_calls(result_2)
    print(tool_calls_2)
    assert tool_calls_2[0].name == "mark_completed"
    assert tool_calls_2[0].args["todo_id"] == "1765065600010"
