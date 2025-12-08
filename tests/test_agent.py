import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from mdt_agent.agent import TodoAgentConfig, TodoAgentRunner
from tests.mock_tools import MockTodoTools
from tests.utils import get_tool_calls


def today() -> datetime:
    return datetime.now().replace(hour=17, minute=0, second=0, microsecond=0)


def _generate_default_todos() -> List[Dict[str, object]]:
    today_date = today()
    tomorrow = today_date + timedelta(days=1)

    return [
        {
            "id": "1765065600010",
            "text": "Update the API spreadsheet",
            "dueDate": today_date.isoformat(),
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


def create_agent(todos: Optional[List[Dict[str, object]]] = None) -> TodoAgentRunner:
    config = TodoAgentConfig()
    if todos is None:
        todos = _generate_default_todos()
    tools = MockTodoTools(todos=todos)
    return TodoAgentRunner(config, todo_tools=tools)


async def _run_prompt(runner: TodoAgentRunner, prompt: str, *, history=None):
    result = await runner.run_prompt(prompt, message_history=history)
    return result, get_tool_calls(result)



@pytest.mark.asyncio
async def test_agent_todo():
    runner = create_agent()
    prompt = "What do I have for today?"
    result, tool_calls = await _run_prompt(runner, prompt)

    assert len(tool_calls) > 0
    assert tool_calls[0].name in ["get_todos", "get_incomplete_todos"]

    if tool_calls[0].name == "get_todos":
        assert "due_date" in tool_calls[0].args


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
    result, tool_calls = await _run_prompt(runner, "What do I have for today?")
    assert tool_calls[0].name in ["get_todos", "get_incomplete_todos"]

    prompt_2 = "I have already updated the API spreadsheet"
    result_2, tool_calls_2 = await _run_prompt(
        runner,
        prompt_2,
        history=result.new_messages(),
    )
    assert tool_calls_2[0].name == "mark_completed"
    assert tool_calls_2[0].args["todo_id"] == "1765065600010"


@pytest.mark.asyncio
async def test_create_simple_todo():
    runner = create_agent()
    title = "Schedule dentist appointment"
    _, tool_calls = await _run_prompt(runner, f'Create a todo titled "{title}".')

    call = tool_calls[0]
    assert call.name == "create_todo"
    assert call.args["title"] == title

    todos = runner._todo_tools.get_incomplete_todos()
    assert any(todo["text"] == title for todo in todos)


@pytest.mark.asyncio
async def test_create_todo_with_due_date():
    runner = create_agent()
    title = "Submit compliance report"
    _, tool_calls = await _run_prompt(
        runner,
        f'Create a todo called "{title}" due tomorrow.',
    )
    call = tool_calls[0]
    assert call.name == "create_todo"
    assert call.args["title"] == title
    assert call.args.get("due_date")

    todos = runner._todo_tools.get_incomplete_todos()
    stored = next(todo for todo in todos if todo["text"] == title)
    assert stored["dueDate"] == call.args["due_date"]


@pytest.mark.asyncio
async def test_create_todo_with_tags():
    runner = create_agent()
    title = "Deploy hotfix"
    _, tool_calls = await _run_prompt(
        runner,
        f'Add a todo titled "{title}" tagged work and urgent.',
    )
    call = tool_calls[0]
    assert call.name == "create_todo"
    assert call.args["title"] == title
    assert {tag.lower() for tag in call.args["tags"]} == {"work", "urgent"}

    todos = runner._todo_tools.get_incomplete_todos()
    stored = next(todo for todo in todos if todo["text"] == title)
    assert set(stored["tags"]) == {"work", "urgent"}


@pytest.mark.asyncio
async def test_get_todos_empty():
    future = (today() + timedelta(days=3)).isoformat()
    todos = [
        {
            "id": "2001",
            "text": "Future item",
            "dueDate": future,
            "tags": ["work"],
            "completed": False,
        }
    ]
    runner = create_agent(todos=todos)
    _, tool_calls = await _run_prompt(
        runner,
        "Only show the todos that are due today.",
    )
    call = tool_calls[0]
    assert call.name == "get_todos"
    assert call.args.get("due_date")


@pytest.mark.asyncio
async def test_get_todos_by_tag():
    runner = create_agent()
    _, tool_calls = await _run_prompt(
        runner,
        "Filter my todo list to tasks tagged with work.",
    )
    call = tool_calls[0]
    assert call.name == "get_todos"
    assert call.args.get("tag") == "work"


@pytest.mark.asyncio
async def test_get_todos_this_week():
    runner = create_agent()
    _, tool_calls = await _run_prompt(
        runner,
        "Show me what is due this week.",
    )
    call = tool_calls[0]
    assert call.name == "get_todos"
    assert call.args.get("due_date") == "this_week"


@pytest.mark.asyncio
async def test_get_incomplete_todos():
    runner = create_agent()
    _, tool_calls = await _run_prompt(
        runner,
        "List only my incomplete todos.",
    )
    call = tool_calls[0]
    assert call.name in ["get_incomplete_todos", "get_todos"]
    if call.name == "get_todos":
        assert call.args.get("completed") is False


@pytest.mark.asyncio
async def test_update_todo_title():
    runner = create_agent()
    new_title = "Update the API spreadsheet ASAP"
    _, tool_calls = await _run_prompt(
        runner,
        f'Rename todo 1765065600010 to "{new_title}".',
    )
    call = tool_calls[0]
    assert call.name == "update_todo"
    assert call.args["todo_id"] == "1765065600010"
    assert call.args["title"] == new_title

    todos = runner._todo_tools.get_incomplete_todos()
    stored = next(todo for todo in todos if todo["id"] == "1765065600010")
    assert stored["text"] == new_title


@pytest.mark.asyncio
async def test_update_todo_due_date():
    runner = create_agent()
    _, tool_calls = await _run_prompt(
        runner,
        "Set the due date for todo 1765065600011 to next week.",
    )
    call = tool_calls[0]
    assert call.name == "update_todo"
    assert call.args["todo_id"] == "1765065600011"
    assert call.args.get("due_date")


@pytest.mark.asyncio
async def test_update_todo_tags():
    runner = create_agent()
    _, tool_calls = await _run_prompt(
        runner,
        "Update todo 1765065600012 to have the tags chores and errand.",
    )
    call = tool_calls[0]
    assert call.name == "update_todo"
    assert call.args["todo_id"] == "1765065600012"
    assert set(call.args["tags"]) == {"chores", "errand"}


@pytest.mark.asyncio
async def test_mark_completed():
    runner = create_agent()
    todo_id = "1765065600010"
    _, tool_calls = await _run_prompt(
        runner,
        f"Mark todo {todo_id} as done.",
    )
    call = tool_calls[0]
    assert call.name == "mark_completed"
    assert call.args["todo_id"] == todo_id

    completed = runner._todo_tools.get_todos(completed=True)
    assert any(todo["id"] == todo_id for todo in completed)


@pytest.mark.asyncio
async def test_mark_completed_uncompleted():
    runner = create_agent()
    todo_id = "1765065600010"
    first_result, _ = await _run_prompt(
        runner,
        f"Mark todo {todo_id} as complete.",
    )

    _, tool_calls = await _run_prompt(
        runner,
        f"Actually set todo {todo_id} back to incomplete.",
        history=first_result.new_messages(),
    )
    update_calls = [
        call for call in tool_calls if call.name == "update_todo"
    ]
    assert any(
        call.args.get("todo_id") == todo_id
        and call.args.get("completed") is False
        for call in update_calls
    )

    incomplete = runner._todo_tools.get_todos(completed=False)
    assert any(todo["id"] == todo_id for todo in incomplete)


@pytest.mark.asyncio
async def test_create_multiple_todos():
    runner = create_agent()
    prompt = (
        "Create two todos: "
        "1) Refill printer paper. "
        "2) Order new office chairs."
    )
    _, tool_calls = await _run_prompt(runner, prompt)

    assert len(tool_calls) >= 2
    titles = [call.args["title"] for call in tool_calls if call.name == "create_todo"]
    assert "Refill printer paper" in titles
    assert "Order new office chairs" in titles


@pytest.mark.asyncio
async def test_filter_by_completion_status():
    todos = _generate_default_todos()
    todos[0]["completed"] = True
    runner = create_agent(todos=todos)
    _, tool_calls = await _run_prompt(
        runner,
        "Show my work todos that are already completed.",
    )
    call = tool_calls[0]
    assert call.name == "get_todos"
    assert call.args.get("completed") is True
    assert call.args.get("tag") == "work"


@pytest.mark.asyncio
async def test_workflow_create_and_complete():
    runner = create_agent()
    title = "Follow up with client"
    result1, calls1 = await _run_prompt(
        runner,
        f'Create a todo titled "{title}".',
    )
    assert calls1[0].name == "create_todo"

    result2, calls2 = await _run_prompt(
        runner,
        "What are all my todos?",
        history=result1.new_messages(),
    )
    assert calls2[0].name in ["get_todos", "get_incomplete_todos"]

    new_id = str(runner._todo_tools._next_id - 1)
    _, calls3 = await _run_prompt(
        runner,
        f'Mark "{title}" as done.',
        history=result1.new_messages() + result2.new_messages(),
    )
    assert calls3[0].name == "mark_completed"
    assert calls3[0].args["todo_id"] == new_id

    completed = runner._todo_tools.get_todos(completed=True)
    assert any(todo["id"] == new_id for todo in completed)


@pytest.mark.asyncio
async def test_no_todos_today():
    tomorrow = (today() + timedelta(days=1)).isoformat()
    todos = [
        {
            "id": "2002",
            "text": "Future planning",
            "dueDate": tomorrow,
            "tags": ["planning"],
            "completed": False,
        }
    ]
    runner = create_agent(todos=todos)
    _, tool_calls = await _run_prompt(
        runner,
        "Do I have anything due today? Only include tasks due today.",
    )
    call = tool_calls[0]
    assert call.name == "get_todos"
    assert call.args.get("due_date")

