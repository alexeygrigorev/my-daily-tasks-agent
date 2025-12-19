import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from mdt_agent.agent import TodoAgentConfig, TodoAgentRunner
from tests.mock_tools import MockTodoTools
from tests.judge import assert_criteria


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

@pytest.mark.asyncio
async def test_agent_todo():
    runner = create_agent()
    prompt = "What do I have for today?"

    result = await runner.run_prompt(prompt)

    await assert_criteria(result, criteria=[
        "agent should use tools to get todos due today",
        "'Update the API spreadsheet' is in the output",
    ])

@pytest.mark.asyncio
async def test_agent_no_todo():
    runner = create_agent(todos=[])
    prompt = "What do I have for today?"

    result = await runner.run_prompt(prompt)

    await assert_criteria(result, criteria=[
        "agent should use tools to get todos due today",
        "there's nothing to do for today",
    ])



@pytest.mark.asyncio
async def test_agent_no_todo():
    runner = create_agent(todos=[])
    prompt = "What do I have for today?"

    result = await runner.run_prompt(prompt)

    await assert_criteria(result, criteria=[
        "agent should use tools to get todos due today",
        "there's nothing to do for today",
    ])


@pytest.mark.asyncio
async def test_mark_todo_completed():
    today_date = today()
    tomorrow = today_date + timedelta(days=1)

    todos = [
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
    ]

    runner = create_agent(todos=todos)

    result = await runner.run_prompt("Show my work todos")

    criteria = [
        "agent should use tools to gets todos with tag 'work'",
        "'Plan engineering sync' is in result",
        "'Update the API spreadsheet' is in result"
    ]

    await assert_criteria(result, criteria)

    result2 = await runner.run_prompt(
        "mark 'Plan engineering sync' as completed and show my incomplete todos",
        message_history=result.new_messages()
    )

    criteria2 = [
        f"todo {todos[1]['id']} marked as completed using tools",
        "agent should use tools to get the list of incomplete todos",
        "'Plan engineering sync' in not included in the results"
    ]

    await assert_criteria(result2, criteria2)


@pytest.mark.asyncio
async def test_create_and_update_todo():
    """Test: Create and Update Todo
    
    This test verifies that the agent can create a todo and then update it with tags.
    """
    runner = create_agent(todos=[])

    # Ask the agent to create a todo "Review code"
    result = await runner.run_prompt("Create a todo 'Review code'")

    # Check that the agent uses tools to create the todo and confirms creation
    criteria = [
        "agent should use tools to create a new todo with text 'Review code'",
        "agent confirms the todo was created",
    ]
    await assert_criteria(result, criteria)

    # Ask to add tags "work" and "urgent" to that todo
    result2 = await runner.run_prompt(
        "Add tags 'work' and 'urgent' to the 'Review code' todo",
        message_history=result.new_messages()
    )

    # Check that the agent updates the todo with those tags and confirms the update
    criteria2 = [
        "agent should use tools to update the todo with tags 'work' and 'urgent'",
        "agent confirms the todo was updated with the tags",
    ]
    await assert_criteria(result2, criteria2)


@pytest.mark.asyncio
async def test_filter_todos_by_date():
    """Test: Filter Todos by Date
    
    This test ensures the agent can filter todos by different date ranges.
    """
    today_date = today()
    tomorrow = today_date + timedelta(days=1)
    next_week = today_date + timedelta(days=7)

    todos = [
        {
            "id": "1765065600020",
            "text": "Today's task",
            "dueDate": today_date.isoformat(),
            "tags": ["work"],
            "completed": False,
        },
        {
            "id": "1765065600021",
            "text": "Tomorrow's task",
            "dueDate": tomorrow.isoformat(),
            "tags": ["personal"],
            "completed": False,
        },
        {
            "id": "1765065600022",
            "text": "Next week's task",
            "dueDate": next_week.isoformat(),
            "tags": ["work"],
            "completed": False,
        },
    ]

    runner = create_agent(todos=todos)

    # Ask which tasks are due today
    result = await runner.run_prompt("What tasks are due today?")

    # Check that the agent retrieves todos and filters to today's only
    criteria = [
        "agent should use tools to get todos due today",
        "agent mentions 'Today's task'",
        "agent does not mention 'Tomorrow's task'",
        "agent does not mention 'Next week's task'",
    ]
    await assert_criteria(result, criteria)

    # Ask what tasks are due this week
    result2 = await runner.run_prompt(
        "What tasks are due this week?",
        message_history=result.new_messages()
    )

    # Check that the agent retrieves todos due this week
    criteria2 = [
        "agent should use tools to get todos due this week",
        "agent mentions 'Today's task'",
        "agent mentions 'Tomorrow's task'",
    ]
    await assert_criteria(result2, criteria2)


@pytest.mark.asyncio
async def test_complete_and_review_tasks():
    """Test: Complete and Review Tasks
    
    This test validates the workflow of completing tasks and reviewing what remains.
    """
    today_date = today()

    todos = [
        {
            "id": "1765065600030",
            "text": "Buy groceries",
            "dueDate": today_date.isoformat(),
            "tags": ["personal"],
            "completed": False,
        },
        {
            "id": "1765065600031",
            "text": "Submit report",
            "dueDate": today_date.isoformat(),
            "tags": ["work"],
            "completed": False,
        },
        {
            "id": "1765065600032",
            "text": "Prepare presentation",
            "dueDate": today_date.isoformat(),
            "tags": ["work"],
            "completed": False,
        },
    ]

    runner = create_agent(todos=todos)

    # Ask to show all todos
    result = await runner.run_prompt("Show all todos")

    # Check that the agent retrieves and lists all tasks
    criteria = [
        "agent should use tools to get all todos",
        "agent mentions 'Buy groceries'",
        "agent mentions 'Submit report'",
        "agent mentions 'Prepare presentation'",
    ]
    await assert_criteria(result, criteria)

    # Ask to mark work tasks completed
    result2 = await runner.run_prompt(
        "Mark all work tasks as completed",
        message_history=result.new_messages()
    )

    # Check that the agent marks the correct two tasks completed and confirms it
    criteria2 = [
        "agent should use tools to mark the work tasks as completed",
        "agent confirms that two tasks were marked as completed",
    ]
    await assert_criteria(result2, criteria2)

    # Ask what tasks are still pending
    result3 = await runner.run_prompt(
        "What tasks are still pending?",
        message_history=result2.new_messages()
    )

    # Check that the agent retrieves todos again and mentions only "Buy groceries" as pending
    criteria3 = [
        "agent should use tools to get incomplete todos",
        "agent mentions 'Buy groceries' as pending",
        "agent does not mention 'Submit report'",
        "agent does not mention 'Prepare presentation'",
    ]
    await assert_criteria(result3, criteria3)


@pytest.mark.asyncio
async def test_modify_todo_due_date():
    """Test: Modify Todo Due Date
    
    This test checks that the agent can update a todo's due date.
    """
    today_date = today()

    todos = [
        {
            "id": "1765065600040",
            "text": "Submit report",
            "dueDate": today_date.isoformat(),
            "tags": ["work"],
            "completed": False,
        },
        {
            "id": "1765065600041",
            "text": "Team meeting",
            "dueDate": today_date.isoformat(),
            "tags": ["work", "meeting"],
            "completed": False,
        },
    ]

    runner = create_agent(todos=todos)

    # Ask to show work todos
    result = await runner.run_prompt("Show my work todos")

    # Check that the agent retrieves todos and mentions "Submit report"
    criteria = [
        "agent should use tools to get todos with tag 'work'",
        "agent mentions 'Submit report'",
    ]
    await assert_criteria(result, criteria)

    # Ask to change its due date to tomorrow
    result2 = await runner.run_prompt(
        "Change the due date of 'Submit report' to tomorrow",
        message_history=result.new_messages()
    )

    # Check that the agent updates the todo with a new due date and confirms the change
    criteria2 = [
        "agent should use tools to update the todo's due date",
        "agent confirms the due date was changed to tomorrow",
    ]
    await assert_criteria(result2, criteria2)


@pytest.mark.asyncio
async def test_organize_todos_by_tags():
    """Test: Organize Todos by Tags
    
    This test ensures the agent can filter todos by multiple tags and combinations.
    """
    today_date = today()

    todos = [
        {
            "id": "1765065600050",
            "text": "Fix bug",
            "dueDate": today_date.isoformat(),
            "tags": ["work", "urgent"],
            "completed": False,
        },
        {
            "id": "1765065600051",
            "text": "Doctor appointment",
            "dueDate": today_date.isoformat(),
            "tags": ["personal", "urgent"],
            "completed": False,
        },
        {
            "id": "1765065600052",
            "text": "Gym",
            "dueDate": today_date.isoformat(),
            "tags": ["personal"],
            "completed": False,
        },
        {
            "id": "1765065600053",
            "text": "Team meeting",
            "dueDate": today_date.isoformat(),
            "tags": ["work", "meeting"],
            "completed": False,
        },
        {
            "id": "1765065600054",
            "text": "Code review",
            "dueDate": today_date.isoformat(),
            "tags": ["work"],
            "completed": False,
        },
    ]

    runner = create_agent(todos=todos)

    # Ask to show urgent tasks
    result = await runner.run_prompt("Show me urgent tasks")

    # Check that the agent filters by "urgent" tag
    criteria = [
        "agent should use tools to filter by 'urgent' tag",
        "agent mentions 'Fix bug'",
        "agent mentions 'Doctor appointment'",
        "agent does not mention 'Gym'",
        "agent does not mention 'Team meeting'",
        "agent does not mention 'Code review'",
    ]
    await assert_criteria(result, criteria)

    # Ask to show personal tasks
    result2 = await runner.run_prompt(
        "Show me personal tasks",
        message_history=result.new_messages()
    )

    # Check that the agent filters by "personal" tag
    criteria2 = [
        "agent should use tools to filter by 'personal' tag",
        "agent mentions 'Gym'",
        "agent mentions 'Doctor appointment'",
        "agent does not mention 'Fix bug'",
        "agent does not mention 'Team meeting'",
        "agent does not mention 'Code review'",
    ]
    await assert_criteria(result2, criteria2)

    # Ask which work tasks are meetings
    result3 = await runner.run_prompt(
        "Which work tasks are meetings?",
        message_history=result2.new_messages()
    )

    # Check that the agent filters by "work" and "meeting" tags
    criteria3 = [
        "agent should use tools to filter by both 'work' and 'meeting' tags",
        "agent mentions 'Team meeting'",
        "agent does not mention 'Fix bug'",
        "agent does not mention 'Code review'",
    ]
    await assert_criteria(result3, criteria3)