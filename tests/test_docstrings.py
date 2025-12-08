from mdt_agent.tools import TodoTools
from tests.mock_tools import MockTodoTools


def _doc(obj) -> str:
    doc = getattr(obj, "__doc__", None)
    return doc.strip() if doc else ""


def test_mock_tools_docstrings_match():
    pairs = [
        (TodoTools, MockTodoTools),
    ]
    members = [
        "__init__",
        "create_todo",
        "update_todo",
        "mark_completed",
        "get_incomplete_todos",
        "_resolve_due_shortcut",
        "get_todos",
    ]

    for reference, mock in pairs:
        assert _doc(reference) == _doc(mock)

    for name in members:
        ref_member = getattr(TodoTools, name)
        mock_member = getattr(MockTodoTools, name)
        assert _doc(ref_member) == _doc(mock_member)
