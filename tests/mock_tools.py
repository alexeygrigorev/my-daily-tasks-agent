from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from mdt_agent.tools import TodoTools


class MockTodoTools:
    def __init__(self, *, todos: Optional[List[Dict[str, object]]] = None):
        self.base_url = "mock://todos"
        self._todos: Dict[str, Dict[str, object]] = {
            todo["id"]: deepcopy(todo) for todo in todos or []
        }
        numeric_ids = [
            int(todo_id) for todo_id in self._todos if todo_id.isdigit()
        ]
        self._next_id = max(numeric_ids, default=1_000_000_000_000) + 1

    def _clone(self, todo: Dict[str, object]) -> Dict[str, object]:
        clone = deepcopy(todo)
        return clone

    def _next_id_value(self) -> str:
        next_id = str(self._next_id)
        self._next_id += 1
        return next_id

    def create_todo(
        self,
        title: str,
        due_date: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        todo_id = self._next_id_value()
        todo = {
            "id": todo_id,
            "text": title,
            "dueDate": due_date,
            "tags": list(tags or []),
            "completed": False,
        }
        self._todos[todo_id] = todo
        return self._clone(todo)

    def update_todo(
        self,
        todo_id: str,
        title: Optional[str] = None,
        due_date: Optional[str] = None,
        tags: Optional[List[str]] = None,
        completed: Optional[bool] = None,
    ):
        todo = self._todos.get(todo_id)
        if todo is None:
            raise ValueError(f"Todo {todo_id} does not exist")

        if title is not None:
            todo["text"] = title
        if due_date is not None:
            todo["dueDate"] = due_date
        if tags is not None:
            todo["tags"] = list(tags)
        if completed is not None:
            todo["completed"] = completed

        return self._clone(todo)

    def mark_completed(self, todo_id: str):
        todo = self._todos.get(todo_id)
        if todo is None:
            raise ValueError(f"Todo {todo_id} does not exist")

        todo["completed"] = True
        return self._clone(todo)

    def get_incomplete_todos(self):
        return [
            self._clone(todo)
            for todo in self._todos.values()
            if not todo["completed"]
        ]

    def _parse_datetime(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _resolve_due_shortcut(self, shortcut: str) -> Optional[str]:
        now = datetime.now()
        today_end = datetime(now.year, now.month, now.day, 23, 59, 59)

        if shortcut == "today":
            return today_end.isoformat()

        if shortcut == "tomorrow":
            d = today_end + timedelta(days=1)
            return d.isoformat()

        if shortcut == "this_week":
            days_left = 6 - now.weekday()
            week_end = today_end + timedelta(days=days_left)
            return week_end.isoformat()

        if shortcut == "next_week":
            days_left = 6 - now.weekday()
            next_week_end = today_end + timedelta(days=days_left + 7)
            return next_week_end.isoformat()

        return None

    def get_todos(
        self,
        due_date: Optional[str] = None,
        tag: Optional[str] = None,
        completed: Optional[bool] = False,
    ):
        todos = list(self._todos.values())

        if due_date:
            target = self._resolve_due_shortcut(due_date) or due_date
            target_dt = self._parse_datetime(target)

            if target_dt is not None:
                todos = [
                    todo
                    for todo in todos
                    if (
                        (todo_dt := self._parse_datetime(todo.get("dueDate")))
                        is not None
                        and todo_dt <= target_dt
                    )
                ]
            else:
                todos = [
                    todo for todo in todos if todo.get("dueDate") == target
                ]

        if tag:
            todos = [
                todo
                for todo in todos
                if tag in (todo.get("tags") or [])
            ]

        if completed is True:
            todos = [todo for todo in todos if todo.get("completed")]
        elif completed is False:
            todos = [todo for todo in todos if not todo.get("completed")]

        return [self._clone(todo) for todo in todos]


def _copy_docstrings():
    def _maybe_copy(src, dest):
        doc = getattr(src, "__doc__", None)
        if doc:
            dest.__doc__ = doc

    _maybe_copy(TodoTools, MockTodoTools)
    _maybe_copy(TodoTools.__init__, MockTodoTools.__init__)

    for name in [
        "create_todo",
        "update_todo",
        "mark_completed",
        "get_incomplete_todos",
        "_resolve_due_shortcut",
        "get_todos",
    ]:
        _maybe_copy(getattr(TodoTools, name), getattr(MockTodoTools, name))


_copy_docstrings()


__all__ = ["MockTodoTools"]
