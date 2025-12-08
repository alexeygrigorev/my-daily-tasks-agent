from datetime import datetime, timedelta
import requests
from typing import Optional, List


class TodoTools:
    def __init__(self, base_url: str):
        """
        Initialize a TodoTools client.

        Args:
            base_url: Base URL of the Todo API (e.g., "http://localhost:8000").
        """
        self.base_url = base_url.rstrip("/")

    # ---------------------------------------------------------
    # CREATE
    # ---------------------------------------------------------
    def create_todo(self, title: str, due_date: Optional[str] = None, tags: Optional[List[str]] = None):
        """
        Create a new todo item.

        Args:
            title: Text describing the todo.
            due_date: Optional ISO-8601 date/time string for when the todo is due.
            tags: Optional list of tag strings to categorize the todo.

        Returns:
            The created todo as a dictionary.
        """
        url = f"{self.base_url}/api/todos"
        payload = {
            "text": title,
            "dueDate": due_date,
            "tags": tags or []
        }
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    # ---------------------------------------------------------
    # UPDATE
    # ---------------------------------------------------------
    def update_todo(
        self,
        todo_id: str,
        title: Optional[str] = None,
        due_date: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Update properties of an existing todo.

        Args:
            todo_id: ID of the todo to update.
            title: New title text, if being changed.
            due_date: New due date (ISO-8601 string), if being changed.
            tags: New list of tags, if being changed.

        Returns:
            The updated todo as a dictionary.
        """
        url = f"{self.base_url}/api/todos/{todo_id}"

        payload = {}
        if title is not None:
            payload["text"] = title
        if due_date is not None:
            payload["dueDate"] = due_date
        if tags is not None:
            payload["tags"] = tags

        resp = requests.patch(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    # ---------------------------------------------------------
    # MARK COMPLETED
    # ---------------------------------------------------------
    def mark_completed(self, todo_id: str):
        """
        Mark a todo as completed.

        Args:
            todo_id: ID of the todo to complete.

        Returns:
            The updated todo reflecting completed status.
        """
        url = f"{self.base_url}/api/todos/{todo_id}/toggle"
        resp = requests.post(url)
        resp.raise_for_status()
        todo = resp.json()

        # If it toggled the wrong way, toggle again
        if not todo["completed"]:
            resp = requests.post(url)
            resp.raise_for_status()
            todo = resp.json()

        return todo

    # ---------------------------------------------------------
    # GET INCOMPLETE TODOS
    # ---------------------------------------------------------
    def get_incomplete_todos(self):
        """
        Retrieve all todos that have not been completed.

        Returns:
            A list of incomplete todos.
        """
        url = f"{self.base_url}/api/todos"
        resp = requests.get(url)
        resp.raise_for_status()
        todos = resp.json()
        return [t for t in todos if not t["completed"]]


    def _resolve_due_shortcut(self, shortcut: str) -> Optional[str]:
        """
        Convert natural-language due date shortcuts into an ISO-8601 datetime string.

        Supported shortcuts:
            - "today"
            - "tomorrow"
            - "this_week"
            - "next_week"

        Returns:
            A string in ISO-8601 format, or None if no shortcut matched.
        """
        now = datetime.now()
        today_end = datetime(now.year, now.month, now.day, 23, 59, 59)

        if shortcut == "today":
            return today_end.isoformat()

        if shortcut == "tomorrow":
            d = today_end + timedelta(days=1)
            return d.isoformat()

        if shortcut == "this_week":
            # Monday is 0; Sunday is 6
            days_left = 6 - now.weekday()
            week_end = today_end + timedelta(days=days_left)
            return week_end.isoformat()

        if shortcut == "next_week":
            # end of next week = end of this week + 7 days
            days_left = 6 - now.weekday()
            next_week_end = today_end + timedelta(days=days_left + 7)
            return next_week_end.isoformat()

        return None

    def get_todos(
        self,
        due_date: Optional[str] = None,
        tag: Optional[str] = None,
        completed: Optional[bool] = False
    ):
        """
        Retrieve todos, optionally filtering by due date, tag, and completion status.

        Args:
            due_date:
                - ISO-8601 date/time string
                - OR shortcut string: "today", "tomorrow", "this_week", "next_week"
            tag:
                A tag name to filter by.
            completed:
                Completion filter:
                    False (default) → return only incomplete todos
                    True            → return only completed todos
                    None            → return all todos

        Returns:
            A list of todos matching all provided filters.
        """
        url = f"{self.base_url}/api/todos"
        params = {}

        # Due date filtering (shortcut or literal)
        if due_date:
            shortcut_value = self._resolve_due_shortcut(due_date)
            params["dueBefore"] = shortcut_value or due_date

        # Tag filter
        if tag:
            params["tags"] = tag

        resp = requests.get(url, params=params)
        resp.raise_for_status()
        todos = resp.json()

        # Client-side completion filtering
        if completed is True:
            return [t for t in todos if t["completed"]]
        elif completed is False:
            return [t for t in todos if not t["completed"]]
        else:
            return todos
