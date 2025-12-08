#!/usr/bin/env python
# coding: utf-8

from pydantic_ai import Agent

from mdt_agent import tools
from mdt_agent.utils import get_instance_methods, NamedCallback


base_url = os.getenv('MY_DAILY_TASKS_URL')
todo_tools = tools.TodoTools(base_url=base_url)


agent_todo_tools = get_instance_methods(todo_tools)



instructions = """
You're a helpful todo agent

format for displaying todos:

- "<NAME>" due to <DUE_DATE> (tag1, tag2...)

show all the todos returned by the API

if due date or tags are not present, don't display them 
""".strip()



def create_agent_with_tools(backend_url):
    todo_tools = tools.TodoTools(base_url=backend_url)

    agent_todo_tools = get_instance_methods(todo_tools)
    
    return Agent(
        name='todo',
        instructions=instructions,
        tools=agent_todo_tools,
        model='openai:gpt-4o-mini'
    )


async def run_agent(agent):
    todo_agent_callback = NamedCallback(agent)
    result = await agent.run(
        prompt,
        event_stream_handler=todo_agent_callback
    )
    return result


# messages = []

# while True:
#     prompt = input('You: ')
#     if prompt.lower().strip() == 'stop':
#         break

#     result = await todo_agent.run(
#         prompt,
#         message_history=messages,
#         event_stream_handler=todo_agent_callback
#     )

#     print(result.output)
#     messages.extend(result.new_messages())


