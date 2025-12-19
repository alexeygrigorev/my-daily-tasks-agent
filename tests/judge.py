from datetime import datetime 

from pydantic_ai import Agent, AgentRunResult
from pydantic import BaseModel

from tests.utils import get_tool_calls


judge_instructions = f"""
you are an expert judge evaluating the performance of an
AI agent.

Today is {datetime.now().date().isoformat()}.
""".strip()


judge_user_prompt_template = """
Evaluate the agent's performance based on the following criteria:
<CRITERIA>
{criteria}
</CRITERIA>

The agent's final output was:
<AGENT_OUTPUT>
{output}
</AGENT_OUTPUT>

Tool calls:
<TOOL_CALLS>
{tool_calls}
</TOOL_CALLS>
""".strip()


class JudgeCriterion(BaseModel):
    criterion_description: str
    passed: bool
    judgement: str


class JudgeFeedback(BaseModel):
    criteria: list[JudgeCriterion]
    feedback: str


def create_judge():
    judge = Agent(
        name="judge",
        instructions=judge_instructions,
        model="openai:gpt-4o-mini",
        output_type=JudgeFeedback,
    )
    return judge


async def evaluate_agent_performance(
        criteria: list[str],
        result: AgentRunResult,
        output_transformer: callable = None
    ) -> JudgeFeedback:

    tool_calls = get_tool_calls(result)

    output = result.output
    if output_transformer is not None:
        output = output_transformer(output)

    user_prompt = judge_user_prompt_template.format(
        criteria="\n".join(f"- {c}" for c in criteria),
        output=output,
        tool_calls="\n".join(str(c) for c in tool_calls)
    )

    print("Judge Prompt:", user_prompt)

    judge = create_judge()
    judge_result = await judge.run(user_prompt)
    return judge_result.output


async def assert_criteria(result, criteria):
    feedback = await evaluate_agent_performance(criteria, result)

    print(feedback)

    for criterion in feedback.criteria:
        assert criterion.passed, f"Criterion failed: {criterion.criterion_description}, {criterion.judgement}"