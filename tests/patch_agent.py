
from functools import wraps
import inspect
from typing import Any

# Don't import pydantic_ai types here

# Central in-process collector
_USAGE_RECORDS = {}


def _record_usage_from_result(agent, result):
    try:
        model_name = agent.model.model_name
        model_provider = agent.model.system
        full_model_name = f"{model_provider}:{model_name}"

        usage = result.usage()

        if usage is None:
            return

        if full_model_name not in _USAGE_RECORDS:
            _USAGE_RECORDS[full_model_name] = []
        _USAGE_RECORDS[full_model_name].append(usage)
    except Exception:
        return


def _wrap_run_callable(orig_run):
    """Return an async wrapper which calls orig_run and records usage."""

    @wraps(orig_run)
    async def wrapped(self, *args, **kwargs):
        result = await orig_run(self, *args, **kwargs)
        try:
            _record_usage_from_result(self, result)
        except Exception:
            pass
        return result

    return wrapped


def install_usage_collector() -> bool:
    installed = False
    
    from pydantic_ai import Agent as PAAgent  # type: ignore

    try:
        if not hasattr(PAAgent, "_orig_run_for_tests"):
            orig_run = getattr(PAAgent, "run", None)
            setattr(PAAgent, "_orig_run_for_tests", orig_run)
            if orig_run is not None and inspect.iscoroutinefunction(orig_run):
                PAAgent.run = _wrap_run_callable(orig_run)
            installed = True
    except Exception:
        pass

    return installed


def get_usage_aggregated():
    from pydantic_ai import RunUsage  # type: ignore

    aggregated = {}

    for model_name, usage_list in _USAGE_RECORDS.items():
        total = RunUsage()

        for usage in usage_list:
            total.incr(usage)

        aggregated[model_name] = total

    return aggregated


def print_report_usage() -> Any:
    print('\n=== USAGE REPORT ===')
    from genai_prices import calc_price  # type: ignore

    aggregated = get_usage_aggregated()

    for full_model_name, usage in aggregated.items():
        provider, model_name = full_model_name.split(":", 1)
        cost = calc_price(
            usage,
            model_ref=model_name,
            provider_id=provider
        )
        print(f'Model: {full_model_name}, Cost: {cost.total_price}')

    print('====================\n')