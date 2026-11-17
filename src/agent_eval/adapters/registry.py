from agent_eval.adapters.base import TraceAdapter
from agent_eval.adapters.mini_swe import MiniSweAgentAdapter
from agent_eval.adapters.ours import OursAdapter

_ADAPTERS: dict[str, type[TraceAdapter]] = {
    OursAdapter.name: OursAdapter,
    MiniSweAgentAdapter.name: MiniSweAgentAdapter,
}


def get_adapter(agent_name: str) -> TraceAdapter:
    try:
        return _ADAPTERS[agent_name]()
    except KeyError as exc:
        supported = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"未知 Agent {agent_name!r}；可选值: {supported}") from exc


def supported_agents() -> list[str]:
    return sorted(_ADAPTERS)
