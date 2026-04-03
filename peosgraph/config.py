"""PEOS configuration."""

from dataclasses import dataclass


@dataclass
class PEOSConfig:
    """Configuration for PEOSGraph execution."""

    # Model settings
    planner_model: str = "gpt-4o-mini"
    synthesiser_model: str = "gpt-4o-mini"
    llm_timeout: int = 25  # seconds

    # Executor settings
    max_executor_iterations: int = 10
    result_cap_bytes: int = 50_000
    tool_timeout: int = 30  # seconds
    parallel_execution: bool = True

    # Observer settings
    max_observer_retries: int = 2

    # Token optimization
    history_window: int = 40
    planner_context_turns: int = 3

    # Output constraints
    max_quick_reply_length: int = 28
    max_output_tokens: int = 2000
