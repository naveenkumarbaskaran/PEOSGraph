"""Observer Node — quality gate with retry logic."""

from __future__ import annotations
from enum import Enum
from typing import Any

from ..state import GraphState


class ObserverDecision(Enum):
    """Observer's decision after evaluating executor results."""

    DONE = "done"        # Results are good — proceed to Synthesiser
    RETRY = "retry"      # Results insufficient — re-run Executor
    FAIL = "fail"        # Unrecoverable error — proceed to error Synthesiser


class ObserverNode:
    """
    Observer Node — quality gate between Executor and Synthesiser.
    
    Evaluates executor results and decides:
    - DONE: data is sufficient, proceed to synthesis
    - RETRY: data is incomplete or has errors, re-run executor
    - FAIL: unrecoverable error, synthesize error message
    
    Can be rule-based (fast, deterministic) or LLM-based (flexible).
    """

    def __init__(
        self,
        retry_on: list[str] | None = None,
        max_retries: int = 2,
        quality_checks: list[str] | None = None,
        use_llm: bool = False,
    ):
        self.retry_on = retry_on or ["tool_error", "empty_result"]
        self.max_retries = max_retries
        self.quality_checks = quality_checks or []
        self.use_llm = use_llm
        self._retry_count = 0

    async def evaluate(self, state: GraphState) -> ObserverDecision:
        """
        Evaluate executor results and decide next action.
        
        Rules (checked in order):
        1. All tools failed → FAIL
        2. Any retryable condition and retries remaining → RETRY
        3. Quality checks pass → DONE
        """
        results = state.executor_results

        if not results:
            return ObserverDecision.FAIL

        # Count successes and failures
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]

        # All failed → FAIL
        if not successes:
            return ObserverDecision.FAIL

        # Check for retryable conditions
        if self._retry_count < self.max_retries:
            for condition in self.retry_on:
                if self._check_condition(condition, results):
                    self._retry_count += 1
                    return ObserverDecision.RETRY

        # Quality checks
        if self.quality_checks:
            for check in self.quality_checks:
                if not self._run_quality_check(check, results):
                    if self._retry_count < self.max_retries:
                        self._retry_count += 1
                        return ObserverDecision.RETRY
                    return ObserverDecision.FAIL

        return ObserverDecision.DONE

    def _check_condition(self, condition: str, results: list) -> bool:
        """Check if a retryable condition exists."""
        if condition == "tool_error":
            return any(not r.success for r in results)
        elif condition == "empty_result":
            return any(r.success and not r.data for r in results)
        elif condition == "insufficient_data":
            return len([r for r in results if r.success and r.data]) < len(results) / 2
        return False

    def _run_quality_check(self, check: str, results: list) -> bool:
        """Run a named quality check."""
        if check == "has_required_fields":
            # All successful results should have data
            return all(r.data for r in results if r.success)
        elif check == "no_error_messages":
            return not any(r.error for r in results if r.success)
        return True

    def reset(self) -> None:
        """Reset retry counter for a new invocation."""
        self._retry_count = 0
