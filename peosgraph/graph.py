"""PEOSGraph — the main graph orchestrator."""

from __future__ import annotations
from typing import Any, Optional
import time

from .state import GraphState, Checkpoint
from .nodes.planner import PlannerNode
from .nodes.executor import ExecutorNode
from .nodes.observer import ObserverNode, ObserverDecision
from .nodes.synthesiser import SynthesiserNode
from .config import PEOSConfig


class PEOSGraph:
    """
    PEOS (Planner-Executor-Observer-Synthesiser) graph orchestrator.
    
    Implements a conditional state machine where:
    - Planner: classifies intent and selects tools (1 LLM call)
    - Executor: runs selected tools in a loop (0 LLM calls)
    - Observer: decides retry/continue/done (1 LLM call or rule-based)
    - Synthesiser: formats final response (1 LLM call)
    
    Total: 2-3 LLM calls per request (vs 5-15 in ReAct loops)
    """

    def __init__(
        self,
        planner: PlannerNode,
        executor: ExecutorNode,
        observer: ObserverNode,
        synthesiser: SynthesiserNode,
        config: PEOSConfig | None = None,
    ):
        self.planner = planner
        self.executor = executor
        self.observer = observer
        self.synthesiser = synthesiser
        self.config = config or PEOSConfig()

        self._state: Optional[GraphState] = None

    async def invoke(self, message: str, history: list[dict] | None = None) -> "GraphResult":
        """
        Run the full PEOS graph for a user message.
        
        Args:
            message: User input text
            history: Optional conversation history
            
        Returns:
            GraphResult with text, metadata, and diagnostics
        """
        start_time = time.time()

        # Initialize state
        self._state = GraphState(
            user_message=message,
            history=history or [],
            config=self.config,
        )

        # ── PLANNER ──────────────────────────────────────────
        plan = await self.planner.run(self._state)
        self._state.plan = plan
        self._state.add_trace("planner", {"intent": plan.intent, "tools": plan.tools})

        # ── EXECUTOR LOOP ────────────────────────────────────
        retries = 0
        while retries <= self.config.max_observer_retries:
            results = await self.executor.run(self._state)
            self._state.executor_results = results
            self._state.add_trace("executor", {
                "results_count": len(results),
                "iteration": retries,
            })

            # ── OBSERVER ─────────────────────────────────────
            decision = await self.observer.evaluate(self._state)
            self._state.add_trace("observer", {"decision": decision.value})

            if decision == ObserverDecision.DONE:
                break
            elif decision == ObserverDecision.RETRY:
                retries += 1
                continue
            elif decision == ObserverDecision.FAIL:
                break
            else:
                break

        # ── SYNTHESISER ──────────────────────────────────────
        output = await self.synthesiser.run(self._state)
        self._state.add_trace("synthesiser", {"output_length": len(output.text)})

        elapsed = time.time() - start_time

        return GraphResult(
            text=output.text,
            card=output.card,
            quick_replies=output.quick_replies,
            metadata={
                "intent": plan.intent,
                "tools_used": plan.tools,
                "retries": retries,
                "elapsed_seconds": round(elapsed, 2),
                "trace": self._state.trace,
            },
        )

    def checkpoint(self) -> Checkpoint:
        """Save current graph state for resumption."""
        if not self._state:
            raise RuntimeError("No active state to checkpoint")
        return self._state.to_checkpoint()

    async def resume(self, checkpoint: Checkpoint) -> "GraphResult":
        """Resume graph execution from a checkpoint."""
        self._state = GraphState.from_checkpoint(checkpoint)
        # Re-run from observer onward
        decision = await self.observer.evaluate(self._state)
        if decision in (ObserverDecision.DONE, ObserverDecision.FAIL):
            output = await self.synthesiser.run(self._state)
            return GraphResult(
                text=output.text,
                card=output.card,
                quick_replies=output.quick_replies,
                metadata={"resumed": True},
            )
        # Otherwise need more executor runs
        return await self.invoke(self._state.user_message, self._state.history)


class GraphResult:
    """Result of a PEOS graph invocation."""

    def __init__(
        self,
        text: str,
        card: dict | None = None,
        quick_replies: list[str] | None = None,
        metadata: dict | None = None,
    ):
        self.text = text
        self.card = card
        self.quick_replies = quick_replies or []
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "card": self.card,
            "quick_replies": self.quick_replies,
            "metadata": self.metadata,
        }
