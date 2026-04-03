"""Graph state management and checkpointing."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import time
import json


@dataclass
class Plan:
    """Output of the Planner node."""

    intent: str
    tools: list[str]
    params: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class ToolResult:
    """Result from a single tool execution."""

    tool_name: str
    success: bool
    data: Any = None
    error: str | None = None
    elapsed_ms: float = 0.0


@dataclass
class SynthesiserOutput:
    """Output of the Synthesiser node."""

    text: str
    card: dict | None = None
    quick_replies: list[str] = field(default_factory=list)


@dataclass
class Checkpoint:
    """Serializable graph state for resumption."""

    user_message: str
    history: list[dict]
    plan: dict | None
    executor_results: list[dict]
    trace: list[dict]
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps({
            "user_message": self.user_message,
            "history": self.history,
            "plan": self.plan,
            "executor_results": self.executor_results,
            "trace": self.trace,
            "timestamp": self.timestamp,
        })

    @classmethod
    def from_json(cls, data: str) -> "Checkpoint":
        d = json.loads(data)
        return cls(**d)


class GraphState:
    """
    Mutable state that flows through the PEOS graph.
    
    Each node reads and writes to this shared state.
    """

    def __init__(
        self,
        user_message: str,
        history: list[dict] | None = None,
        config: Any = None,
    ):
        self.user_message = user_message
        self.history = history or []
        self.config = config

        # Set by Planner
        self.plan: Optional[Plan] = None

        # Set by Executor
        self.executor_results: list[ToolResult] = []

        # Internal tracking
        self.trace: list[dict] = []
        self.created_at: float = time.time()

    def add_trace(self, node: str, data: dict) -> None:
        """Add a trace entry for debugging."""
        self.trace.append({
            "node": node,
            "timestamp": time.time() - self.created_at,
            **data,
        })

    @property
    def trimmed_history(self) -> list[dict]:
        """Return history trimmed to config window size."""
        if self.config and hasattr(self.config, "history_window"):
            return self.history[-self.config.history_window:]
        return self.history[-40:]

    @property
    def planner_context(self) -> list[dict]:
        """Return only last N turns for planner (token optimization)."""
        if self.config and hasattr(self.config, "planner_context_turns"):
            n = self.config.planner_context_turns
        else:
            n = 3
        # Only user messages for planner context
        user_msgs = [m for m in self.history if m.get("role") == "user"]
        return user_msgs[-n:]

    def to_checkpoint(self) -> Checkpoint:
        """Serialize state to checkpoint."""
        return Checkpoint(
            user_message=self.user_message,
            history=self.history,
            plan={"intent": self.plan.intent, "tools": self.plan.tools, "params": self.plan.params} if self.plan else None,
            executor_results=[
                {"tool_name": r.tool_name, "success": r.success, "data": str(r.data)[:1000], "error": r.error}
                for r in self.executor_results
            ],
            trace=self.trace,
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: Checkpoint) -> "GraphState":
        """Restore state from checkpoint."""
        state = cls(
            user_message=checkpoint.user_message,
            history=checkpoint.history,
        )
        if checkpoint.plan:
            state.plan = Plan(
                intent=checkpoint.plan["intent"],
                tools=checkpoint.plan["tools"],
                params=checkpoint.plan.get("params", {}),
            )
        state.trace = checkpoint.trace
        return state
