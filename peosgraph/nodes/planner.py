"""Planner Node — intent classification and tool selection."""

from __future__ import annotations
from typing import Any

from ..state import GraphState, Plan


class PlannerNode:
    """
    Planner Node — first stage of the PEOS pipeline.
    
    Responsibilities:
    1. Classify user intent from a limited context window
    2. Select relevant tools (dynamic binding)
    3. Extract parameters from the user query
    
    Token optimization: Only sees last 3 user turns (not full history).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        intent_catalog: list[str] | None = None,
        tool_groups: dict[str, list[str]] | None = None,
        context_window: int = 3,
    ):
        self.model = model
        self.intent_catalog = intent_catalog or []
        self.tool_groups = tool_groups or {}
        self.context_window = context_window

    async def run(self, state: GraphState) -> Plan:
        """
        Classify intent and produce an execution plan.
        
        In production, this calls an LLM with a focused prompt.
        Here we provide the planning logic framework.
        """
        message = state.user_message
        context = state.planner_context

        # Intent classification (rule-based fallback)
        intent = self._classify_intent(message)

        # Tool selection based on intent → tool group mapping
        tools = self._select_tools(intent)

        # Parameter extraction
        params = self._extract_params(message, intent)

        return Plan(
            intent=intent,
            tools=tools,
            params=params,
            confidence=0.9 if intent in self.intent_catalog else 0.5,
        )

    def _classify_intent(self, message: str) -> str:
        """Rule-based intent classification (LLM in production)."""
        message_lower = message.lower()

        for intent in self.intent_catalog:
            # Simple keyword matching — LLM does this better
            keywords = intent.replace("_", " ").split()
            if any(kw in message_lower for kw in keywords):
                return intent

        return "unknown"

    def _select_tools(self, intent: str) -> list[str]:
        """Select tools for the classified intent."""
        if intent in self.tool_groups:
            return self.tool_groups[intent]

        # Fallback: return all tools from all groups
        all_tools = []
        for tools in self.tool_groups.values():
            all_tools.extend(tools)
        return list(set(all_tools))

    def _extract_params(self, message: str, intent: str) -> dict[str, Any]:
        """Extract parameters from user message (simplified)."""
        params = {}

        # Extract numbers (could be order IDs, quantities, etc.)
        import re
        numbers = re.findall(r"\b\d{7}\b", message)
        if numbers:
            params["order_id"] = numbers[0]

        # Extract quoted strings
        quoted = re.findall(r'"([^"]*)"', message)
        if quoted:
            params["search_term"] = quoted[0]

        return params
