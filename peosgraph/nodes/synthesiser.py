"""Synthesiser Node — response formatting and output schema enforcement."""

from __future__ import annotations
from typing import Any

from ..state import GraphState, SynthesiserOutput


class SynthesiserNode:
    """
    Synthesiser Node — final stage of the PEOS pipeline.
    
    Responsibilities:
    1. Format executor results into user-facing response
    2. Generate quick replies (≤28 chars each)
    3. Build card/chart data for UI rendering
    4. Enforce output schema constraints
    
    In production, uses an LLM call for natural language synthesis.
    For deterministic intents, can use templates instead (0 LLM calls).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        output_schema: dict[str, str] | None = None,
        max_tokens: int = 1000,
        templates: dict[str, str] | None = None,
        max_quick_reply_length: int = 28,
    ):
        self.model = model
        self.output_schema = output_schema or {}
        self.max_tokens = max_tokens
        self.templates = templates or {}
        self.max_quick_reply_length = max_quick_reply_length

    async def run(self, state: GraphState) -> SynthesiserOutput:
        """
        Synthesize final response from executor results.
        
        Strategy:
        1. If template exists for intent → use template (0 LLM calls)
        2. Otherwise → LLM synthesis (1 call)
        """
        intent = state.plan.intent if state.plan else "unknown"
        results = state.executor_results

        # Check for template
        if intent in self.templates:
            text = self._apply_template(intent, results)
        else:
            text = self._format_results(results)

        # Generate quick replies
        quick_replies = self._generate_quick_replies(intent, results)

        # Build card data if applicable
        card = self._build_card(intent, results)

        return SynthesiserOutput(
            text=text,
            card=card,
            quick_replies=quick_replies,
        )

    def _apply_template(self, intent: str, results: list) -> str:
        """Apply a response template."""
        template = self.templates[intent]
        # Simple placeholder replacement
        data = {}
        for r in results:
            if r.success and r.data:
                if isinstance(r.data, dict):
                    data.update(r.data)
                else:
                    data[r.tool_name] = r.data

        try:
            return template.format(**data)
        except (KeyError, IndexError):
            return self._format_results(results)

    def _format_results(self, results: list) -> str:
        """Default formatting when no template available."""
        lines = []
        for r in results:
            if r.success:
                if isinstance(r.data, dict):
                    lines.append(f"**{r.tool_name}:**")
                    for k, v in r.data.items():
                        lines.append(f"  - {k}: {v}")
                else:
                    lines.append(f"**{r.tool_name}:** {r.data}")
            else:
                lines.append(f"⚠️ {r.tool_name}: {r.error}")

        return "\n".join(lines) if lines else "No data available."

    def _generate_quick_replies(self, intent: str, results: list) -> list[str]:
        """Generate contextual quick replies."""
        replies = []

        # Intent-specific quick replies
        qr_map = {
            "order_summary": ["Show costs", "Show operations", "Check TECO"],
            "cost_analysis": ["Show summary", "Compare orders"],
            "search": ["Filter by plant", "Show details"],
        }

        if intent in qr_map:
            replies = qr_map[intent]
        else:
            replies = ["Show more", "New search"]

        # Enforce length limit
        return [qr[:self.max_quick_reply_length] for qr in replies]

    def _build_card(self, intent: str, results: list) -> dict | None:
        """Build UI card data from results."""
        # Only build cards for certain intents
        if intent == "search" and results:
            items = []
            for r in results:
                if r.success and isinstance(r.data, list):
                    for item in r.data[:10]:
                        items.append(item if isinstance(item, dict) else {"value": str(item)})
            if items:
                return {
                    "type": "List",
                    "title": "Search Results",
                    "items": items,
                }
        return None
