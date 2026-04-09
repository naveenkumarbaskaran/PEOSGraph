"""Tests for PEOSGraph."""

import pytest
import asyncio
from peosgraph import (
    PEOSGraph, PlannerNode, ExecutorNode,
    ObserverNode, ObserverDecision, SynthesiserNode,
    GraphState, PEOSConfig,
)
from peosgraph.state import Plan, ToolResult, Checkpoint


# ─── State Tests ────────────────────────────────────────────────

class TestGraphState:
    def test_state_creation(self):
        state = GraphState(user_message="Hello", history=[])
        assert state.user_message == "Hello"
        assert state.plan is None

    def test_planner_context_window(self):
        history = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        config = PEOSConfig(planner_context_turns=3)
        state = GraphState(user_message="latest", history=history, config=config)
        ctx = state.planner_context
        assert len(ctx) == 3

    def test_trace_recording(self):
        state = GraphState(user_message="test")
        state.add_trace("planner", {"intent": "search"})
        state.add_trace("executor", {"tools": 3})
        assert len(state.trace) == 2
        assert state.trace[0]["node"] == "planner"

    def test_checkpoint_roundtrip(self):
        state = GraphState(user_message="hello", history=[{"role": "user", "content": "hi"}])
        state.plan = Plan(intent="search", tools=["tool_a"])
        state.executor_results = [ToolResult(tool_name="tool_a", success=True, data="result")]

        checkpoint = state.to_checkpoint()
        restored = GraphState.from_checkpoint(checkpoint)
        assert restored.user_message == "hello"
        assert restored.plan.intent == "search"


# ─── Planner Tests ──────────────────────────────────────────────

class TestPlannerNode:
    def test_intent_classification(self):
        planner = PlannerNode(
            intent_catalog=["order_summary", "cost_analysis", "search"],
            tool_groups={
                "order_summary": ["get_order", "get_costs"],
                "cost_analysis": ["get_costs", "get_budget"],
                "search": ["search_orders"],
            },
        )
        state = GraphState(user_message="Show me the cost analysis for this order")
        plan = asyncio.run(planner.run(state))
        assert plan.intent == "cost_analysis"
        assert "get_costs" in plan.tools

    def test_unknown_intent(self):
        planner = PlannerNode(intent_catalog=["search"])
        state = GraphState(user_message="Something completely unrelated xyz")
        plan = asyncio.run(planner.run(state))
        assert plan.intent == "unknown"
        assert plan.confidence < 1.0

    def test_param_extraction(self):
        planner = PlannerNode(intent_catalog=["order_summary"])
        state = GraphState(user_message="Show order 4002310")
        plan = asyncio.run(planner.run(state))
        assert plan.params.get("order_id") == "4002310"


# ─── Executor Tests ─────────────────────────────────────────────

class TestExecutorNode:
    def test_successful_execution(self):
        async def mock_tool(**kwargs):
            return {"status": "OK", "data": [1, 2, 3]}

        executor = ExecutorNode(tools={"my_tool": mock_tool})
        state = GraphState(user_message="test")
        state.plan = Plan(intent="test", tools=["my_tool"])

        results = asyncio.run(executor.run(state))
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].data["status"] == "OK"

    def test_tool_not_found(self):
        executor = ExecutorNode(tools={})
        state = GraphState(user_message="test")
        state.plan = Plan(intent="test", tools=["nonexistent"])

        results = asyncio.run(executor.run(state))
        assert results[0].success is False
        assert "not found" in results[0].error

    def test_tool_exception_handled(self):
        async def failing_tool(**kwargs):
            raise ValueError("Something went wrong")

        executor = ExecutorNode(tools={"bad_tool": failing_tool})
        state = GraphState(user_message="test")
        state.plan = Plan(intent="test", tools=["bad_tool"])

        results = asyncio.run(executor.run(state))
        assert results[0].success is False
        assert "Something went wrong" in results[0].error

    def test_result_truncation(self):
        async def big_tool(**kwargs):
            return "x" * 100_000

        executor = ExecutorNode(tools={"big": big_tool}, result_cap_bytes=1000)
        state = GraphState(user_message="test")
        state.plan = Plan(intent="test", tools=["big"])

        results = asyncio.run(executor.run(state))
        assert "TRUNCATED" in str(results[0].data)

    def test_no_plan_returns_error(self):
        executor = ExecutorNode(tools={})
        state = GraphState(user_message="test")
        # No plan set
        results = asyncio.run(executor.run(state))
        assert results[0].success is False


# ─── Observer Tests ─────────────────────────────────────────────

class TestObserverNode:
    def test_all_success_returns_done(self):
        observer = ObserverNode()
        state = GraphState(user_message="test")
        state.executor_results = [
            ToolResult(tool_name="a", success=True, data="ok"),
            ToolResult(tool_name="b", success=True, data="fine"),
        ]
        decision = asyncio.run(observer.evaluate(state))
        assert decision == ObserverDecision.DONE

    def test_all_failed_returns_fail(self):
        observer = ObserverNode()
        state = GraphState(user_message="test")
        state.executor_results = [
            ToolResult(tool_name="a", success=False, error="timeout"),
        ]
        decision = asyncio.run(observer.evaluate(state))
        assert decision == ObserverDecision.FAIL

    def test_partial_failure_retries(self):
        observer = ObserverNode(retry_on=["tool_error"], max_retries=2)
        state = GraphState(user_message="test")
        state.executor_results = [
            ToolResult(tool_name="a", success=True, data="ok"),
            ToolResult(tool_name="b", success=False, error="transient"),
        ]
        decision = asyncio.run(observer.evaluate(state))
        assert decision == ObserverDecision.RETRY

    def test_max_retries_exhausted(self):
        observer = ObserverNode(retry_on=["tool_error"], max_retries=0)
        state = GraphState(user_message="test")
        state.executor_results = [
            ToolResult(tool_name="a", success=True, data="ok"),
            ToolResult(tool_name="b", success=False, error="err"),
        ]
        # With max_retries=0, should proceed to done (partial success)
        decision = asyncio.run(observer.evaluate(state))
        assert decision == ObserverDecision.DONE

    def test_empty_results_fails(self):
        observer = ObserverNode()
        state = GraphState(user_message="test")
        state.executor_results = []
        decision = asyncio.run(observer.evaluate(state))
        assert decision == ObserverDecision.FAIL


# ─── Synthesiser Tests ──────────────────────────────────────────

class TestSynthesiserNode:
    def test_formats_results(self):
        synth = SynthesiserNode()
        state = GraphState(user_message="test")
        state.plan = Plan(intent="search", tools=["search_orders"])
        state.executor_results = [
            ToolResult(tool_name="search_orders", success=True, data={"count": 5}),
        ]
        output = asyncio.run(synth.run(state))
        assert "search_orders" in output.text
        assert "5" in output.text

    def test_template_applied(self):
        synth = SynthesiserNode(templates={
            "greeting": "Hello! You asked: {query}",
        })
        state = GraphState(user_message="hi")
        state.plan = Plan(intent="greeting", tools=[])
        state.executor_results = [
            ToolResult(tool_name="t", success=True, data={"query": "hello world"}),
        ]
        output = asyncio.run(synth.run(state))
        assert "hello world" in output.text

    def test_quick_replies_length_enforced(self):
        synth = SynthesiserNode(max_quick_reply_length=28)
        state = GraphState(user_message="test")
        state.plan = Plan(intent="order_summary", tools=[])
        state.executor_results = [ToolResult(tool_name="t", success=True, data="ok")]
        output = asyncio.run(synth.run(state))
        for qr in output.quick_replies:
            assert len(qr) <= 28

    def test_error_results_shown(self):
        synth = SynthesiserNode()
        state = GraphState(user_message="test")
        state.plan = Plan(intent="unknown", tools=[])
        state.executor_results = [
            ToolResult(tool_name="api", success=False, error="Connection refused"),
        ]
        output = asyncio.run(synth.run(state))
        assert "Connection refused" in output.text


# ─── Integration Test ───────────────────────────────────────────

class TestPEOSGraphIntegration:
    def test_full_pipeline(self):
        """Full PEOS graph execution end-to-end."""

        async def search_tool(**kwargs):
            return [{"id": "4002310", "status": "In Progress"}]

        async def cost_tool(**kwargs):
            return {"total": 15000, "currency": "EUR"}

        graph = PEOSGraph(
            planner=PlannerNode(
                intent_catalog=["search", "cost_analysis"],
                tool_groups={
                    "search": ["search_tool"],
                    "cost_analysis": ["cost_tool"],
                },
            ),
            executor=ExecutorNode(
                tools={"search_tool": search_tool, "cost_tool": cost_tool},
            ),
            observer=ObserverNode(max_retries=1),
            synthesiser=SynthesiserNode(),
        )

        result = asyncio.run(graph.invoke("Search for maintenance orders"))
        assert result.text  # Has some output
        assert result.metadata["intent"] == "search"
        assert "search_tool" in result.metadata["tools_used"]

    def test_config_propagation(self):
        config = PEOSConfig(
            max_executor_iterations=5,
            max_observer_retries=1,
            planner_context_turns=2,
        )

        graph = PEOSGraph(
            planner=PlannerNode(),
            executor=ExecutorNode(tools={}),
            observer=ObserverNode(),
            synthesiser=SynthesiserNode(),
            config=config,
        )
        assert graph.config.max_executor_iterations == 5
