"""Executor Node — tool execution with dynamic binding."""

from __future__ import annotations
from typing import Any, Callable
import time
import asyncio

from ..state import GraphState, ToolResult


class ExecutorNode:
    """
    Executor Node — runs tools selected by the Planner.
    
    Features:
    - Dynamic tool binding (only selected tools are invoked)
    - Parallel execution of independent tools
    - Result truncation (50KB cap)
    - Per-tool timeout
    - Iteration limit (max 10 loops)
    """

    def __init__(
        self,
        tools: list[Callable] | dict[str, Callable] | None = None,
        max_iterations: int = 10,
        result_cap_bytes: int = 50_000,
        timeout: int = 30,
        parallel: bool = True,
    ):
        self.max_iterations = max_iterations
        self.result_cap_bytes = result_cap_bytes
        self.timeout = timeout
        self.parallel = parallel

        # Tool registry
        if isinstance(tools, dict):
            self._tools = tools
        elif isinstance(tools, list):
            self._tools = {
                getattr(t, "__name__", f"tool_{i}"): t
                for i, t in enumerate(tools)
            }
        else:
            self._tools = {}

    def register_tool(self, name: str, func: Callable) -> None:
        """Register a tool function."""
        self._tools[name] = func

    async def run(self, state: GraphState) -> list[ToolResult]:
        """Execute tools from the plan."""
        if not state.plan:
            return [ToolResult(tool_name="none", success=False, error="No plan available")]

        selected_tools = state.plan.tools
        params = state.plan.params
        results = []

        if self.parallel and len(selected_tools) > 1:
            results = await self._run_parallel(selected_tools, params)
        else:
            results = await self._run_sequential(selected_tools, params)

        # Truncate results if too large
        results = self._truncate_results(results)

        return results

    async def _run_sequential(self, tool_names: list[str], params: dict) -> list[ToolResult]:
        """Run tools one by one."""
        results = []
        for name in tool_names[:self.max_iterations]:
            result = await self._execute_tool(name, params)
            results.append(result)
        return results

    async def _run_parallel(self, tool_names: list[str], params: dict) -> list[ToolResult]:
        """Run tools concurrently."""
        tasks = [
            self._execute_tool(name, params)
            for name in tool_names[:self.max_iterations]
        ]
        return await asyncio.gather(*tasks)

    async def _execute_tool(self, name: str, params: dict) -> ToolResult:
        """Execute a single tool with error handling and timeout."""
        if name not in self._tools:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Tool '{name}' not found in registry",
            )

        func = self._tools[name]
        start = time.time()

        try:
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(**params),
                    timeout=self.timeout,
                )
            else:
                result = func(**params)

            elapsed = (time.time() - start) * 1000

            return ToolResult(
                tool_name=name,
                success=True,
                data=result,
                elapsed_ms=elapsed,
            )

        except asyncio.TimeoutError:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Timeout after {self.timeout}s",
                elapsed_ms=self.timeout * 1000,
            )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return ToolResult(
                tool_name=name,
                success=False,
                error=str(e),
                elapsed_ms=elapsed,
            )

    def _truncate_results(self, results: list[ToolResult]) -> list[ToolResult]:
        """Truncate results that exceed the byte cap."""
        for result in results:
            if result.data:
                data_str = str(result.data)
                if len(data_str.encode()) > self.result_cap_bytes:
                    truncated = data_str[:self.result_cap_bytes].encode()[:self.result_cap_bytes].decode(errors="ignore")
                    result.data = truncated + f"\n... [TRUNCATED at {self.result_cap_bytes} bytes]"
        return results

    @property
    def available_tools(self) -> list[str]:
        """List registered tool names."""
        return list(self._tools.keys())
