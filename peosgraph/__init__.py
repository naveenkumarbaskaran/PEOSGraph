"""PEOSGraph — LangGraph implementation of the PEOS orchestration pattern."""

from .graph import PEOSGraph
from .nodes.planner import PlannerNode
from .nodes.executor import ExecutorNode
from .nodes.observer import ObserverNode, ObserverDecision
from .nodes.synthesiser import SynthesiserNode
from .state import GraphState
from .config import PEOSConfig

__version__ = "0.1.0"
__all__ = [
    "PEOSGraph",
    "PlannerNode",
    "ExecutorNode",
    "ObserverNode",
    "ObserverDecision",
    "SynthesiserNode",
    "GraphState",
    "PEOSConfig",
]
