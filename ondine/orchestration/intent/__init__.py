"""Intent layer for ondine.

``plan()`` is an L5 front door (see ARCHITECTURE_PROPOSAL §5). It uses one
LLM call to draft a :class:`PipelineSpecifications` from a data sample and
a natural-language goal, then returns a :class:`Plan` for
approval-by-inspection. There is no agent loop and no new execution path:
``Plan.build()`` simply hands the drafted spec to the existing
:class:`ondine.api.pipeline.PipelineBuilder`.
"""

from ondine.orchestration.intent.planner import Plan, plan

__all__ = ["Plan", "plan"]
