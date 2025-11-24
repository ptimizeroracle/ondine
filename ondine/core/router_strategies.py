"""
Router strategy enums for LiteLLM Router.

See: https://docs.litellm.ai/docs/routing
"""

from enum import Enum


class RouterStrategy(str, Enum):
    """
    LiteLLM Router routing strategies.

    Each strategy determines how the Router selects which deployment to use
    for each request.
    """

    # Basic Strategies
    SIMPLE_SHUFFLE = "simple-shuffle"
    """
    Random selection with equal probability.
    
    Best for: General use, testing
    Use when: You want basic load balancing without optimization
    """

    # Performance-Based Strategies
    LATENCY_BASED = "latency-based-routing"
    """
    Routes to deployment with lowest average latency.
    
    Best for: Latency-sensitive applications
    Use when: Response time is critical
    Requires: Redis for distributed latency tracking
    """

    USAGE_BASED = "usage-based-routing"
    """
    Routes to deployment with lowest usage/load.
    
    Best for: Balanced utilization
    Use when: You want to avoid overloading any single deployment
    Requires: Redis for distributed usage tracking
    """

    # Cost Optimization
    COST_BASED = "cost-based-routing"
    """
    Routes to cheapest deployment based on LiteLLM's cost database.
    
    Best for: Cost optimization
    Use when: Minimizing API costs is priority
    Note: Costs must be defined in LiteLLM or model_list
    """

    # Advanced Strategies
    LEAST_BUSY = "least-busy"
    """
    Routes to deployment with fewest active requests.
    
    Best for: Real-time load balancing
    Use when: Deployments have different capacities
    Requires: Redis for distributed state
    """

    WEIGHTED_PICK = "weighted-pick"
    """
    Routes based on weights assigned to each deployment.
    
    Best for: Custom traffic distribution (e.g., 80% Groq, 20% OpenAI)
    Use when: You want explicit control over traffic split
    Note: Set "weight" in each model's litellm_params
    """

    # Fallback
    USAGE_BASED_ROUTING_V2 = "usage-based-routing-v2"
    """
    Improved usage-based routing with better load distribution.
    
    Best for: Production workloads with multiple deployments
    Use when: You need sophisticated load balancing
    Requires: Redis for state
    """


# Quick reference for docstrings
ROUTER_STRATEGY_DOCS = """
Available Router Strategies:
- simple-shuffle: Random selection (default, no Redis needed)
- latency-based-routing: Lowest latency deployment (needs Redis)
- usage-based-routing: Least loaded deployment (needs Redis)
- cost-based-routing: Cheapest deployment
- least-busy: Fewest active requests (needs Redis)
- weighted-pick: Custom traffic distribution (set weight in model_list)

See: https://docs.litellm.ai/docs/routing
"""

