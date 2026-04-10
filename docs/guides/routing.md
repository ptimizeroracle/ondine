# Routing

LLM routing lets a single pipeline distribute requests across multiple model deployments — different providers, models, or API keys — with automatic failover, load balancing, and circuit breaking built in. Ondine exposes this through `with_router()`, which configures a [LiteLLM Router](https://docs.litellm.ai/docs/routing) under the hood.

---

## What LLM Routing Solves

A single provider has practical limits: rate limits that throttle throughput, outages that stop processing entirely, and pricing that may not be optimal for your workload. A router abstracts these away:

- **Failover** — if Groq returns errors, requests automatically go to OpenAI instead.
- **Load balancing** — spread 1,000 concurrent requests across three deployments instead of hammering one.
- **Cost optimization** — route to the cheapest provider that meets your latency requirement.
- **Resilience** — a built-in circuit breaker puts failing providers into a temporary cooldown rather than retrying indefinitely.

---

## `with_router()`

### Signature

```python
def with_router(
    model_list: list[dict],
    routing_strategy: str = "simple-shuffle",
    timeout: int = 120,
    num_retries: int = 2,
    redis_url: str | None = None,
    allowed_fails: int = 3,
    cooldown_time: int = 60,
    **router_kwargs,
) -> PipelineBuilder
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_list` | `list[dict]` | required | Deployment configurations (see below) |
| `routing_strategy` | `str` | `"simple-shuffle"` | Strategy for selecting a deployment per request |
| `timeout` | `int` | `120` | Request timeout in seconds |
| `num_retries` | `int` | `2` | Retry attempts using other deployments on failure |
| `redis_url` | `str \| None` | `None` | Redis URL for distributed state |
| `allowed_fails` | `int` | `3` | Failures before a deployment enters cooldown |
| `cooldown_time` | `int` | `60` | Cooldown duration in seconds |
| `**router_kwargs` | | | Any additional [LiteLLM Router parameter](https://docs.litellm.ai/docs/routing) |

---

## The `model_list` Format

Each entry in `model_list` represents one deployment. The `model_name` field is the shared logical name the router uses for load balancing — multiple entries with the same `model_name` are treated as interchangeable replicas.

```python
{
    "model_name": "my-model",          # Logical name (shared across replicas)
    "model_id": "groq-llama",          # Optional: friendly ID for tracking
    "litellm_params": {
        "model": "groq/llama-3.3-70b-versatile",  # LiteLLM model string
        "api_key": "...",
        "rpm": 30,                     # Optional: per-deployment rate limit
    }
}
```

Deployments with different `model_name` values are not load-balanced against each other — they are treated as separate pools. For automatic failover, all deployments must share the same `model_name`.

---

## Basic Usage

### Failover between two providers

The simplest setup: two deployments with the same `model_name`. The router distributes requests between them and automatically fails over if one provider goes down.

```python
import os
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["label"])
    .with_prompt("Classify: {text}")
    .with_router(
        model_list=[
            {
                "model_name": "fast-llm",
                "litellm_params": {
                    "model": "groq/llama-3.3-70b-versatile",
                    "api_key": os.getenv("GROQ_API_KEY"),
                    "rpm": 30,
                },
            },
            {
                "model_name": "fast-llm",  # Same name = automatic failover
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "rpm": 500,
                },
            },
        ]
    )
    .build()
)

result = pipeline.execute()
```

With the default `simple-shuffle` strategy and two deployments, traffic is split roughly 50/50. If Groq starts returning errors, the circuit breaker kicks in after 3 failures, puts Groq into a 60-second cooldown, and the remaining requests all go to OpenAI until Groq recovers.

---

## Routing Strategies

The `routing_strategy` parameter accepts any of the following values, defined in `RouterStrategy`:

| Strategy | Value | Best for | Redis required |
|----------|-------|----------|----------------|
| Simple shuffle | `"simple-shuffle"` | General use, testing | No |
| Latency-based | `"latency-based-routing"` | Latency-sensitive workloads | Yes |
| Usage-based | `"usage-based-routing"` | Balanced utilization | Yes |
| Usage-based v2 | `"usage-based-routing-v2"` | Production multi-deployment | Yes |
| Cost-based | `"cost-based-routing"` | Minimizing API spend | No |
| Least busy | `"least-busy"` | Deployments with different capacities | Yes |
| Weighted pick | `"weighted-pick"` | Explicit traffic splits | No |

You can also import `RouterStrategy` for IDE autocompletion:

```python
from ondine.core.router_strategies import RouterStrategy

.with_router(
    model_list=[...],
    routing_strategy=RouterStrategy.LATENCY_BASED,
)
```

### Simple shuffle (default)

Random selection with equal probability. No external state required.

```python
.with_router(
    model_list=[...],
    routing_strategy="simple-shuffle",
)
```

### Latency-based routing

Routes each request to the deployment with the lowest recorded average latency. Requires Redis to share latency measurements across workers.

```python
.with_router(
    model_list=[...],
    routing_strategy="latency-based-routing",
    redis_url="redis://localhost:6379",
)
```

### Weighted pick

Routes based on explicit weights. Set a `"weight"` key inside `litellm_params` for each deployment. Useful when you want most traffic on a fast free-tier provider with a smaller paid-tier fallback.

```python
.with_router(
    model_list=[
        {
            "model_name": "llm",
            "litellm_params": {
                "model": "groq/llama-3.3-70b-versatile",
                "api_key": os.getenv("GROQ_API_KEY"),
                "weight": 8,   # 80% of traffic
            },
        },
        {
            "model_name": "llm",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "weight": 2,   # 20% of traffic
            },
        },
    ],
    routing_strategy="weighted-pick",
)
```

### Cost-based routing

Routes to the cheapest deployment based on LiteLLM's cost database. Costs must be defined in your `model_list` or in LiteLLM's built-in pricing tables.

```python
.with_router(
    model_list=[
        {
            "model_name": "llm",
            "litellm_params": {
                "model": "groq/llama-3.3-70b-versatile",
                "api_key": os.getenv("GROQ_API_KEY"),
            },
        },
        {
            "model_name": "llm",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
    ],
    routing_strategy="cost-based-routing",
)
```

---

## Circuit Breaker / Resilience

The circuit breaker is enabled by default with sensible values. It prevents a failing provider from continuing to receive requests during an outage.

**Default behavior:**
- After **3 consecutive failures**, the deployment enters cooldown.
- The cooldown lasts **60 seconds**.
- During cooldown, all requests are routed to healthy deployments.
- After cooldown, the deployment is re-admitted and monitored again.

### Tuning the circuit breaker

```python
# More tolerant — allow more failures before cooldown
.with_router(
    model_list=[...],
    allowed_fails=5,
    cooldown_time=120,   # Longer cooldown
)

# Stricter — pull a deployment faster
.with_router(
    model_list=[...],
    allowed_fails=1,
    cooldown_time=30,
)

# Disable circuit breaker entirely (not recommended in production)
.with_router(
    model_list=[...],
    allowed_fails=0,
)
```

---

## Multi-Provider Load Balancing

For high-throughput pipelines, spread load across three or more deployments. Each deployment can have its own per-deployment rate limit (`rpm`) set in `litellm_params`.

```python
import os
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("large_dataset.csv",
              input_columns=["text"],
              output_columns=["category"])
    .with_prompt("Categorize this text: {text}")
    .with_router(
        model_list=[
            {
                "model_name": "classifier",
                "model_id": "groq-primary",
                "litellm_params": {
                    "model": "groq/llama-3.3-70b-versatile",
                    "api_key": os.getenv("GROQ_API_KEY"),
                    "rpm": 25,
                },
            },
            {
                "model_name": "classifier",
                "model_id": "openai-fallback",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "rpm": 400,
                },
            },
            {
                "model_name": "classifier",
                "model_id": "together-secondary",
                "litellm_params": {
                    "model": "together_ai/togethercomputer/llama-2-70b-chat",
                    "api_key": os.getenv("TOGETHER_API_KEY"),
                    "rpm": 60,
                },
            },
        ],
        routing_strategy="usage-based-routing-v2",
        redis_url="redis://localhost:6379",
        num_retries=2,
        timeout=60,
    )
    .with_concurrency(20)
    .build()
)

result = pipeline.execute()
print(f"Total cost: ${result.costs.total_cost:.4f}")
```

---

## Combining Router with Redis Caching

When using the router in production, pair it with `with_redis_cache()` to avoid duplicate API calls. The cache is checked before the router selects a deployment, so a cache hit never touches any provider.

```python
import os
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("support_tickets.csv",
              input_columns=["ticket"],
              output_columns=["category", "priority"])
    .with_prompt("Triage this support ticket: {ticket}")
    .with_system_prompt("Classify into: billing, technical, account, other. Assign priority: high, medium, low.")
    .with_router(
        model_list=[
            {
                "model_name": "triage-model",
                "litellm_params": {
                    "model": "groq/llama-3.3-70b-versatile",
                    "api_key": os.getenv("GROQ_API_KEY"),
                    "rpm": 25,
                },
            },
            {
                "model_name": "triage-model",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "rpm": 400,
                },
            },
        ],
        routing_strategy="simple-shuffle",
        allowed_fails=3,
        cooldown_time=60,
    )
    .with_redis_cache(redis_url="redis://localhost:6379", ttl=86400)
    .with_concurrency(15)
    .build()
)
```

---

## Passing Additional LiteLLM Router Parameters

`with_router()` accepts `**router_kwargs`, which pass through directly to LiteLLM Router. Any parameter documented at [docs.litellm.ai/docs/routing](https://docs.litellm.ai/docs/routing) can be used this way.

```python
.with_router(
    model_list=[...],
    set_verbose=True,            # Enable LiteLLM debug logging
    enable_pre_call_checks=False, # Skip health checks for faster startup
    debug=True,                  # Enable provider tracking
)
```

---

## Deployment Distribution Tracking

Ondine uses an internal `DeploymentTracker` to map LiteLLM's internal deployment hash IDs to the friendly `model_id` values from your `model_list`. This powers the progress display during execution, showing which provider handled each request.

To give deployments readable labels in the progress UI, set `model_id` in each entry:

```python
{
    "model_name": "fast-llm",
    "model_id": "groq-llama",      # Displayed in progress output
    "litellm_params": {
        "model": "groq/llama-3.3-70b-versatile",
        ...
    },
}
```

If `model_id` is omitted, `model_name` is used as the label.

---

## Related

- [Caching](caching.md) — disk and Redis response caching, rate limiting
- [Cost Control](cost-control.md) — budget limits, prefix caching, token optimization
- [Execution Modes](execution-modes.md) — concurrency and streaming configuration
