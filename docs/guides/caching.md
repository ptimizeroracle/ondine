# Caching and Rate Limiting

Ondine provides two response caching backends — disk and Redis — that eliminate redundant API calls by storing LLM responses and returning them instantly on repeated requests. Combined with rate limiting, these tools give you direct control over cost and throughput.

## How Response Caching Works

When caching is enabled, Ondine passes a cache configuration to LiteLLM, which intercepts every outgoing request, computes a hash of the prompt and parameters, and checks the store before hitting the API. A cache hit returns the stored response immediately at zero cost. A cache miss calls the API, then writes the response to the store for future use.

This is distinct from [prefix caching](cost-control.md), which is a provider-side optimization for repeated system prompts. Response caching happens on your infrastructure and eliminates the API call entirely.

---

## Disk Caching

### `with_disk_cache(cache_dir=".cache")`

Stores responses as files on the local filesystem. No external service required.

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["summary"])
    .with_prompt("Summarize in one sentence: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_disk_cache()
    .build()
)

result = pipeline.execute()
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache_dir` | `str` | `".cache"` | Directory to store cache files |

**Custom cache location:**

```python
.with_disk_cache(cache_dir="/tmp/ondine_cache")
```

### When to use disk caching

- **Development and testing** — run the same pipeline repeatedly while tuning prompts or downstream code without incurring API costs after the first pass.
- **Single-machine workloads** — no Redis instance to provision or maintain.
- **Reproducibility** — cache files persist between runs, so results are identical across sessions.
- **Iterative data pipelines** — if your input data rarely changes, disk caching turns reprocessing into a near-instant operation.

### Cost savings example

```python
# First run: 1,000 rows × $0.00015/1K tokens ≈ $0.15
# Second run (same inputs): $0.00 — all hits from cache

pipeline = (
    PipelineBuilder.create()
    .from_csv("products.csv", input_columns=["title"], output_columns=["category"])
    .with_prompt("Classify this product title into one category: {title}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_disk_cache(cache_dir=".cache/product_classification")
    .build()
)

# Run once to populate cache
result = pipeline.execute()
print(f"First run cost: ${result.costs.total_cost:.4f}")

# Run again — cache hits, zero cost
result2 = pipeline.execute()
print(f"Second run cost: ${result2.costs.total_cost:.4f}")  # $0.0000
```

---

## Redis Caching

### `with_redis_cache(redis_url="redis://localhost:6379", ttl=3600)`

Stores responses in Redis. Supports TTL-based expiry and is accessible from multiple processes or machines.

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["review"], output_columns=["sentiment"])
    .with_prompt("Classify sentiment as positive, negative, or neutral: {review}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_redis_cache(redis_url="redis://localhost:6379", ttl=3600)
    .build()
)

result = pipeline.execute()
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `redis_url` | `str` | `"redis://localhost:6379"` | Redis connection URL |
| `ttl` | `int` | `3600` | Cache TTL in seconds (1 hour) |

### When to use Redis caching

- **Multiple workers or processes** — disk cache is per-machine; Redis is shared across all workers.
- **Production pipelines** — entries expire automatically via TTL, preventing stale data.
- **High-throughput workloads** — Redis handles concurrent reads and writes without file locking.
- **Combined with the router** — when using `with_router()` across multiple providers, Redis ensures a cached response is reused regardless of which provider would have handled the request.

### TTL configuration

Set TTL based on how frequently your inputs change:

```python
# Short TTL — data changes daily (e.g., news classification)
.with_redis_cache(ttl=86400)  # 24 hours

# Long TTL — stable reference data (e.g., product taxonomy)
.with_redis_cache(ttl=604800)  # 7 days

# Minimum TTL — for testing or debugging only
.with_redis_cache(ttl=300)  # 5 minutes
```

### Production example with remote Redis

```python
import os
from ondine import PipelineBuilder

REDIS_URL = os.getenv("REDIS_URL", "redis://redis-host:6379")

pipeline = (
    PipelineBuilder.create()
    .from_csv("support_tickets.csv",
              input_columns=["ticket_text"],
              output_columns=["category", "priority"])
    .with_prompt("Categorize this support ticket: {ticket_text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_redis_cache(redis_url=REDIS_URL, ttl=43200)  # 12 hours
    .build()
)
```

---

## Disk vs Redis: Choosing a Backend

| Consideration | Disk | Redis |
|---------------|------|-------|
| Setup | None | Redis server required |
| Multi-process | No | Yes |
| TTL / expiry | No | Yes |
| Best for | Local development | Production, distributed |
| Persistence | Permanent (manual cleanup) | Configurable via TTL |

**Use disk caching** when you are working locally and want zero-infrastructure caching with permanent persistence.

**Use Redis caching** when you run pipelines in parallel, across multiple machines, or need automatic expiry.

---

## Cache Invalidation

Neither backend invalidates automatically based on prompt changes. If you update your prompt template, the old cached responses will still be returned for identical input values.

**To invalidate disk cache:** delete or rename the cache directory.

```python
import shutil

shutil.rmtree(".cache")  # Clear entire cache
# or
shutil.rmtree(".cache/product_classification")  # Clear specific run
```

**To invalidate Redis cache:** flush the relevant keys or wait for TTL expiry. For a full flush during development:

```python
import redis

r = redis.from_url("redis://localhost:6379")
r.flushdb()  # Clears all keys in the current database
```

To avoid stale responses after a prompt change, use a namespaced cache directory for disk, or a dedicated Redis database per pipeline version.

---

## Rate Limiting

### `with_rate_limit(rpm)`

Throttles outgoing API requests using a token bucket algorithm. Set this below your provider's stated limit to avoid 429 errors.

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Classify: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_rate_limit(50)  # 50 requests per minute
    .build()
)
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `rpm` | `int` | Maximum requests per minute |

Rate limiting is applied per pipeline execution, not globally across processes.

### Typical rate limits by provider tier

| Provider / tier | Actual limit | Recommended `rpm` |
|-----------------|-------------|-------------------|
| OpenAI free | 3 RPM | 2 |
| OpenAI Tier 1 | 500 RPM | 450 |
| Groq free | 30 RPM | 25 |
| Groq paid | 6,000 RPM | 5,000 |
| Anthropic Tier 1 | 50 RPM | 45 |

Set your `rpm` value 10–20% below the actual limit to leave headroom for retries and burst requests from other processes.

### Rate limiting alongside concurrency

`with_rate_limit()` and `with_concurrency()` work together. Concurrency controls how many requests are in flight simultaneously; rate limiting caps how many are dispatched per minute.

```python
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["label"])
    .with_prompt("Label: {text}")
    .with_llm(provider="groq", model="llama-3.3-70b-versatile")
    .with_concurrency(10)     # Up to 10 simultaneous requests
    .with_rate_limit(25)      # But no more than 25 per minute total
    .build()
)
```

---

## Combining Caching and Rate Limiting

All three features compose freely:

```python
from ondine import PipelineBuilder

pipeline = (
    PipelineBuilder.create()
    .from_csv("reviews.csv", input_columns=["review"], output_columns=["sentiment"])
    .with_prompt("Review: {review}\nSentiment:")
    .with_system_prompt("Classify as positive, negative, or neutral. Return only the label.")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_redis_cache(redis_url="redis://localhost:6379", ttl=86400)
    .with_rate_limit(50)
    .with_concurrency(10)
    .build()
)

result = pipeline.execute()
print(f"Total cost: ${result.costs.total_cost:.4f}")
print(f"Total tokens: {result.costs.total_tokens:,}")
```

Cache hits bypass rate limiting entirely since no API call is made. This means the effective throughput for repeated data is unbounded by the rate limit.

---

## Related

- [Cost Control](cost-control.md) — prefix caching, budget limits, and token optimization
- [Routing](routing.md) — load balancing and failover across multiple providers
- [Execution Modes](execution-modes.md) — concurrency and streaming configuration
