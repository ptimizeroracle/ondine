<!-- 2405d050-cb71-448f-83e8-efc9370910fb 558b4768-7690-40ed-8b49-f55bfff10ced -->
# Rate Limiting Migration to limits Library

## Executive Summary

Migrate from custom `RateLimiter` (94 lines) to `limits` library for distributed rate limiting, multiple algorithm support, and production-grade reliability.

---

## Part 1: Problem Definition & Analysis (Strategist + Expert)

### Current State Analysis

**Current Implementation** (`ondine/utils/rate_limiter.py`):

- Custom token bucket algorithm (94 lines)
- Thread-safe with `threading.Lock`
- Single-process only (no distributed support)
- Hardcoded polling-based waiting (0.1s sleep)
- No persistence across restarts
- Limited to requests-per-minute (RPM) metric

**Usage Pattern**:

```python
# In ondine/api/pipeline.py (lines 527-536)
rate_limiter = RateLimiter(
    specs.processing.rate_limit_rpm,
    burst_size=min(20, specs.processing.concurrency)
) if specs.processing.rate_limit_rpm else None

# In ondine/stages/llm_invocation_stage.py (line 463)
if self.rate_limiter:
    self.rate_limiter.acquire()
```

**Current Limitations**:

1. No distributed rate limiting (critical for multi-instance deployments)
2. No Redis/Memcached backend support
3. Single algorithm (token bucket only)
4. No rate limit sharing across pipelines
5. Polling-based waiting (inefficient)
6. Manual thread safety implementation

### Why `limits` Library is Superior

**Key Benefits**:

1. **Distributed Rate Limiting**: Redis/Memcached backends for multi-instance coordination
2. **Multiple Algorithms**: Fixed window, sliding window, moving window, token bucket
3. **Battle-Tested**: Production-ready, actively maintained (2000+ GitHub stars)
4. **Zero Maintenance**: Delegate complexity to specialized library
5. **Async Support**: Identical API for sync/async codebases
6. **Storage Backends**: Memory, Redis, Memcached, MongoDB
7. **Decorator Support**: Easy function-level rate limiting

**Alignment with Memory [[memory:11232352]]**: "Always check if a library already exists before implementing custom solutions. Use tenacity for retries (already a dependency), use limits library for rate limiting instead of custom implementation."

---

## Part 2: Architecture Design & Options (Strategist + Expert)

### Option 1: Drop-In Replacement (Minimal Changes) ⭐ RECOMMENDED

**Approach**: Replace `RateLimiter` class with `limits` wrapper, maintain existing API

**Implementation**:

```python
# ondine/utils/rate_limiter.py (REPLACE entire file)
from limits import RateLimitItemPerMinute
from limits.strategies import MovingWindowRateLimiter
from limits.storage import MemoryStorage

class RateLimiter:
    """Wrapper around limits library for backward compatibility."""

    def __init__(self, requests_per_minute: int, burst_size: int | None = None,
                 storage_uri: str | None = None):
        self.rpm = requests_per_minute
        self.capacity = burst_size or requests_per_minute

        # Initialize limits library
        self.limit = RateLimitItemPerMinute(requests_per_minute)

        # Storage backend (memory or Redis)
        if storage_uri:
            from limits.storage import storage_from_string
            self.storage = storage_from_string(storage_uri)
        else:
            self.storage = MemoryStorage()

        # Strategy (moving window = smooth token bucket)
        self.limiter = MovingWindowRateLimiter(self.storage)

    def acquire(self, tokens: int = 1, timeout: float | None = None) -> bool:
        """Acquire tokens (backward compatible API)."""
        # limits library doesn't support multi-token acquire directly
        # Simulate by checking N times
        for _ in range(tokens):
            if not self.limiter.hit(self.limit):
                return False
        return True

    @property
    def available_tokens(self) -> float:
        """Get available tokens (approximate)."""
        stats = self.limiter.get_window_stats(self.limit)
        return self.capacity - stats.hit_count

    def reset(self) -> None:
        """Reset rate limiter."""
        self.limiter.clear(self.limit)
```

**Changes Required**:

- `ondine/utils/rate_limiter.py`: Replace entire file (94 lines → ~60 lines)
- `ondine/core/specifications.py`: Add optional `storage_uri` field to `ProcessingSpec`
- `ondine/api/pipeline.py`: Pass `storage_uri` to `RateLimiter` constructor
- `pyproject.toml`: Add `limits>=3.0.0` dependency
- `tests/unit/test_rate_limiter.py`: Update tests for new backend

**Pros**:

- ✅ Minimal code changes (backward compatible)
- ✅ Existing API preserved (no breaking changes)
- ✅ Gradual migration path (memory → Redis later)
- ✅ All existing tests pass with minor updates

**Cons**:

- ⚠️ Multi-token acquire requires workaround (loop)
- ⚠️ Some `limits` features not exposed

**Effort**: 4 hours (Low)

---

### Option 2: Native `limits` API (Modern Approach)

**Approach**: Expose `limits` library directly, deprecate custom wrapper

**Implementation**:

```python
# ondine/api/pipeline.py (lines 527-536)
from limits import parse
from limits.strategies import MovingWindowRateLimiter
from limits.storage import MemoryStorage, storage_from_string

# Create rate limiter
if specs.processing.rate_limit_rpm:
    limit = parse(f"{specs.processing.rate_limit_rpm}/minute")
    storage = storage_from_string(specs.processing.storage_uri or "memory://")
    rate_limiter = MovingWindowRateLimiter(storage)
else:
    rate_limiter = None

# In llm_invocation_stage.py
if self.rate_limiter:
    if not self.rate_limiter.hit(self.limit):
        # Wait and retry
        time.sleep(0.1)
```

**Changes Required**:

- Remove `ondine/utils/rate_limiter.py` entirely
- Update `ondine/stages/llm_invocation_stage.py` to use `limits` API directly
- Update `ondine/api/pipeline.py` to create `limits` objects
- Update all tests

**Pros**:

- ✅ Full access to `limits` features
- ✅ Less code to maintain (remove custom wrapper)
- ✅ More Pythonic (use library as intended)

**Cons**:

- ❌ Breaking API change (major version bump required)
- ❌ More invasive changes across codebase
- ❌ Steeper learning curve for contributors

**Effort**: 8 hours (Medium)

---

### Option 3: Hybrid Approach (Gradual Migration)

**Approach**: Keep wrapper but add advanced features via optional parameters

**Implementation**:

```python
class RateLimiter:
    def __init__(self, requests_per_minute: int, burst_size: int | None = None,
                 storage_uri: str | None = None, strategy: str = "moving-window"):
        # Support multiple strategies
        if strategy == "fixed-window":
            self.limiter = FixedWindowRateLimiter(self.storage)
        elif strategy == "sliding-window":
            self.limiter = SlidingWindowRateLimiter(self.storage)
        else:
            self.limiter = MovingWindowRateLimiter(self.storage)
```

**Pros**:

- ✅ Backward compatible + new features
- ✅ Gradual adoption of advanced features

**Cons**:

- ⚠️ More complex wrapper code
- ⚠️ Maintenance burden remains

**Effort**: 6 hours (Medium)

---

### Option 4: Distributed-First (Redis Required)

**Approach**: Make Redis mandatory, optimize for multi-instance deployments

**Implementation**:

```python
# Require Redis for production
rate_limiter = RateLimiter(
    requests_per_minute=60,
    storage_uri="redis://localhost:6379"  # Required
)
```

**Pros**:

- ✅ True distributed rate limiting
- ✅ Simpler implementation (one backend)

**Cons**:

- ❌ Breaking change (Redis now required)
- ❌ Complicates local development
- ❌ Not suitable for single-instance use cases

**Effort**: 10 hours (High)

---

### Option 5: Plugin Architecture (Maximum Flexibility)

**Approach**: Abstract rate limiter interface, support multiple backends

**Implementation**:

```python
class RateLimiterBackend(ABC):
    @abstractmethod
    def acquire(self, tokens: int) -> bool: ...

class LimitsBackend(RateLimiterBackend):
    # Uses limits library

class CustomBackend(RateLimiterBackend):
    # Keep existing implementation
```

**Pros**:

- ✅ Maximum flexibility
- ✅ Easy to add new backends

**Cons**:

- ❌ Over-engineering for current needs
- ❌ High complexity

**Effort**: 12 hours (High)

---

## Part 3: Consensus & Recommendation (All Roles)

### Decision Matrix

| Criteria | Option 1 | Option 2 | Option 3 | Option 4 | Option 5 |

|----------|----------|----------|----------|----------|----------|

| Backward Compatible | ✅ Yes | ❌ No | ✅ Yes | ❌ No | ✅ Yes |

| Distributed Support | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |

| Code Reduction | ✅ 36% | ✅ 100% | ⚠️ 0% | ✅ 50% | ❌ +50% |

| Effort | ✅ 4h | ⚠️ 8h | ⚠️ 6h | ❌ 10h | ❌ 12h |

| Breaking Changes | ✅ None | ❌ Major | ✅ None | ❌ Major | ✅ None |

| Maintenance | ✅ Low | ✅ Minimal | ⚠️ Medium | ✅ Low | ❌ High |

### Consensus: **Option 1 - Drop-In Replacement** ⭐

**Strategist**: "Option 1 provides the best ROI - minimal effort, maximum compatibility, enables future distributed deployments."

**Expert**: "The `limits` library's moving window algorithm is superior to our token bucket. The wrapper maintains our API contract while delegating complexity."

**Executor**: "Implementation is straightforward - replace one file, add one dependency, update tests. Low risk."

**Critic**: "Multi-token acquire workaround is acceptable since we only use `acquire(1)` in practice. The abstraction is clean and testable."

**Final Verdict**: Option 1 balances pragmatism with improvement. It delivers distributed rate limiting without breaking existing code.

---

## Part 4: Implementation Plan (Executor)

### Phase 1: Core Migration (2 hours)

1. **Add dependency** (`pyproject.toml`):
```toml
dependencies = [
    # ... existing ...
    "limits>=3.0.0",
]
```

2. **Replace RateLimiter** (`ondine/utils/rate_limiter.py`):

   - Import `limits` library
   - Implement wrapper class (60 lines)
   - Maintain existing API: `__init__()`, `acquire()`, `available_tokens`, `reset()`
   - Add optional `storage_uri` parameter

3. **Update specifications** (`ondine/core/specifications.py` line 369):
```python
rate_limit_rpm: int | None = Field(default=None, gt=0)
rate_limit_storage: str | None = Field(
    default=None,
    description="Storage backend URI (memory://, redis://host:port, memcached://)"
)
```

4. **Update pipeline** (`ondine/api/pipeline.py` line 528):
```python
rate_limiter = RateLimiter(
    specs.processing.rate_limit_rpm,
    burst_size=min(20, specs.processing.concurrency),
    storage_uri=specs.processing.rate_limit_storage  # NEW
) if specs.processing.rate_limit_rpm else None
```


### Phase 2: Testing (1 hour)

5. **Update unit tests** (`tests/unit/test_rate_limiter.py`):

   - Test memory backend (default)
   - Test Redis backend (if available)
   - Test backward compatibility
   - Test multi-token acquire workaround

6. **Add integration test**:
```python
def test_distributed_rate_limiting_redis():
    """Test rate limiting across multiple pipeline instances."""
    # Requires Redis running
    limiter1 = RateLimiter(60, storage_uri="redis://localhost:6379")
    limiter2 = RateLimiter(60, storage_uri="redis://localhost:6379")

    # Both should share the same limit
    for _ in range(30):
        limiter1.acquire()

    # limiter2 should see reduced capacity
    assert limiter2.available_tokens < 30
```


### Phase 3: Documentation (1 hour)

7. **Update technical reference** (`docs/architecture/technical-reference.md` lines 696-943):

   - Document new `limits` library integration
   - Add distributed rate limiting examples
   - Update architecture diagrams

8. **Add usage examples**:
```python
# Example: Distributed rate limiting with Redis
pipeline = (
    PipelineBuilder.create()
    .with_rate_limit(rpm=60)
    .with_rate_limit_storage("redis://localhost:6379")  # NEW
    .build()
)
```


### Phase 4: Migration Guide (Optional)

9. **Create migration guide** for users:

   - No breaking changes (backward compatible)
   - Optional Redis setup for distributed deployments
   - Performance comparison (before/after)

---

## Part 5: Risk Assessment & Mitigation (Critic)

### Risks Identified

**Risk 1: Multi-token acquire workaround**

- **Impact**: Medium (performance overhead for `acquire(N)` where N > 1)
- **Likelihood**: Low (we only use `acquire(1)` in current codebase)
- **Mitigation**: Document limitation, consider custom extension if needed

**Risk 2: Redis dependency for distributed deployments**

- **Impact**: Low (optional feature)
- **Likelihood**: Medium (users may not have Redis)
- **Mitigation**: Keep memory backend as default, Redis is opt-in

**Risk 3: `limits` library maintenance**

- **Impact**: High (if library becomes unmaintained)
- **Likelihood**: Very Low (active project, 2000+ stars, recent commits)
- **Mitigation**: Wrapper isolates us from library changes

**Risk 4: Breaking changes in `limits` API**

- **Impact**: Medium (would require wrapper updates)
- **Likelihood**: Low (mature library, stable API)
- **Mitigation**: Pin version range in `pyproject.toml` (`limits>=3.0.0,<4.0.0`)

### Rollback Plan

If migration fails:

1. Revert `ondine/utils/rate_limiter.py` to original implementation
2. Remove `limits` dependency from `pyproject.toml`
3. Revert specification changes
4. All existing tests should pass (no breaking changes)

---

## Part 6: Success Metrics

**Quantitative**:

- Code reduction: 94 lines → 60 lines (36% reduction)
- Test coverage: Maintain 100% for rate limiter module
- Performance: No regression in single-instance benchmarks
- Distributed: Rate limit shared across 2+ instances (new capability)

**Qualitative**:

- Cleaner codebase (delegate to specialized library)
- Production-ready distributed rate limiting
- Future-proof (Redis/Memcached support)
- Reduced maintenance burden

---

## Conclusion

**Option 1 (Drop-In Replacement)** is the clear winner:

- ✅ Minimal effort (4 hours)
- ✅ Zero breaking changes
- ✅ Enables distributed rate limiting
- ✅ 36% code reduction
- ✅ Battle-tested library
- ✅ Aligns with memory [[memory:11232352]]

**Next Steps**:

1. Get user approval for Option 1
2. Execute Phase 1-3 implementation
3. Run full test suite
4. Update documentation
5. Deploy to staging for validation

### To-dos

- [ ] Add limits>=3.0.0 to pyproject.toml dependencies
- [ ] Replace ondine/utils/rate_limiter.py with limits wrapper (60 lines)
- [ ] Add rate_limit_storage field to ProcessingSpec in specifications.py
- [ ] Update pipeline.py to pass storage_uri to RateLimiter
- [ ] Update test_rate_limiter.py for new backend
- [ ] Add distributed rate limiting integration test (Redis)
- [ ] Update technical-reference.md with limits library integration
- [ ] Add distributed rate limiting usage examples
