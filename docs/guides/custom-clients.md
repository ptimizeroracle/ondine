# Custom LLM Clients

Ondine talks to providers through LiteLLM, which covers most cases. When it
doesn't, supply your own client and the pipeline uses it everywhere — batching,
concurrency, retries, cost tracking, streaming and checkpointing all work
unchanged.

Reach for this when you need to:

- route through an **internal gateway** that adds auth, quota or audit logging
- call an API that is **not OpenAI-shaped** and has no LiteLLM route
- add a **caching or rate-limiting layer** of your own
- **stub the provider in tests** without patching internals

## The interface

Implement three methods:

```python
from decimal import Decimal
from typing import Any

from ondine.adapters.llm_client import LLMClient
from ondine.core.models import LLMResponse
from ondine.core.specifications import LLMSpec


class GatewayClient(LLMClient):
    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        text = my_gateway.complete(prompt)          # your call here
        return LLMResponse(
            text=text,
            tokens_in=self.estimate_tokens(prompt),
            tokens_out=self.estimate_tokens(text),
            model=self.model,
            cost=self.calculate_cost(...),
            latency_ms=...,
        )

    def structured_invoke(self, prompt: str, output_cls: Any, **kwargs: Any):
        # Called when the pipeline uses with_structured_output().
        # Ask your API for JSON and validate into output_cls, or raise
        # NotImplementedError if it has no structured mode.
        ...

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4
```

Then inject it:

```python
pipeline = (
    PipelineBuilder.create()
    .from_dataframe(df, input_columns=["text"], output_columns=["result"])
    .with_prompt("Summarize: {text}")
    .with_custom_llm_client(GatewayClient(LLMSpec(provider="openai", model="internal-v2")))
    .build()
)
```

`with_llm()` is not needed — the spec you pass to your client carries the model
name, pricing and limits. If you call both, the custom client wins.

A runnable version is in
[`examples/13_custom_client.py`](https://github.com/ptimizeroracle/ondine/blob/main/examples/13_custom_client.py).

## Async: the method that actually runs

The default executor is async, so **`ainvoke()` is what a run calls**, not
`invoke()`. The base class delegates it to `invoke()` for you, which is correct
but serialises requests — the sync call blocks the event loop.

For real concurrency, override it:

```python
    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        text = await my_async_gateway.complete(prompt)
        return LLMResponse(...)
```

Same for `structured_invoke_async()`.

## Optional lifecycle hooks

If your client holds a session, a connection pool or a subprocess, override
these. They default to doing nothing.

```python
    async def start(self) -> None:
        self._session = aiohttp.ClientSession()

    async def stop(self) -> None:
        await self._session.close()
```

`start()` runs before the first request of a run and `stop()` after the last.

## What you get for free

Your client is called through the full pipeline, so everything else keeps
working:

| | |
|---|---|
| **Batching** | `with_batch_size()` groups rows into one prompt before reaching you |
| **Concurrency** | `with_concurrency()` runs many `ainvoke()` calls at once |
| **Retries** | Transient failures retry per your error policy |
| **Cost tracking** | Whatever you report in `LLMResponse.cost` is summed and budgeted |
| **Checkpointing** | Responses are cached for resume, same as any provider |

## One instance, shared

The pipeline uses **the object you passed**, not a copy — including in
streamed runs, where each chunk is its own sub-pipeline, and in the automatic
retry pass. State your client holds (a session, a token, a counter, a shared
rate limiter) is shared across all of them.

That matters if your client is stateful: a connection pool is opened once, and
a rate limiter genuinely limits the whole run rather than per chunk.

## Testing with a stub

The same mechanism makes provider-free tests straightforward:

```python
class StubClient(LLMClient):
    def __init__(self, spec, answer="ok"):
        super().__init__(spec)
        self.answer = answer
        self.prompts: list[str] = []

    def invoke(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return LLMResponse(
            text=self.answer, tokens_in=1, tokens_out=1, model="stub",
            cost=Decimal("0"), latency_ms=0.0,
        )

    def structured_invoke(self, prompt, output_cls, **kwargs):
        return self.invoke(prompt)

    def estimate_tokens(self, text):
        return 1
```

Assert on `stub.prompts` to check what your pipeline actually sent — no network,
no keys, no patching of ondine internals.
