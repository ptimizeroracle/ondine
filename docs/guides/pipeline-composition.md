# Pipeline Composition

Some tasks need multiple LLM passes where later steps depend on earlier results. Calculate a similarity score first, then explain it. Tag a category first, then generate a recommendation based on that category. `PipelineComposer` wires these dependencies together and handles execution order automatically.

## Basic Usage

```python
from ondine import PipelineBuilder, PipelineComposer

# Pipeline 1: Calculate similarity
similarity_pipeline = (
    PipelineBuilder.create()
    .with_prompt("Calculate similarity (0-1): {text1} vs {text2}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .build()
)

# Pipeline 2: Explain (depends on similarity result)
explanation_pipeline = (
    PipelineBuilder.create()
    .with_prompt("""
        The similarity score is {similarity}.
        Explain why these texts are similar or different:
        Text 1: {text1}
        Text 2: {text2}
    """)
    .with_llm(provider="openai", model="gpt-4o-mini")
    .build()
)

# Compose pipelines
composer = (
    PipelineComposer(input_data="data.csv")
    .add_column("similarity", similarity_pipeline)
    .add_column("explanation", explanation_pipeline, depends_on=["similarity"])
)

result = composer.execute()
```

## Dependencies

Pass `depends_on` to declare that one column's pipeline needs another column's output. Ondine will not run a dependent pipeline until its prerequisites finish:

```python
composer = (
    PipelineComposer(input_data=df)
    .add_column("category", category_pipeline)  # No dependencies
    .add_column("sentiment", sentiment_pipeline)  # No dependencies
    .add_column("recommendation", recommendation_pipeline, 
                depends_on=["category", "sentiment"])  # Depends on both
)
```

## Execution Order

Independent pipelines (no `depends_on`) run first and can execute in parallel when using async. Once they finish, their output columns become available as input for downstream pipelines. A pipeline will not start until every column listed in its `depends_on` has been populated. This means you get automatic parallelism where possible and sequential execution where required, without manually orchestrating anything.

## Related

- [Multi-Column Processing](multi-column.md)
- [Core Concepts](../getting-started/core-concepts.md)

