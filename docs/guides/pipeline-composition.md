# Pipeline Composition

Compose multiple pipelines to process independent columns with dependencies between them.

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

Specify dependencies between columns:

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

Pipelines execute in dependency order:
1. Independent pipelines run first (parallel if async)
2. Dependent pipelines wait for their dependencies
3. Results from previous pipelines are available as input columns

## Related

- [Multi-Column Processing](multi-column.md)
- [Core Concepts](../getting-started/core-concepts.md)

