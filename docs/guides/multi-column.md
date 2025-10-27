# Multi-Column Processing

Generate multiple output columns from a single LLM call using JSON parsing.

## Basic Usage

Use JSON parsing to extract multiple fields:

```python
from ondine import PipelineBuilder
from ondine.stages.parser_factory import JSONParser

pipeline = (
    PipelineBuilder.create()
    .from_csv(
        "products.csv",
        input_columns=["description"],
        output_columns=["brand", "category", "price"]
    )
    .with_prompt("""
        Extract product information as JSON:
        {{
          "brand": "...",
          "category": "...",
          "price": "..."
        }}
        
        Description: {description}
    """)
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_parser(JSONParser())
    .build()
)

result = pipeline.execute()
# Result has 3 new columns: brand, category, price
```

## With Pydantic Validation

For type-safe validation:

```python
from pydantic import BaseModel
from ondine.stages.response_parser_stage import PydanticParser

class ProductInfo(BaseModel):
    brand: str
    category: str
    price: float

pipeline = (
    PipelineBuilder.create()
    .from_csv("products.csv", ...)
    .with_prompt("...")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_parser(PydanticParser(ProductInfo))
    .build()
)
```

## Multiple Input Columns

Use multiple input columns in your prompt:

```python
pipeline = (
    PipelineBuilder.create()
    .from_csv(
        "products.csv",
        input_columns=["title", "description", "category"],
        output_columns=["brand", "model", "price"]
    )
    .with_prompt("""
        Extract product information:
        {{
          "brand": "...",
          "model": "...",
          "price": 0.0
        }}
        
        Title: {title}
        Description: {description}
        Category: {category}
    """)
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_parser(JSONParser())
    .build()
)
```

## Related

- [Structured Output](structured-output.md) - Pydantic models
- [Pipeline Composition](pipeline-composition.md) - Complex workflows

