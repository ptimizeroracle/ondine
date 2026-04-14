# Multi-Column Processing

One LLM call can populate several output columns at once. Instead of running separate pipelines for brand, category, and price, you ask the model to return a JSON object and let Ondine split it into columns. This cuts API calls proportionally: three output columns from one call means one-third the cost.

## Basic Usage

Define your output columns in `from_csv`, then tell the model to return matching JSON keys:

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

JSON parsing alone does not enforce types. If the model returns `"price": "twenty bucks"`, you will not know until downstream code breaks. Pydantic catches this at parse time:

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

You can feed several columns into the prompt template. Each `{placeholder}` maps to a column name from `input_columns`:

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

