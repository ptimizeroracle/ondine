"""
Configuration File Example - Load pipeline from YAML/JSON.

This example shows how to define pipelines in configuration files
for easy management and version control.
"""

from ondine.api.pipeline import Pipeline
from ondine.config import ConfigLoader

# Load configuration from YAML
print("Loading configuration from YAML...")
specifications = ConfigLoader.from_yaml("examples/config_example.yaml")

print(f"Pipeline: {specifications.metadata.get('project', 'N/A')}")
print(f"LLM Provider: {specifications.llm.provider}")
print(f"Model: {specifications.llm.model}")
print(f"Batch Size: {specifications.processing.batch_size}")
print(f"Max Budget: ${specifications.processing.max_budget}")

# Create pipeline from specifications
pipeline = Pipeline(specifications)

# Validate configuration
validation = pipeline.validate()
if validation.is_valid:
    print("\n✅ Configuration is valid")
else:
    print(f"\n❌ Configuration errors: {validation.errors}")

# Note: To actually execute, you would need:
# 1. Create products.csv file
# 2. Set GROQ_API_KEY environment variable
# 3. Run: result = pipeline.execute()

print("\nTo execute this pipeline:")
print("  1. Create 'products.csv' with a 'description' column")
print("  2. export GROQ_API_KEY='your-key-here'")
print("  3. Uncomment and run: result = pipeline.execute()")

# Example: Save specifications to JSON
output_config_path = "/tmp/pipeline_config.json"
ConfigLoader.to_json(specifications, output_config_path)
print(f"\nConfiguration saved to: {output_config_path}")

# Example: Load from JSON
specs_from_json = ConfigLoader.from_json(output_config_path)
print("✅ Successfully loaded from JSON")
