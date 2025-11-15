"""Processing stages for data transformation."""

from ondine.stages.batch_aggregator_stage import BatchAggregatorStage
from ondine.stages.batch_disaggregator_stage import BatchDisaggregatorStage
from ondine.stages.data_loader_stage import DataLoaderStage
from ondine.stages.llm_invocation_stage import LLMInvocationStage
from ondine.stages.multi_run_stage import (
    AggregationStrategy,
    AllStrategy,
    AverageStrategy,
    ConsensusStrategy,
    FirstSuccessStrategy,
    MultiRunStage,
)
from ondine.stages.parser_factory import create_response_parser
from ondine.stages.pipeline_stage import PipelineStage
from ondine.stages.prompt_formatter_stage import (
    PromptFormatterStage,
)
from ondine.stages.response_parser_stage import (
    JSONParser,
    PydanticParser,
    RawTextParser,
    RegexParser,
    ResponseParser,
    ResponseParserStage,
)
from ondine.stages.result_writer_stage import ResultWriterStage
from ondine.stages.stage_registry import StageRegistry, stage

__all__ = [
    "PipelineStage",
    "BatchAggregatorStage",
    "BatchDisaggregatorStage",
    "DataLoaderStage",
    "PromptFormatterStage",
    "LLMInvocationStage",
    "ResponseParserStage",
    "ResultWriterStage",
    "MultiRunStage",
    "ResponseParser",
    "RawTextParser",
    "JSONParser",
    "PydanticParser",
    "RegexParser",
    "create_response_parser",
    "AggregationStrategy",
    "ConsensusStrategy",
    "FirstSuccessStrategy",
    "AllStrategy",
    "AverageStrategy",
    # Stage Registry
    "StageRegistry",
    "stage",
]
