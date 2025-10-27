"""
Stage registry for extensible pipeline stage plugins.

Enables custom pipeline stages to be registered and injected into
processing pipelines without modifying core code.
"""


class PipelineStage:
    """Protocol for pipeline stage implementations (imported to avoid circular dependency)."""

    pass


class StageRegistry:
    """
    Global registry for custom pipeline stages.

    Enables registration and discovery of custom stages that can be
    injected into pipelines at specific positions.

    Example:
        # Register custom stage
        @StageRegistry.register("rag_retrieval")
        class RAGRetrievalStage(PipelineStage):
            def execute(self, context):
                # Custom retrieval logic
                ...

        # Use in pipeline
        pipeline = (
            PipelineBuilder.create()
            .with_stage("rag_retrieval", position="before_prompt", index="my-docs")
            .build()
        )
    """

    _stages: dict[str, type] = {}

    @classmethod
    def register(cls, stage_name: str, stage_class: type) -> type:
        """
        Register a custom pipeline stage.

        Args:
            stage_name: Unique stage identifier (e.g., "rag_retrieval", "content_moderation")
            stage_class: Stage class implementing PipelineStage interface

        Returns:
            The registered stage class (enables use as decorator)

        Raises:
            ValueError: If stage_name already registered

        Example:
            @StageRegistry.register("fact_checker")
            class FactCheckerStage(PipelineStage):
                def execute(self, context):
                    # Verify LLM output against sources
                    ...
        """
        if stage_name in cls._stages:
            raise ValueError(
                f"Stage '{stage_name}' already registered. "
                f"Use a different stage_name or unregister first."
            )

        cls._stages[stage_name] = stage_class
        return stage_class

    @classmethod
    def get(cls, stage_name: str) -> type:
        """
        Get stage class by name.

        Args:
            stage_name: Stage identifier

        Returns:
            Pipeline stage class

        Raises:
            ValueError: If stage not found

        Example:
            stage_class = StageRegistry.get("rag_retrieval")
            stage = stage_class(vector_store="pinecone", index="my-docs")
        """
        if stage_name not in cls._stages:
            available = ", ".join(sorted(cls._stages.keys()))
            raise ValueError(
                f"Unknown stage: '{stage_name}'. "
                f"Available stages: {available if available else 'none'}"
            )

        return cls._stages[stage_name]

    @classmethod
    def list_stages(cls) -> dict[str, type]:
        """
        List all registered stages.

        Returns:
            Dictionary mapping stage names to stage classes

        Example:
            stages = StageRegistry.list_stages()
            print(f"Available custom stages: {list(stages.keys())}")
        """
        return cls._stages.copy()

    @classmethod
    def is_registered(cls, stage_name: str) -> bool:
        """
        Check if stage is registered.

        Args:
            stage_name: Stage identifier

        Returns:
            True if registered, False otherwise
        """
        return stage_name in cls._stages

    @classmethod
    def unregister(cls, stage_name: str) -> None:
        """
        Unregister a stage (mainly for testing).

        Args:
            stage_name: Stage identifier

        Raises:
            ValueError: If stage not found
        """
        if stage_name not in cls._stages:
            raise ValueError(f"Stage '{stage_name}' not registered")

        del cls._stages[stage_name]

    @classmethod
    def _reset(cls) -> None:
        """
        Reset registry (for testing only).

        Clears all registered stages.
        """
        cls._stages.clear()


def stage(name: str):
    """
    Decorator to register a custom pipeline stage.

    Args:
        name: Unique stage identifier

    Returns:
        Decorator function

    Example:
        @stage("rag_retrieval")
        class RAGRetrievalStage(PipelineStage):
            '''Retrieve context from vector store and enrich data.'''

            def __init__(self, vector_store: str, index_name: str, top_k: int = 3):
                super().__init__(name="rag_retrieval")
                self.vector_store = vector_store
                self.index_name = index_name
                self.top_k = top_k

            def execute(self, context: ExecutionContext) -> StageResult:
                # Retrieve context for each row
                enriched_rows = []
                for _, row in context.data.iterrows():
                    query = row['text']
                    results = self._retrieve(query)
                    row['retrieved_context'] = self._format_context(results)
                    enriched_rows.append(row)

                context.data = pd.DataFrame(enriched_rows)
                return StageResult(success=True, data=context.data)

            def _retrieve(self, query: str):
                # Integration with vector store
                ...

            def _format_context(self, results):
                # Format retrieved docs
                ...
    """

    def decorator(cls):
        StageRegistry.register(name, cls)
        return cls

    return decorator
