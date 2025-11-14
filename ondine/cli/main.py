"""
Main CLI entry point for LLM Dataset Engine.

Provides command-line interface for processing datasets, estimating costs,
and managing pipeline execution.
"""

import sys
from pathlib import Path
from uuid import UUID

import click
from rich.console import Console, Group
from rich.table import Table
from rich.text import Text

from ondine import __version__
from ondine.api import Pipeline
from ondine.config import ConfigLoader
from ondine.core.specifications import (
    DataSourceType,
    LLMProvider,
)

console = Console()


# Provider metadata for list-providers command
PROVIDER_METADATA = {
    LLMProvider.OPENAI: {
        "name": "OpenAI",
        "platform": "Cloud (All)",
        "cost": "$$",
        "use_case": "Production, high quality",
        "requirements": "OPENAI_API_KEY",
    },
    LLMProvider.AZURE_OPENAI: {
        "name": "Azure OpenAI",
        "platform": "Cloud (All)",
        "cost": "$$",
        "use_case": "Enterprise, compliance",
        "requirements": "AZURE_OPENAI_API_KEY",
    },
    LLMProvider.ANTHROPIC: {
        "name": "Anthropic Claude",
        "platform": "Cloud (All)",
        "cost": "$$$",
        "use_case": "Long context, high quality",
        "requirements": "ANTHROPIC_API_KEY",
    },
    LLMProvider.GROQ: {
        "name": "Groq",
        "platform": "Cloud (All)",
        "cost": "Free tier",
        "use_case": "Fast inference, development",
        "requirements": "GROQ_API_KEY",
    },
    LLMProvider.OPENAI_COMPATIBLE: {
        "name": "OpenAI-Compatible",
        "platform": "Custom/Local/Cloud",
        "cost": "Varies",
        "use_case": "Ollama, vLLM, Together.AI, custom APIs",
        "requirements": "base_url (API key optional)",
    },
    LLMProvider.MLX: {
        "name": "Apple MLX",
        "platform": "macOS (M1/M2/M3/M4)",
        "cost": "Free",
        "use_case": "Local inference, privacy, offline",
        "requirements": "Apple Silicon Mac, pip install ondine[mlx]",
    },
}

# Validate metadata completeness at module load
assert set(LLMProvider) == set(PROVIDER_METADATA.keys()), (
    "PROVIDER_METADATA must include all LLMProvider values"
)


ONDINE_ART = r"""
 ▄▄▄▄▄▄▄▄▄▄▄  ▄▄        ▄  ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄  ▄▄        ▄  ▄▄▄▄▄▄▄▄▄▄▄
▐░░░░░░░░░░░▌▐░░▌      ▐░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░▌      ▐░▌▐░░░░░░░░░░░▌
▐░█▀▀▀▀▀▀▀█░▌▐░▌░▌     ▐░▌▐░█▀▀▀▀▀▀▀█░▌ ▀▀▀▀█░█▀▀▀▀ ▐░▌░▌     ▐░▌▐░█▀▀▀▀▀▀▀▀▀
▐░▌       ▐░▌▐░▌▐░▌    ▐░▌▐░▌       ▐░▌    ▐░▌     ▐░▌▐░▌    ▐░▌▐░▌
▐░▌       ▐░▌▐░▌ ▐░▌   ▐░▌▐░▌       ▐░▌    ▐░▌     ▐░▌ ▐░▌   ▐░▌▐░█▄▄▄▄▄▄▄▄▄
▐░▌       ▐░▌▐░▌  ▐░▌  ▐░▌▐░▌       ▐░▌    ▐░▌     ▐░▌  ▐░▌  ▐░▌▐░░░░░░░░░░░▌
▐░▌       ▐░▌▐░▌   ▐░▌ ▐░▌▐░▌       ▐░▌    ▐░▌     ▐░▌   ▐░▌ ▐░▌▐░█▀▀▀▀▀▀▀▀▀
▐░▌       ▐░▌▐░▌    ▐░▌▐░▌▐░▌       ▐░▌    ▐░▌     ▐░▌    ▐░▌▐░▌▐░▌
▐░█▄▄▄▄▄▄▄█░▌▐░▌     ▐░▐░▌▐░█▄▄▄▄▄▄▄█░▌▄▄▄▄█░█▄▄▄▄ ▐░▌     ▐░▐░▌▐░█▄▄▄▄▄▄▄▄▄
▐░░░░░░░░░░░▌▐░▌      ▐░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░▌      ▐░░▌▐░░░░░░░░░░░▌
 ▀▀▀▀▀▀▀▀▀▀▀  ▀        ▀▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀  ▀        ▀▀  ▀▀▀▀▀▀▀▀▀▀▀
"""


def show_banner():
    """Display the Ondine banner (centered, creative, robust)."""
    # Color gradient: cyan to magenta
    lines = ONDINE_ART.strip().split("\n")
    colored_lines = []
    colors = [
        "bright_cyan",
        "cyan",
        "bright_blue",
        "blue",
        "bright_magenta",
        "magenta",
        "bright_magenta",
        "blue",
        "bright_blue",
        "cyan",
        "bright_cyan",
    ]

    for i, line in enumerate(lines):
        color = colors[i % len(colors)]
        colored_lines.append(Text(line, style=f"bold {color}"))

    title = Group(*colored_lines)
    subtitle_text = Text("The LLM Dataset Engine", style="dim italic bright_white")
    content = Group(title, "", subtitle_text)

    console.print()
    console.print(content)
    console.print("[bold bright_cyan]" + "─" * 80 + "[/bold bright_cyan]")
    console.print()


@click.group()
@click.version_option(version=__version__, prog_name="ondine")
@click.pass_context
def cli(ctx):
    """
    🌊 ONDINE - LLM Dataset Engine

    Process tabular datasets using LLMs with built-in reliability,
    cost control, and observability.

    Examples:

        # Process a dataset
        ondine process --config config.yaml

        # Estimate cost before processing
        ondine estimate --config config.yaml

        # Resume from checkpoint
        ondine resume --session-id abc-123

        # Validate configuration
        ondine validate --config config.yaml
    """
    # Show banner only for main commands (not for --help)
    if ctx.invoked_subcommand is not None:
        show_banner()


@cli.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to YAML/JSON configuration file",
)
@click.option(
    "--input",
    "-i",
    required=False,
    type=click.Path(exists=True, path_type=Path),
    help="Path to input data file (overrides config)",
)
@click.option(
    "--output",
    "-o",
    required=False,
    type=click.Path(path_type=Path),
    help="Path to output file (overrides config)",
)
@click.option(
    "--provider",
    type=click.Choice([p.value for p in LLMProvider]),
    help="Override LLM provider from config (use 'ondine list-providers' to see all)",
)
@click.option(
    "--model",
    help="Override model name from config",
)
@click.option(
    "--max-budget",
    type=float,
    help="Override maximum budget (USD) from config",
)
@click.option(
    "--batch-size",
    type=int,
    help="Override batch size from config",
)
@click.option(
    "--concurrency",
    type=int,
    help="Override concurrency from config",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(path_type=Path),
    help="Override checkpoint directory from config",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate and estimate only, don't execute",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def process(
    config: Path,
    input: Path | None,
    output: Path | None,
    provider: str | None,
    model: str | None,
    max_budget: float | None,
    batch_size: int | None,
    concurrency: int | None,
    checkpoint_dir: Path | None,
    dry_run: bool,
    verbose: bool,
):
    """
    Process a dataset using LLM transformations.

    Reads data from config file. INPUT and OUTPUT flags override config values if provided.

    Examples:

        # Basic usage
        llm-dataset process -c config.yaml -i data.csv -o result.csv

        # Override provider and model
        llm-dataset process -c config.yaml -i data.csv -o result.csv \\
            --provider groq --model openai/gpt-oss-120b

        # Set budget limit
        llm-dataset process -c config.yaml -i data.csv -o result.csv \\
            --max-budget 10.0

        # Dry run (estimate only)
        llm-dataset process -c config.yaml -i data.csv -o result.csv --dry-run
    """
    try:
        # Load configuration
        console.print(f"[cyan]Loading configuration from {config}...[/cyan]")
        specs = ConfigLoader.from_yaml(str(config))

        # Override with CLI arguments (if provided)
        if input:
            specs.dataset.source_path = input

        # Set output configuration (if provided)
        if output:
            if specs.output:
                specs.output.destination_path = output
            else:
                # Create output spec if not in config
                from ondine.core.specifications import MergeStrategy, OutputSpec

                # Detect output type from extension
                output_suffix = output.suffix.lower()
                if output_suffix == ".csv":
                    output_type = DataSourceType.CSV
                elif output_suffix in [".xlsx", ".xls"]:
                    output_type = DataSourceType.EXCEL
                elif output_suffix == ".parquet":
                    output_type = DataSourceType.PARQUET
                else:
                    output_type = DataSourceType.CSV  # Default

                specs.output = OutputSpec(
                    destination_type=output_type,
                    destination_path=output,
                    merge_strategy=MergeStrategy.REPLACE,
                )

        if provider:
            specs.llm.provider = LLMProvider(provider)

        if model:
            specs.llm.model = model

        if max_budget is not None:
            from decimal import Decimal

            specs.processing.max_budget = Decimal(str(max_budget))

        if batch_size is not None:
            specs.processing.batch_size = batch_size

        if concurrency is not None:
            specs.processing.concurrency = concurrency

        if checkpoint_dir is not None:
            specs.processing.checkpoint_dir = checkpoint_dir

        # Create pipeline
        console.print("[cyan]Creating pipeline...[/cyan]")
        pipeline = Pipeline(specs)

        # Validate
        console.print("[cyan]Validating pipeline...[/cyan]")
        validation = pipeline.validate()

        if not validation.is_valid:
            console.print("[red]❌ Validation failed:[/red]")
            for error in validation.errors:
                console.print(f"  [red]• {error}[/red]")
            sys.exit(1)

        console.print("[green]✅ Validation passed[/green]")

        # Estimate cost
        console.print("\n[cyan]Estimating cost...[/cyan]")
        estimate = pipeline.estimate_cost()

        table = Table(title="Cost Estimate")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Cost", f"${estimate.total_cost}")
        table.add_row("Total Tokens", f"{estimate.total_tokens:,}")
        table.add_row("Input Tokens", f"{estimate.input_tokens:,}")
        table.add_row("Output Tokens", f"{estimate.output_tokens:,}")
        table.add_row("Rows", f"{estimate.rows:,}")

        console.print(table)

        if dry_run:
            console.print("\n[yellow]Dry run mode - skipping execution[/yellow]")
            return

        # Execute
        console.print("\n[cyan]Processing dataset...[/cyan]")
        result = pipeline.execute()

        # Display results
        console.print("\n[green]✅ Processing complete![/green]")

        results_table = Table(title="Execution Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")

        results_table.add_row("Total Rows", str(result.metrics.total_rows))
        results_table.add_row("Processed", str(result.metrics.processed_rows))
        results_table.add_row("Failed", str(result.metrics.failed_rows))
        results_table.add_row("Skipped", str(result.metrics.skipped_rows))
        results_table.add_row("Duration", f"{result.duration:.2f}s")
        results_table.add_row("Total Cost", f"${result.costs.total_cost}")
        results_table.add_row(
            "Cost per Row",
            f"${result.costs.total_cost / result.metrics.total_rows:.6f}",
        )

        console.print(results_table)

        # Validate output quality
        console.print("\n[cyan]📊 Validating output quality...[/cyan]")
        quality = result.validate_output_quality(specs.dataset.output_columns)

        quality_table = Table(title="Quality Report")
        quality_table.add_column("Metric", style="cyan")
        quality_table.add_column("Value", style="green")

        quality_table.add_row(
            "Valid Outputs", f"{quality.valid_outputs}/{quality.total_rows}"
        )
        quality_table.add_row("Success Rate", f"{quality.success_rate:.1f}%")
        quality_table.add_row("Null Outputs", str(quality.null_outputs))
        quality_table.add_row("Empty Outputs", str(quality.empty_outputs))

        # Color-code quality score
        score_color = (
            "green"
            if quality.quality_score in ["excellent", "good"]
            else "yellow"
            if quality.quality_score == "poor"
            else "red"
        )
        quality_table.add_row(
            "Quality Score",
            f"[{score_color}]{quality.quality_score.upper()}[/{score_color}]",
        )

        console.print(quality_table)

        # Display warnings and issues
        if quality.warnings:
            console.print("\n[yellow]⚠️  Warnings:[/yellow]")
            for warning in quality.warnings:
                console.print(f"  [yellow]• {warning}[/yellow]")

        if quality.issues:
            console.print("\n[red]🚨 Issues Detected:[/red]")
            for issue in quality.issues:
                console.print(f"  [red]• {issue}[/red]")
            console.print("\n[red]Consider:[/red]")
            console.print(
                "  [dim]• Review your prompt complexity (simpler prompts often work better)[/dim]"
            )
            console.print("  [dim]• Check LLM provider logs for errors[/dim]")
            console.print("  [dim]• Increase max_tokens if outputs are truncated[/dim]")
            console.print("  [dim]• Verify API key and rate limits[/dim]")

        if quality.is_acceptable:
            console.print(
                f"\n[green]✅ Output quality is acceptable ({quality.success_rate:.1f}% success)[/green]"
            )
        else:
            console.print(
                f"\n[red]❌ Output quality is below acceptable threshold ({quality.success_rate:.1f}% < 70%)[/red]"
            )

        console.print(
            f"\n[green]Output written to: {specs.output.destination_path}[/green]"
        )

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to YAML/JSON configuration file",
)
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to input data file",
)
@click.option(
    "--provider",
    type=click.Choice([p.value for p in LLMProvider]),
    help="Override LLM provider from config (use 'ondine list-providers' to see all)",
)
@click.option(
    "--model",
    help="Override model name from config",
)
def estimate(
    config: Path,
    input: Path,
    provider: str | None,
    model: str | None,
):
    """
    Estimate processing cost without executing.

    Useful for budget planning and cost validation before running
    expensive operations.

    Examples:

        # Estimate cost
        llm-dataset estimate -c config.yaml -i data.csv

        # Estimate with different model
        llm-dataset estimate -c config.yaml -i data.csv --model gpt-4o
    """
    try:
        # Load configuration
        console.print(f"[cyan]Loading configuration from {config}...[/cyan]")
        specs = ConfigLoader.from_yaml(str(config))

        # Override
        specs.dataset.source_path = input

        if provider:
            specs.llm.provider = LLMProvider(provider)

        if model:
            specs.llm.model = model

        # Create pipeline
        pipeline = Pipeline(specs)

        # Validate
        validation = pipeline.validate()
        if not validation.is_valid:
            console.print("[red]❌ Validation failed:[/red]")
            for error in validation.errors:
                console.print(f"  [red]• {error}[/red]")
            sys.exit(1)

        # Estimate
        console.print("[cyan]Estimating cost...[/cyan]")
        estimate = pipeline.estimate_cost()

        # Display results
        table = Table(title="Cost Estimate", show_header=True)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="green", width=20)

        table.add_row("Total Cost", f"${estimate.total_cost}")
        table.add_row("Total Tokens", f"{estimate.total_tokens:,}")
        table.add_row("Input Tokens", f"{estimate.input_tokens:,}")
        table.add_row("Output Tokens", f"{estimate.output_tokens:,}")
        table.add_row("Rows to Process", f"{estimate.rows:,}")
        table.add_row("Confidence", estimate.confidence)

        console.print("\n")
        console.print(table)

        # Cost per row
        if estimate.rows > 0:
            cost_per_row = estimate.total_cost / estimate.rows
            console.print(f"\n[cyan]Cost per row: ${cost_per_row:.6f}[/cyan]")

        # Warning if expensive
        if estimate.total_cost > 10.0:
            console.print(
                f"\n[yellow]⚠️  Warning: Estimated cost (${estimate.total_cost}) exceeds $10[/yellow]"
            )

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--session-id",
    "-s",
    required=True,
    help="Session ID to resume (UUID)",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(exists=True, path_type=Path),
    default=".checkpoints",
    help="Checkpoint directory (default: .checkpoints)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Override output path",
)
def resume(
    session_id: str,
    checkpoint_dir: Path,
    output: Path | None,
):
    """
    Resume pipeline execution from checkpoint.

    Useful for recovering from failures or continuing interrupted processing.

    Examples:

        # Resume from checkpoint
        llm-dataset resume --session-id abc-123-def

        # Resume with custom checkpoint directory
        llm-dataset resume -s abc-123 --checkpoint-dir /path/to/checkpoints
    """
    try:
        from ondine.adapters import LocalFileCheckpointStorage
        from ondine.orchestration import StateManager

        # Load checkpoint
        console.print(f"[cyan]Looking for checkpoint in {checkpoint_dir}...[/cyan]")

        storage = LocalFileCheckpointStorage(str(checkpoint_dir))
        state_manager = StateManager(storage)

        session_uuid = UUID(session_id)

        if not state_manager.can_resume(session_uuid):
            console.print(f"[red]❌ No checkpoint found for session {session_id}[/red]")
            console.print(
                f"[yellow]Check checkpoint directory: {checkpoint_dir}[/yellow]"
            )
            sys.exit(1)

        # Load checkpoint
        checkpoint_info = state_manager.get_latest_checkpoint(session_uuid)
        console.print(
            f"[green]✅ Found checkpoint at row {checkpoint_info.row_index}[/green]"
        )

        # Resume execution
        console.print("[cyan]Resuming execution...[/cyan]")

        # Note: Full resume implementation would load the original pipeline
        # and continue from checkpoint. For now, we show the checkpoint info.
        console.print(
            "\n[yellow]⚠️  Full resume functionality requires the original pipeline configuration[/yellow]"
        )
        console.print(
            "[yellow]Please use Pipeline.execute(resume_from=session_id) in Python code[/yellow]"
        )

        # Display checkpoint info
        table = Table(title="Checkpoint Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Session ID", str(checkpoint_info.session_id))
        table.add_row("Checkpoint Path", checkpoint_info.checkpoint_path)
        table.add_row("Last Row", str(checkpoint_info.row_index))
        table.add_row("Last Stage", str(checkpoint_info.stage_index))
        table.add_row("Timestamp", str(checkpoint_info.timestamp))
        table.add_row("Size", f"{checkpoint_info.size_bytes:,} bytes")

        console.print(table)

    except ValueError:
        console.print(f"[red]❌ Invalid session ID format: {session_id}[/red]")
        console.print(
            "[yellow]Session ID should be a UUID (e.g., abc-123-def-456)[/yellow]"
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to YAML/JSON configuration file",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed validation results",
)
def validate(config: Path, verbose: bool):
    """
    Validate pipeline configuration.

    Checks configuration file for errors and warnings without executing
    the pipeline.

    Examples:

        # Validate configuration
        llm-dataset validate -c config.yaml

        # Verbose validation
        llm-dataset validate -c config.yaml --verbose
    """
    try:
        # Load configuration
        console.print(f"[cyan]Loading configuration from {config}...[/cyan]")
        specs = ConfigLoader.from_yaml(str(config))

        console.print("[green]✅ Configuration loaded successfully[/green]")

        # Display configuration summary
        if verbose:
            table = Table(title="Configuration Summary")
            table.add_column("Component", style="cyan")
            table.add_column("Details", style="green")

            table.add_row("Dataset", f"{specs.dataset.source_type.value}")
            table.add_row("Input Columns", ", ".join(specs.dataset.input_columns))
            table.add_row("Output Columns", ", ".join(specs.dataset.output_columns))
            table.add_row("LLM Provider", specs.llm.provider.value)
            table.add_row("Model", specs.llm.model)
            table.add_row("Batch Size", str(specs.processing.batch_size))
            table.add_row("Concurrency", str(specs.processing.concurrency))

            if specs.processing.max_budget:
                table.add_row("Max Budget", f"${specs.processing.max_budget}")

            console.print("\n")
            console.print(table)

        # Create pipeline for validation
        console.print("\n[cyan]Validating pipeline...[/cyan]")
        pipeline = Pipeline(specs)
        validation = pipeline.validate()

        if validation.is_valid:
            console.print("[green]✅ Pipeline configuration is valid[/green]")

            if validation.warnings:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in validation.warnings:
                    console.print(f"  [yellow]• {warning}[/yellow]")
        else:
            console.print("[red]❌ Pipeline configuration is invalid[/red]")
            console.print("\n[red]Errors:[/red]")
            for error in validation.errors:
                console.print(f"  [red]• {error}[/red]")

            if validation.warnings:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in validation.warnings:
                    console.print(f"  [yellow]• {warning}[/yellow]")

            sys.exit(1)

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--checkpoint-dir",
    type=click.Path(exists=True, path_type=Path),
    default=".checkpoints",
    help="Checkpoint directory to list (default: .checkpoints)",
)
def list_checkpoints(checkpoint_dir: Path):
    """
    List available checkpoints.

    Shows all saved checkpoints in the specified directory.

    Examples:

        # List checkpoints
        llm-dataset list-checkpoints

        # List from custom directory
        llm-dataset list-checkpoints --checkpoint-dir /path/to/checkpoints
    """
    try:
        from ondine.adapters import LocalFileCheckpointStorage

        console.print(f"[cyan]Scanning {checkpoint_dir} for checkpoints...[/cyan]")

        storage = LocalFileCheckpointStorage(checkpoint_dir)
        checkpoints = storage.list_checkpoints()

        if not checkpoints:
            console.print("[yellow]No checkpoints found[/yellow]")
            return

        # Display checkpoints
        table = Table(title=f"Checkpoints in {checkpoint_dir}")
        table.add_column("Session ID", style="cyan")
        table.add_column("Row", style="green")
        table.add_column("Stage", style="green")
        table.add_column("Timestamp", style="yellow")
        table.add_column("Size", style="magenta")

        for cp in checkpoints:
            table.add_row(
                str(cp.session_id)[:8] + "...",
                str(cp.row_index),
                str(cp.stage_index),
                cp.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                f"{cp.size_bytes:,} bytes",
            )

        console.print("\n")
        console.print(table)
        console.print(f"\n[cyan]Total checkpoints: {len(checkpoints)}[/cyan]")

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to input file to inspect",
)
@click.option(
    "--head",
    type=int,
    default=5,
    help="Number of rows to show (default: 5)",
)
def inspect(input: Path, head: int):
    """
    Inspect input data file.

    Shows file info and preview of first N rows.

    Examples:

        # Inspect CSV file
        llm-dataset inspect -i data.csv

        # Show first 10 rows
        llm-dataset inspect -i data.csv --head 10
    """
    try:
        import pandas as pd

        console.print(f"[cyan]Inspecting {input}...[/cyan]")

        # Detect file type
        suffix = input.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(input)
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(input)
        elif suffix == ".parquet":
            df = pd.read_parquet(input)
        else:
            console.print(f"[red]❌ Unsupported file type: {suffix}[/red]")
            sys.exit(1)

        # File info
        info_table = Table(title="File Information")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")

        info_table.add_row("File Path", str(input))
        info_table.add_row("File Type", suffix[1:].upper())
        info_table.add_row("Total Rows", f"{len(df):,}")
        info_table.add_row("Total Columns", str(len(df.columns)))
        info_table.add_row(
            "Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        )

        console.print("\n")
        console.print(info_table)

        # Columns
        console.print("\n[cyan]Columns:[/cyan]")
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            console.print(f"  • {col} ({dtype}) - {null_count} nulls")

        # Preview
        console.print(f"\n[cyan]First {head} rows:[/cyan]")
        console.print(df.head(head).to_string())

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


@cli.command()
def list_providers():
    """
    List all available LLM providers with details.

    Shows supported providers, their platforms, costs, and requirements.

    Examples:

        # List all providers
        ondine list-providers
    """
    try:
        # Create table
        table = Table(title="🪽 Available LLM Providers", show_header=True)
        table.add_column("Provider ID", style="cyan", width=20)
        table.add_column("Name", style="bright_white", width=20)
        table.add_column("Platform", style="yellow", width=25)
        table.add_column("Cost", style="magenta", width=12)
        table.add_column("Use Case", style="white", width=35)

        # Add rows for each provider
        for provider in LLMProvider:
            metadata = PROVIDER_METADATA[provider]

            # Color-code cost
            cost = metadata["cost"]
            if "Free" in cost or cost == "Varies":
                cost_colored = f"[green]{cost}[/green]"
            elif cost == "$$":
                cost_colored = f"[yellow]{cost}[/yellow]"
            else:  # $$$
                cost_colored = f"[red]{cost}[/red]"

            table.add_row(
                f"[bold]{provider.value}[/bold]",
                metadata["name"],
                metadata["platform"],
                cost_colored,
                metadata["use_case"],
            )

        console.print("\n")
        console.print(table)

        # Requirements section
        console.print("\n[cyan]📋 Requirements by Provider:[/cyan]")
        for provider in LLMProvider:
            metadata = PROVIDER_METADATA[provider]
            console.print(
                f"  [bold cyan]{provider.value}[/bold cyan]: {metadata['requirements']}"
            )

        # Usage examples
        console.print("\n[cyan]💡 Usage Examples:[/cyan]")
        console.print("  [dim]# Use OpenAI[/dim]")
        console.print("  ondine process --provider openai --config config.yaml")
        console.print("\n  [dim]# Use local MLX on Apple Silicon[/dim]")
        console.print("  ondine process --provider mlx --config config.yaml")
        console.print("\n  [dim]# Use custom API (Ollama, vLLM, Together.AI)[/dim]")
        console.print(
            "  ondine process --provider openai_compatible --config config.yaml"
        )
        console.print(
            "\n  [dim]💡 Tip: Set provider in your YAML config file or use --provider flag[/dim]\n"
        )

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
