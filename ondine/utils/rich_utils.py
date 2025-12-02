"""
Rich console utilities for beautiful terminal output.

Centralized Rich integration for consistent styling across Ondine.
Provides reusable display functions for tables, panels, progress, and logging.
"""

from typing import Any

# Lazy imports to avoid Rich dependency at module level
_console = None


def get_console():
    """Get shared Rich console instance (lazy initialization)."""
    global _console
    if _console is None:
        try:
            from rich.console import Console

            _console = Console()
        except ImportError:
            _console = None
    return _console


def is_rich_available() -> bool:
    """Check if Rich library is available."""
    try:
        import rich  # noqa: F401

        return True
    except ImportError:
        return False


# =============================================================================
# Color Theme - Consistent across all Ondine output
# =============================================================================

THEME = {
    # Status colors
    "success": "bold green",
    "error": "bold red",
    "warning": "bold yellow",
    "info": "bold cyan",
    "dim": "dim white",
    # Data colors
    "label": "cyan",
    "value": "green",
    "highlight": "magenta",
    "number": "yellow",
    # Table styles
    "header": "bold cyan",
    "row_odd": "white",
    "row_even": "dim white",
}


# =============================================================================
# Table Utilities
# =============================================================================


def create_table(
    title: str,
    columns: list[tuple[str, dict[str, Any] | None]],
    rows: list[list[str]],
    show_header: bool = True,
    box_style: str = "rounded",
) -> Any:
    """
    Create a Rich table with consistent styling.

    Args:
        title: Table title
        columns: List of (name, style_kwargs) tuples
        rows: List of row data (list of strings)
        show_header: Show column headers
        box_style: Box style ("rounded", "simple", "heavy", etc.)

    Returns:
        Rich Table object (or None if Rich unavailable)

    Example:
        table = create_table(
            "Results",
            [("ID", {"style": "dim"}), ("Name", {"style": "cyan"})],
            [["1", "Alice"], ["2", "Bob"]]
        )
    """
    if not is_rich_available():
        return None

    from rich import box
    from rich.table import Table

    box_map = {
        "rounded": box.ROUNDED,
        "simple": box.SIMPLE,
        "heavy": box.HEAVY,
        "double": box.DOUBLE,
        "minimal": box.MINIMAL,
        "none": None,
    }

    table = Table(
        title=title,
        show_header=show_header,
        header_style=THEME["header"],
        box=box_map.get(box_style, box.ROUNDED),
    )

    for col_name, col_kwargs in columns:
        kwargs = col_kwargs or {}
        table.add_column(col_name, **kwargs)

    for row in rows:
        table.add_row(*row)

    return table


def print_table(
    title: str,
    columns: list[tuple[str, dict[str, Any] | None]],
    rows: list[list[str]],
    **kwargs: Any,
) -> None:
    """Create and print a table (convenience function)."""
    console = get_console()
    if console:
        table = create_table(title, columns, rows, **kwargs)
        if table:
            console.print(table)
    else:
        # Fallback: plain text
        print(f"\n{title}")
        print("-" * len(title))
        for row in rows:
            print("  ".join(row))


# =============================================================================
# Router Display
# =============================================================================


def display_router_deployments(
    model_name: str,
    strategy: str,
    deployments: list[dict[str, Any]],
    verbose: bool = False,
) -> None:
    """
    Display Router deployment information in a clean format.

    Args:
        model_name: The shared model name for all deployments
        strategy: Routing strategy (e.g., "usage-based-routing-v2")
        deployments: List of deployment configs from model_list
        verbose: Show detailed table (True) or summary only (False)
    """
    console = get_console()

    # Summary line (always shown)
    summary = f"Router: {len(deployments)} deployments for '{model_name}' | Strategy: {strategy}"

    if console:
        console.print(f"[{THEME['info']}]{summary}[/{THEME['info']}]")

        if verbose:
            # Detailed table
            from rich.table import Table

            table = Table(
                title=None,
                show_header=True,
                header_style=THEME["header"],
                padding=(0, 1),
            )
            table.add_column("#", style="dim", width=4)
            table.add_column("Model ID", style=THEME["highlight"])
            table.add_column("Model", style=THEME["value"])
            table.add_column("RPM", justify="right", style=THEME["number"])
            table.add_column("TPM", justify="right", style=THEME["number"])

            for idx, deployment in enumerate(deployments, 1):
                model_id = deployment.get("model_id", f"deployment-{idx}")
                actual_model = deployment.get("litellm_params", {}).get(
                    "model", "unknown"
                )
                rpm = str(deployment.get("rpm", "∞"))
                tpm = str(deployment.get("tpm", "∞"))
                table.add_row(str(idx), model_id, actual_model, rpm, tpm)

            console.print(table)
    else:
        # Fallback: plain text
        print(summary)
        if verbose:
            for idx, deployment in enumerate(deployments, 1):
                model_id = deployment.get("model_id", f"deployment-{idx}")
                actual_model = deployment.get("litellm_params", {}).get(
                    "model", "unknown"
                )
                rpm = deployment.get("rpm", "∞")
                tpm = deployment.get("tpm", "∞")
                print(f"   [{idx}] {model_id}: {actual_model} (RPM: {rpm}, TPM: {tpm})")


# =============================================================================
# Pipeline Result Display
# =============================================================================


def display_pipeline_summary(
    total_rows: int,
    processed_rows: int,
    failed_rows: int,
    duration: float,
    total_cost: float,
    model: str | None = None,
) -> None:
    """
    Display pipeline execution summary.

    Args:
        total_rows: Total rows in dataset
        processed_rows: Successfully processed rows
        failed_rows: Failed rows
        duration: Execution duration in seconds
        total_cost: Total cost in USD
        model: Model name (optional)
    """
    console = get_console()

    if console:
        from rich.panel import Panel
        from rich.table import Table

        # Create summary table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style=THEME["label"])
        table.add_column("Value", style=THEME["value"])

        if model:
            table.add_row("Model", model)
        table.add_row("Total Rows", f"{total_rows:,}")
        table.add_row("Processed", f"{processed_rows:,}")

        # Color-code failed rows
        failed_style = THEME["error"] if failed_rows > 0 else THEME["value"]
        table.add_row("Failed", f"[{failed_style}]{failed_rows:,}[/{failed_style}]")

        table.add_row("Duration", f"{duration:.2f}s")
        table.add_row("Total Cost", f"${total_cost:.4f}")

        if processed_rows > 0:
            cost_per_row = total_cost / processed_rows
            table.add_row("Cost/Row", f"${cost_per_row:.6f}")
            rows_per_sec = processed_rows / duration if duration > 0 else 0
            table.add_row("Throughput", f"{rows_per_sec:.1f} rows/sec")

        # Success rate
        success_rate = (processed_rows / total_rows * 100) if total_rows > 0 else 0
        rate_style = (
            THEME["success"]
            if success_rate >= 95
            else THEME["warning"]
            if success_rate >= 80
            else THEME["error"]
        )
        table.add_row(
            "Success Rate", f"[{rate_style}]{success_rate:.1f}%[/{rate_style}]"
        )

        panel = Panel(
            table,
            title="[bold green]Pipeline Complete[/bold green]",
            border_style="green",
        )
        console.print(panel)
    else:
        # Fallback
        print("\nPipeline Complete")
        print(f"   Rows: {processed_rows:,}/{total_rows:,}")
        print(f"   Failed: {failed_rows:,}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Cost: ${total_cost:.4f}")


def display_sample_results(
    results: list[dict[str, Any]],
    columns: list[str] | None = None,
    max_rows: int = 5,
    title: str = "Sample Results",
) -> None:
    """
    Display sample results in a table.

    Args:
        results: List of result dictionaries
        columns: Columns to display (None = auto-detect)
        max_rows: Maximum rows to show
        title: Table title
    """
    if not results:
        return

    console = get_console()

    # Auto-detect columns if not specified
    if columns is None:
        columns = list(results[0].keys())[:5]  # First 5 columns

    if console:
        from rich.table import Table

        table = Table(title=title, show_header=True, header_style=THEME["header"])

        for col in columns:
            table.add_column(
                col, style=THEME["value"], overflow="ellipsis", max_width=40
            )

        for row in results[:max_rows]:
            row_values = []
            for col in columns:
                val = row.get(col, "")
                # Truncate long values
                str_val = str(val)[:50] + ("..." if len(str(val)) > 50 else "")
                row_values.append(str_val)
            table.add_row(*row_values)

        console.print(table)

        if len(results) > max_rows:
            console.print(f"[dim]... and {len(results) - max_rows} more rows[/dim]")
    else:
        # Fallback
        print(f"\n{title}")
        for row in results[:max_rows]:
            print(row)


# =============================================================================
# Status Messages
# =============================================================================


def print_success(message: str) -> None:
    """Print success message with green checkmark."""
    console = get_console()
    if console:
        console.print(f"[{THEME['success']}]{message}[/{THEME['success']}]")
    else:
        print(f"{message}")


def print_error(message: str) -> None:
    """Print error message with red X."""
    console = get_console()
    if console:
        console.print(f"[{THEME['error']}]❌ {message}[/{THEME['error']}]")
    else:
        print(f"❌ {message}")


def print_warning(message: str) -> None:
    """Print warning message with yellow triangle."""
    console = get_console()
    if console:
        console.print(f"[{THEME['warning']}]WARNING: {message}[/{THEME['warning']}]")
    else:
        print(f"WARNING: {message}")


def print_info(message: str) -> None:
    """Print info message with cyan color."""
    console = get_console()
    if console:
        console.print(f"[{THEME['info']}]{message}[/{THEME['info']}]")
    else:
        print(f"INFO: {message}")


def print_step(step: int, total: int, message: str) -> None:
    """Print a numbered step (e.g., "[1/3] Loading data...")."""
    console = get_console()
    prefix = f"[{step}/{total}]"
    if console:
        console.print(f"[{THEME['info']}]{prefix}[/{THEME['info']}] {message}")
    else:
        print(f"{prefix} {message}")


# =============================================================================
# LLM Invocation Display
# =============================================================================


def display_llm_invocation_start(
    total_rows: int,
    batch_count: int,
    concurrency: int,
    model: str,
    router_deployments: int | None = None,
) -> None:
    """
    Display LLM invocation start info.

    Args:
        total_rows: Total rows to process
        batch_count: Number of API calls (batches)
        concurrency: Concurrent calls
        model: Model name
        router_deployments: Number of Router deployments (None if no Router)
    """
    console = get_console()

    router_info = (
        f" [Router: {router_deployments} deployments]" if router_deployments else ""
    )
    message = (
        f"Processing {total_rows:,} rows in {batch_count} API calls "
        f"({concurrency} concurrent) | Model: {model}{router_info}"
    )

    if console:
        console.print(f"[{THEME['info']}]{message}[/{THEME['info']}]")
    else:
        print(message)


# =============================================================================
# Cost Display
# =============================================================================


def display_cost_estimate(
    total_cost: float,
    total_tokens: int,
    input_tokens: int,
    output_tokens: int,
    rows: int,
    model: str | None = None,
) -> None:
    """Display cost estimate in a formatted table."""
    console = get_console()

    if console:
        from rich.table import Table

        table = Table(
            title="💰 Cost Estimate", show_header=True, header_style=THEME["header"]
        )
        table.add_column("Metric", style=THEME["label"])
        table.add_column("Value", style=THEME["value"], justify="right")

        if model:
            table.add_row("Model", model)
        table.add_row("Total Cost", f"${total_cost:.4f}")
        table.add_row("Total Tokens", f"{total_tokens:,}")
        table.add_row("Input Tokens", f"{input_tokens:,}")
        table.add_row("Output Tokens", f"{output_tokens:,}")
        table.add_row("Rows", f"{rows:,}")

        if rows > 0:
            table.add_row("Cost/Row", f"${total_cost / rows:.6f}")

        console.print(table)
    else:
        print("\n💰 Cost Estimate")
        print(f"   Total: ${total_cost:.4f}")
        print(f"   Tokens: {total_tokens:,}")
        print(f"   Rows: {rows:,}")
