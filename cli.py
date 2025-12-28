"""Court Finder CLI."""

from pathlib import Path
from typing import Optional
import json
import logging

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.pipeline import CourtFinderPipeline, save_geojson
from src.cache import ResultCache

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="court-finder",
    help="Detect tennis courts from NAIP aerial imagery."
)
cache_app = typer.Typer(help="Cache management commands.")
app.add_typer(cache_app, name="cache")

console = Console()


@app.command()
def search(
    zipcode: str = typer.Option(..., "--zipcode", "-z", help="US zipcode to search"),
    radius: float = typer.Option(1.0, "--radius", "-r", help="Search radius in miles"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output GeoJSON file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Skip cache and reprocess"),
    mock: bool = typer.Option(False, "--mock", help="Use mock segmenter (for testing without SAM 3)")
):
    """
    Search for tennis courts in an area.
    
    Example:
        python cli.py search --zipcode 30306 --radius 2 --output courts.geojson
    """
    console.print(f"[bold]🎾 Court Finder[/bold]")
    console.print(f"Searching: {zipcode}, {radius} mile radius")
    
    # Default output path
    if output is None:
        output = Path(f"courts_{zipcode}_{radius}mi.geojson")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            expand=True
        ) as progress:
            # Initialize pipeline
            init_task = progress.add_task("Initializing...", total=None)
            pipeline = CourtFinderPipeline(use_mock_segmenter=mock)
            
            # Search task
            search_task = progress.add_task("Starting search...", total=100)
            progress.remove_task(init_task)
            
            def progress_callback(desc: str, frac: float, current: Optional[int] = None, total: Optional[int] = None):
                progress.update(search_task, description=desc, completed=frac * 100)
                if current is not None and total is not None:
                    # Update description with chip progress if available
                    progress.update(search_task, description=f"{desc} ({current}/{total})")
            
            # Run search
            geojson = pipeline.search(
                zipcode=zipcode,
                radius_miles=radius,
                skip_cache=no_cache,
                progress_callback=progress_callback
            )
        
        # Save results
        save_geojson(geojson, output)
        
        # Display summary
        court_count = geojson["properties"]["court_count"]
        from_cache = geojson["properties"].get("from_cache", False)
        
        console.print()
        console.print(f"[green]✓ Found {court_count} tennis courts[/green]")
        if from_cache:
            console.print("[dim]  (from cache)[/dim]")
        console.print(f"[blue]→ Output: {output}[/blue]")
        console.print()
        console.print("[dim]View results at: https://geojson.io[/dim]")
        
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Search failed")
        raise typer.Exit(1)


@cache_app.command("stats")
def cache_stats():
    """Show cache statistics."""
    cache = ResultCache()
    stats = cache.get_stats()
    
    table = Table(title="Cache Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total entries", str(stats["total_entries"]))
    table.add_row("Unique imagery dates", str(stats["unique_imagery_dates"]))
    table.add_row("Database path", stats["db_path"])
    table.add_row("Database size", f"{stats['db_size_mb']:.2f} MB")
    
    console.print(table)
    
    if stats["recent_dates"]:
        console.print("\n[bold]Recent imagery dates:[/bold]")
        for date_str, count in stats["recent_dates"].items():
            console.print(f"  {date_str}: {count} entries")


@cache_app.command("clear")
def cache_clear(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")
):
    """Clear all cached results."""
    cache = ResultCache()
    stats = cache.get_stats()
    
    if stats["total_entries"] == 0:
        console.print("[dim]Cache is already empty.[/dim]")
        return
    
    if not confirm:
        confirm = typer.confirm(
            f"Clear {stats['total_entries']} cached entries?"
        )
    
    if confirm:
        cache.clear()
        console.print("[green]✓ Cache cleared[/green]")
    else:
        console.print("[dim]Cancelled[/dim]")


@app.command()
def info():
    """Show system information."""
    import torch
    
    table = Table(title="System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    # PyTorch
    table.add_row("PyTorch version", torch.__version__)
    
    # CUDA
    if torch.cuda.is_available():
        table.add_row("CUDA available", "Yes")
        table.add_row("CUDA device", torch.cuda.get_device_name(0))
        table.add_row("CUDA memory", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        table.add_row("CUDA available", "No")
    
    # MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        table.add_row("MPS available", "Yes (Apple Silicon)")
    else:
        table.add_row("MPS available", "No")
    
    # SAM 3
    try:
        import sam3
        table.add_row("SAM 3 installed", "Yes")
    except ImportError:
        table.add_row("SAM 3 installed", "[yellow]No (use --mock for testing)[/yellow]")
    
    console.print(table)


if __name__ == "__main__":
    app()
