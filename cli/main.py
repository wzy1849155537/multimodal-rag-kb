"""CLI entry point: rag-kb command with subcommands."""

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="rag-kb",
    help="Multi-modal RAG Knowledge Base Q&A System",
    add_completion=False,
)


@app.command("ingest")
def ingest(
    path: str = typer.Argument(..., help="File or directory path to ingest"),
    config_dir: str = typer.Option(
        "./config", "--config", "-c", help="Config directory path"
    ),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", help="Recursively ingest directory"
    ),
):
    """Ingest documents into the knowledge base."""
    from src.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    target = Path(path).resolve()

    if not target.exists():
        typer.echo(f"Error: Path not found: {target}", err=True)
        raise typer.Exit(code=1)

    if target.is_file():
        typer.echo(f"Ingesting file: {target.name}")
        count = pipeline.ingest_file(target)
        typer.echo(f"Indexed {count} chunks")
    elif target.is_dir():
        typer.echo(f"Ingesting directory: {target}")
        count = pipeline.ingest_directory(target)
        typer.echo(f"Total: {count} chunks indexed")
        stats = pipeline.get_stats()
        typer.echo(f"Index stats: {stats}")
    else:
        typer.echo(f"Error: {target} is neither a file nor directory", err=True)
        raise typer.Exit(code=1)


@app.command("query")
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results to retrieve"),
    config_dir: str = typer.Option(
        "./config", "--config", "-c", help="Config directory path"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show sources"),
):
    """Query the knowledge base."""
    from src.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    answer = pipeline.query(question, top_k=top_k)

    typer.echo(f"\n{'='*60}")
    typer.echo(f"  Q: {question}")
    typer.echo(f"{'='*60}")
    typer.echo(f"\n{answer.answer}\n")
    typer.echo(f"{'='*60}")
    typer.echo(f"  Latency: {answer.latency_ms:.0f}ms | "
                f"Confidence: {answer.confidence:.2f}")

    if verbose and answer.sources:
        typer.echo(f"\n{'─'*60}")
        typer.echo("  Sources:")
        for i, src in enumerate(answer.sources, 1):
            typer.echo(f"  [{i}] {src['doc_name']} (score: {src['score']:.3f})")
            typer.echo(f"      {src['content_snippet'][:120]}...")


@app.command("stats")
def stats(
    config_dir: str = typer.Option(
        "./config", "--config", "-c", help="Config directory path"
    ),
):
    """Show index statistics."""
    from src.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    s = pipeline.get_stats()
    typer.echo(f"Collection: {s.get('collection_name', 'N/A')}")
    typer.echo(f"Total chunks: {s.get('total_chunks', 0)}")
    typer.echo(f"Storage: {s.get('persist_directory', 'N/A')}")


@app.command("serve")
def serve(
    config_dir: str = typer.Option(
        "./config", "--config", "-c", help="Config directory path"
    ),
    port: int = typer.Option(8501, "--port", "-p", help="Web UI port"),
    host: str = typer.Option("localhost", "--host", help="Web UI host"),
):
    """Launch the Streamlit web UI."""
    import subprocess
    import sys

    web_app = Path(__file__).parent.parent / "web" / "app.py"
    if not web_app.exists():
        typer.echo(f"Error: Web app not found at {web_app}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Starting Streamlit UI at http://{host}:{port}")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(web_app),
        "--server.port", str(port),
        "--server.address", host,
    ])


if __name__ == "__main__":
    app()
