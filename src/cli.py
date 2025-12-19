"""Rich CLI for Churn Radar with beautiful terminal output."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

try:
    import click
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table
    from rich.tree import Tree
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    click = None

if RICH_AVAILABLE:
    console = Console()
else:
    console = None


def print_banner():
    """Print application banner."""
    if not RICH_AVAILABLE:
        print("=" * 50)
        print("Churn Radar - ML Platform")
        print("=" * 50)
        return

    banner = """
    [bold cyan]Churn Radar[/bold cyan]
    [dim]ML Platform for Customer Churn Prediction[/dim]
    """
    console.print(Panel(banner, border_style="cyan"))


def print_table(title: str, data: dict, columns: list = None):
    """Print data as a formatted table."""
    if not RICH_AVAILABLE:
        print(f"\n{title}")
        print("-" * 40)
        for k, v in data.items():
            print(f"  {k}: {v}")
        return
    
    table = Table(title=title, show_header=True, header_style="bold magenta")
    
    if columns:
        for col in columns:
            table.add_column(col)
    else:
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
    
    if isinstance(data, dict):
        for k, v in data.items():
            table.add_row(str(k), str(v))
    elif isinstance(data, list):
        for row in data:
            if isinstance(row, (list, tuple)):
                table.add_row(*[str(x) for x in row])
            else:
                table.add_row(str(row))
    
    console.print(table)


def print_success(message: str):
    """Print success message."""
    if RICH_AVAILABLE:
        console.print(f"[bold green][OK][/bold green] {message}")
    else:
        print(f"[OK] {message}")


def print_error(message: str):
    """Print error message."""
    if RICH_AVAILABLE:
        console.print(f"[bold red][FAIL][/bold red] {message}")
    else:
        print(f"[FAIL] {message}", file=sys.stderr)


def print_warning(message: str):
    """Print warning message."""
    if RICH_AVAILABLE:
        console.print(f"[bold yellow][WARN][/bold yellow] {message}")
    else:
        print(f"[WARN] {message}")


def print_info(message: str):
    """Print info message."""
    if RICH_AVAILABLE:
        console.print(f"[bold blue][INFO][/bold blue] {message}")
    else:
        print(f"[INFO] {message}")


class ProgressBar:
    """Context manager for progress bar."""
    
    def __init__(self, description: str = "Processing...", total: int = 100):
        self.description = description
        self.total = total
        self._progress = None
        self._task = None
    
    def __enter__(self):
        if RICH_AVAILABLE:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            )
            self._progress.start()
            self._task = self._progress.add_task(self.description, total=self.total)
        else:
            print(f"{self.description}...")
        return self
    
    def __exit__(self, *args):
        if self._progress:
            self._progress.stop()
    
    def update(self, advance: int = 1, description: str = None):
        if self._progress and self._task is not None:
            if description:
                self._progress.update(self._task, description=description)
            self._progress.update(self._task, advance=advance)


if RICH_AVAILABLE:

    @click.group()
    @click.version_option(version="1.0.0")
    def cli():
        """Churn Radar CLI - ML Platform for Customer Churn Prediction"""
        pass
    
    @cli.command()
    @click.option("--demo", is_flag=True, help="Generate demo data")
    @click.option("--real", is_flag=True, help="Process real data")
    @click.option("--auto", is_flag=True, help="Auto-detect data source")
    @click.option("--n-users", default=1000, help="Number of users for demo")
    @click.option("--sample", type=float, help="Sample fraction for real data")
    def data(demo: bool, real: bool, auto: bool, n_users: int, sample: float):
        """Generate or process dataset."""
        print_banner()
        
        from src.data.make_dataset import generate_demo_data, process_real_data
        
        with ProgressBar("Preparing data") as pbar:
            if demo:
                print_info("Generating demo data...")
                result = generate_demo_data(n_users=n_users)
                pbar.update(100)
            elif real:
                print_info("Processing real data...")
                result = process_real_data(sample_frac=sample)
                pbar.update(100)
            elif auto:
                raw_path = Path("data/raw")
                if raw_path.exists() and list(raw_path.glob("*.csv")):
                    print_info("Found real data, processing...")
                    result = process_real_data(sample_frac=sample)
                else:
                    print_info("No real data found, generating demo...")
                    result = generate_demo_data(n_users=n_users)
                pbar.update(100)
            else:
                print_error("Specify --demo, --real, or --auto")
                return
        
        print_success("Data prepared successfully!")
        print_table("Dataset Summary", {
            "Train size": result.get("train_size", "N/A"),
            "Test size": result.get("test_size", "N/A"),
            "Train path": result.get("train_path", "N/A"),
            "Test path": result.get("test_path", "N/A"),
        })
    
    @cli.command()
    @click.option("--config", default="configs/config.yaml", help="Config file path")
    @click.option("--mlflow/--no-mlflow", default=True, help="Use MLflow tracking")
    def train(config: str, mlflow: bool):
        """Train churn prediction model."""
        print_banner()
        
        from src.models.train import train_model
        
        print_info(f"Training with config: {config}")
        
        with ProgressBar("Training model") as pbar:
            pbar.update(10, "Loading data...")
            result = train_model(config_path=config, use_mlflow=mlflow)
            pbar.update(90)
        
        print_success("Training complete!")
        print_table("Training Results", {
            "Algorithm": result.algorithm,
            "ROC-AUC": f"{result.metrics.get('val_roc_auc', result.metrics.get('train_roc_auc', 0)):.4f}",
            "F1 Score": f"{result.metrics.get('val_f1', result.metrics.get('train_f1', 0)):.4f}",
            "Training Time": f"{result.training_time_sec:.1f}s",
        })
    
    @cli.command()
    @click.option("--algorithm", default="lightgbm", help="Algorithm to optimize")
    @click.option("--n-trials", default=50, help="Number of trials")
    @click.option("--timeout", default=None, type=int, help="Timeout in seconds")
    def hpo(algorithm: str, n_trials: int, timeout: int):
        """Run hyperparameter optimization."""
        print_banner()
        
        import pandas as pd
        from src.models.hpo import optimize_hyperparameters
        from src.features.build_features import split_xy
        
        print_info(f"Running HPO for {algorithm} with {n_trials} trials...")
        
        train_df = pd.read_csv("data/train.csv")
        X, y = split_xy(train_df)
        
        with ProgressBar("Optimizing", total=n_trials) as pbar:
            result = optimize_hyperparameters(
                X, y,
                algorithm=algorithm,
                n_trials=n_trials,
                timeout=timeout,
            )
            pbar.update(n_trials)
        
        print_success("HPO complete!")
        print_table("Best Parameters", result.best_params)
        console.print(f"\n[bold]Best Score:[/bold] {result.best_score:.4f}")
    
    @cli.command()
    @click.option("--host", default="0.0.0.0", help="API host")
    @click.option("--port", default=8000, help="API port")
    @click.option("--reload", is_flag=True, help="Enable auto-reload")
    def serve(host: str, port: int, reload: bool):
        """Start API server."""
        print_banner()
        
        import uvicorn
        
        print_info(f"Starting API server at http://{host}:{port}")
        print_info("Swagger UI: http://localhost:8000/docs")
        
        uvicorn.run(
            "src.api.app:app",
            host=host,
            port=port,
            reload=reload,
        )
    
    @cli.command()
    def status():
        """Show system status."""
        print_banner()
        
        from pathlib import Path
        
        # Check data
        train_exists = Path("data/train.csv").exists()
        test_exists = Path("data/test.csv").exists()
        raw_exists = Path("data/raw").exists() and list(Path("data/raw").glob("*.csv"))
        
        # Check model
        model_exists = Path("models/model.pkl").exists()
        
        status_data = {
            "Training data": "yes" if train_exists else "no",
            "Test data": "yes" if test_exists else "no",
            "Raw data": "yes" if raw_exists else "no",
            "Trained model": "yes" if model_exists else "no",
        }
        
        print_table("System Status", status_data)
        
        # Quick start suggestions
        if not train_exists:
            print_warning("No training data. Run: churn data --demo")
        elif not model_exists:
            print_warning("No trained model. Run: churn train")
        else:
            print_success("System ready! Run: churn serve")
    
    @cli.command()
    def validate():
        """Validate data quality."""
        print_banner()
        
        import pandas as pd
        from src.data.validation import validate_features
        
        train_path = Path("data/train.csv")
        if not train_path.exists():
            print_error("No training data found")
            return
        
        df = pd.read_csv(train_path)
        result = validate_features(df)
        
        if result.is_valid:
            print_success("Data validation passed!")
        else:
            print_error("Data validation failed!")
            for error in result.errors:
                print_error(f"  {error}")
        
        for warning in result.warnings:
            print_warning(warning)
        
        print_table("Data Statistics", result.stats)


def main():
    """CLI entry point."""
    if not RICH_AVAILABLE:
        print("Rich CLI requires: pip install rich click")
        print("Falling back to basic CLI...")
        return
    
    cli()


if __name__ == "__main__":
    main()
