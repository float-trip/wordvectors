# pip install optuna scikit-learn rich numpy typer floret unidecode markdown bs4 psycopg2-binary

import json
import floret
import subprocess
from collections import Counter
import logging
import os
import sys
from collections import namedtuple
from multiprocessing import Pool, cpu_count
from multiprocessing import Manager

import numpy as np
import optuna
import typer
from rich import print as rprint
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track

from csv_loader import CSVLoader
from floret_model import FloretModel
from text_cleaner import TextCleaner

app = typer.Typer()
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)


Metric = namedtuple("Metric", "weight, score")


def calculate_weighted_score(metrics):
    total_weight = total_score = 0
    for metric in metrics:
        if metric.score is not None:
            total_score += metric.weight * metric.score
            total_weight += metric.weight
    return total_score / total_weight if total_weight else float("-inf")


def objective(trial: optuna.Trial, file_path: str, validation_data: dict) -> float:
    model = FloretModel.create_from_trial(trial)
    try:
        model.train(file_path)
    except:
        return float("-inf")

    evaluation_results = model.evaluate(validation_data)

    similarity_score = (
        evaluation_results["similarity"] if "similarity" in evaluation_results else None
    )

    metrics = [
        Metric(weight=0.50, score=similarity_score),
        Metric(weight=0.25, score=evaluation_results["clusters"][0]),  # ARI
        Metric(weight=0.25, score=evaluation_results["clusters"][1]),  # NMI
        # Metric(weight=0.15, score=evaluation_results["analogy"]),  # Analogy
    ]

    rprint(metrics)

    return calculate_weighted_score(metrics)


def validate_input_file(file_path: str):
    if not os.path.exists(file_path):
        raise typer.Exit(f"Input file {file_path} does not exist")

def worker(thread):
    cleaner = TextCleaner()
    tokens = cleaner.tokenize(thread)
    token_frequency = Counter(tokens)
    return (" ".join(tokens) + "\n", token_frequency)

@app.command(
    help="Tokenize posts and comments from CSV files and output to a text file."
)
def tokenize(
    posts_csv: str = typer.Argument("posts.csv", help="Path to the posts CSV file"),
    comments_csv: str = typer.Argument("comments.csv", help="Path to the comments CSV file"),
    output_file: str = typer.Option(
        "training_data.txt", help="Output file for the tokenized data"
    ),
    frequency_output_file: str = typer.Option(
        "frequencies.json", help="Output file for the token frequency data"
    ),
    num_workers: int = typer.Option(
        cpu_count(), help="Number of workers for multiprocessing"
    ),
):
    validate_input_file(posts_csv)
    validate_input_file(comments_csv)

    loader = CSVLoader(skip_ids=[16049, 261, 1703, 10953])
    formatted_threads = loader.parse(posts_csv, comments_csv)

    with Pool(num_workers) as pool:
        results = pool.map(worker, formatted_threads)

    combined_token_frequency = Counter()
    with open(output_file, "w", encoding="utf-8") as f:
        for result, token_frequency in track(results, description="Tokenizing"):
            f.write(result)
            combined_token_frequency.update(token_frequency)

    with open(frequency_output_file, "w", encoding="utf-8") as freq_file:
        json.dump(dict(combined_token_frequency), freq_file, ensure_ascii=False, indent=4)

    logging.info(f"Tokenization complete. Output saved to {output_file}")
    logging.info(f"Token frequency data saved to {frequency_output_file}")

@app.command()
def optimize(
    train_path: str = typer.Argument("training_data.txt", help="Path to the training data file"),
    validation_path: str = typer.Argument("validation.json", help="Path to the validation data file"),
    study_name: str = typer.Option(
        "wv", help="Name of the Optuna study"
    ),
    db_path: str = typer.Option(
        "postgresql://",
        help="Database file for Optuna study",
    ),
):
    with open(validation_path, "r") as f:
        validation_data = json.load(f)

    if train_path is None or validation_data is None:
        raise ValueError("Training or validation data could not be loaded.")

    study = optuna.create_study(
        study_name=study_name, storage=db_path, load_if_exists=True, direction="maximize"
    )

    def objective_wrapper(trial):
        return objective(trial, train_path, validation_data)

    try:
        study.optimize(objective_wrapper, n_trials=None, timeout=None)
    except KeyboardInterrupt:
        rprint("[bold red]Interrupted, study state saved.[/bold red]")


@app.command()
def retrain(
    train_path: str = typer.Argument("training_data.txt", help="Path to the training data file"),
    output_stem: str = typer.Argument(
        "vectors", help="Output stem for the .vec and .magnitude files"
    ),
    frequency_path: str = typer.Option(
        "frequencies.json", help="Path to the token frequency JSON file"
    ),
    study_name: str = typer.Option(
        "wv", help="Name of the Optuna study"
    ),
    frequency_limit: int = typer.Option(
        20, help="Frequency limit for pruning tokens"
    ),
    db_path: str = typer.Option(
        "postgresql://",
        help="Database file for Optuna study",
    ),
):
    study = optuna.load_study(study_name=study_name, storage=db_path)
    model = FloretModel.create_from_trial(study.best_trial)

    if True or model.train(train_path):
        model_path = f"{output_stem}.bin"
        model.save(model_path)
        rprint(f"[bold yellow]Model retrained and saved to {model_path}[/bold yellow]")

        with open(frequency_path, 'r') as freq_file:
            frequencies = json.load(freq_file)

        pruned_vec_path = model.prune_and_save_vectors(output_stem, frequencies, frequency_limit)
        rprint(f"[bold yellow]Pruned vectors saved to {pruned_vec_path}[/bold yellow]")

        vectors_tsv_path = f"{output_stem}_vectors.tsv"
        metadata_tsv_path = f"{output_stem}_metadata.tsv"
        model.convert_vec_to_tsv(pruned_vec_path, vectors_tsv_path, metadata_tsv_path)
        rprint(f"[bold yellow]Created TSV files.[/bold yellow]")

        rprint("[bold yellow]Converting .vec to .magnitude...[/bold yellow]")
        magnitude_path = f"{output_stem}.magnitude"
        if model.convert_vec_to_magnitude(pruned_vec_path, magnitude_path):
            rprint(f"[bold green]Magnitude file created at {magnitude_path}[/bold green]")
        else:
            rprint("[bold red]Failed to convert .vec to .magnitude.[/bold red]")
    else:
        rprint("[bold red]Failed to retrain model.[/bold red]")


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        rprint("\n[bold red]Interrupted. Exiting now...[/bold red]")
        sys.exit(0)

