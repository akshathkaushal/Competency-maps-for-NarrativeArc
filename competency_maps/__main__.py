# type: ignore[attr-defined]

import os
import random
from pathlib import Path
from time import time

import pandas as pd
import typer
from rich.console import Console

from competency_maps import __version__, cli_utils
from competency_maps.competency_maps import CompetencyMap
from competency_maps.utils.map_config import CompetencyMapConfig

app = typer.Typer(
    name="competency-maps",
    help="competency_maps is a python cli/package that is used to create a competency map from a learning corpus",
    add_completion=False,
)
console = Console()


def version_callback(value: bool):
    """Prints the version of the package."""
    if value:
        console.print(
            f"[yellow]test-template[/] version: [bold blue]{__version__}[/]"
        )
        raise typer.Exit()


@app.command()
def create(
    input: str = typer.Option(
        ..., help="Path Containing the Learning Resources"
    ),
    output: str = typer.Option(
        ..., help="Output Location where the results will be stored"
    ),
    map_name: str = typer.Option(..., help="Name of the Map"),
    delimiter: str = typer.Option(
        ",", help="Field Seperator if input is csv file"
    ),
    fields: str = typer.Option(
        None,
        help="Fields(seperated by ,) containing the resources if input is csv.",
    ),
    config: str = typer.Option(
        None, help="Path Containing the Config File for other parameters"
    ),
):
    output_folder = output
    if config is not None:
        console.log(f"Loading from file {config}...")
        map_config = CompetencyMapConfig(config)
    else:
        map_config = CompetencyMapConfig()
        console.log("Loading default config...")
    console.log(map_config)
    console.log(cli_utils.get_map_config(map_config))

    doc_type = (
        "custom"
        if map_config.PRETRAINED_DOCUMENT_EMBEDDING_PATH is None
        else "pretrained"
    )
    map_path = Path(f"{output_folder}/{map_name}")
    preprocessed_texts_path = Path.joinpath(map_path, "preprocessed_texts.obj")
    if os.path.exists(preprocessed_texts_path):
        corpus = pd.read_pickle(preprocessed_texts_path)
    else:
        corpus = cli_utils.get_corpus(Path(input), delimiter)
    test = CompetencyMap(
        map_name, map_config, corpus, output_folder, fields, is_debug=False
    )
    topics, resources, resource_topic_mapping, map_summary = test.create_map()

    # Write Results to Files
    results_directory = Path(f"{output_folder}/{map_name}/cmaps/")

    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    map_summary["results"] = results_directory.as_posix()

    # Write Results to files
    topics.to_csv(Path.joinpath(results_directory, "topics.csv"), index=False)
    required_cols = [
        "resource_id",
        "resource_volume",
        "topic_name",
        "topic_type",
        "topic_volume",
        "document_mapped_probability",
    ]
    resource_topic_mapping[required_cols].drop_duplicates().to_csv(
        Path.joinpath(results_directory, "resource_mapping.csv"), index=False
    )
    resource_topic_mapping.drop_duplicates().to_csv(
        Path.joinpath(results_directory, "resource_mapping_debug.csv"),
        index=False,
    )
    resources.to_csv(
        Path.joinpath(results_directory, "resources.csv"), index=False
    )
    import json

    with open(f"{results_directory}/map_details.json", "w") as f:
        json.dump(map_summary, f)

    with open(f"{results_directory}/map_config.json", "w") as f:
        json.dump(cli_utils.get_map_config(map_config), f)

    return 0


@app.command()
def add_resource(
    map_name: str = typer.Option(..., help="Name of the Existing Map"),
    output: str = typer.Option(
        ..., help="Output Location where the results are stored"
    ),
    input: str = typer.Option(
        ..., help="Path Containing the New Learning Resources"
    ),
    delimiter: str = typer.Option(
        ",", help="Field Seperator if input is csv file"
    ),
    fields: str = typer.Option(
        None,
        help="Fields(seperated by ,) containing the resources if input is csv.",
    ),
):
    console.log("Adding Resources to an Existing Competency Map...")
    results_directory = Path(f"{output}/{map_name}/cmaps")
    config_path = f"{results_directory}/map_config.json"

    import json

    with open(config_path) as json_file:
        map_config_parameters = json.load(json_file)
        console.log(f"Map Parameters: \n {map_config_parameters}")
        if "doc_embedding_path" not in map_config_parameters.keys():
            map_config_parameters["doc_embedding_path"] = None
        map_config = cli_utils.load_map_config(map_config_parameters)

    corpus = cli_utils.get_corpus(Path(input), delimiter)
    map_object = CompetencyMap(
        map_name,
        map_config,
        corpus,
        output,
        content_fields=fields,
        is_debug=False,
    )
    topics_df = pd.read_csv(f"{results_directory}/topics.csv").drop(
        ["X"], axis=1
    )
    resources, resource_topic_mapping, map_summary = map_object.add_resource(
        topics_df
    )
    add_time = int(time() * 1000)
    # Write Results to files
    required_cols = [
        "resource_id",
        "resource_volume",
        "topic_name",
        "topic_type",
        "topic_volume",
        "document_mapped_probability",
    ]
    resource_topic_mapping[required_cols].drop_duplicates().to_csv(
        Path.joinpath(
            results_directory, f"resource_mapping_add_{add_time}.csv"
        ),
        index=False,
    )
    resource_topic_mapping.drop_duplicates().to_csv(
        Path.joinpath(
            results_directory, f"resource_mapping_debug_add_{add_time}.csv"
        ),
        index=False,
    )
    resources.to_csv(
        Path.joinpath(results_directory, f"resources_add_{add_time}.csv"),
        index=False,
    )
    import json

    with open(f"{results_directory}/map_details_add_{add_time}.json", "w") as f:
        json.dump(map_summary, f)
    return map_summary


@app.command()
def refresh():
    pass
