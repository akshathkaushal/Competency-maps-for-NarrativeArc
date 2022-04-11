"""Console script for competency_maps."""
import argparse
import os
import sys
from pathlib import Path
from pprint import pprint
from time import time

import pandas as pd
from pyfiglet import Figlet
from PyInquirer import (
    Token,
    ValidationError,
    Validator,
    prompt,
    style_from_dict,
)

from competency_maps import cli_utils
from competency_maps.competency_maps import CompetencyMap
from competency_maps.utils.map_config import CompetencyMapConfig

style = style_from_dict(
    {
        Token.QuestionMark: "#E91E63 bold",
        Token.Selected: "#673AB7 bold",
        Token.Instruction: "",  # default
        Token.Answer: "#2196f3 bold",
        Token.Question: "",
    }
)


class NumberValidator(Validator):
    def validate(self, document):
        try:
            int(document.text)
        except ValueError:
            raise ValidationError(
                message="Please enter a number",
                cursor_position=len(document.text),
            )  # Move cursor to end


class PathValidator(Validator):
    def validate(self, document):
        if not os.path.exists(document.text):
            raise ValidationError(
                message="Please enter a valid path",
                cursor_position=len(document.text),
            )


main_prompt = [
    {
        "type": "expand",
        "name": "operation",
        "message": "Which operation do you want to perform?",
        "choices": [
            {"key": "c", "name": "Create New Map", "value": "create"},
            {
                "key": "a",
                "name": "Add New Resources to existing map",
                "value": "add",
            },
            {"key": "r", "name": "Refresh Competency Maps", "value": "refresh"},
            {"key": "d", "name": "Delete Competency Map", "value": "delete"},
        ],
    }
]

create_map_prompt = [
    {
        "type": "input",
        "name": "map_name",
        "message": "Name of the Map to Create",
    },
    {
        "type": "input",
        "name": "input_path",
        "message": "Path where the learning resources are located",
        "validate": PathValidator,
    },
    {
        "type": "input",
        "name": "field_delimiter",
        "message": "Field Delimiter in the input file",
        "default": ",",
        "when": lambda answers: answers["input_path"].endswith(".csv"),
    },
    {
        "type": "input",
        "name": "content_fields",
        "message": "Field names having the content of the resource. Comma Separated",
        "default": "title,description",
        "when": lambda answers: answers["input_path"].endswith(".csv"),
    },
    {
        "type": "input",
        "name": "output_path",
        "message": "Output Directory where results will be stored",
        "default": lambda answers: answers["input_path"],
    },
    {
        "type": "expand",
        "name": "config_type",
        "message": "Type of Config to Use",
        "choices": [
            {"key": "c", "name": "Custom Config", "value": "custom"},
            {"key": "d", "name": "Default Config", "value": "default"},
            {
                "key": "e",
                "name": "External Config From a File",
                "value": "external",
            },
        ],
        "default": "d",
    },
    {
        "type": "input",
        "name": "config_file",
        "message": "External Map Config File",
        "validate": PathValidator,
        "when": lambda answers: answers["config_type"] == "external",
    },
]

add_map_prompt = [
    {
        "type": "input",
        "name": "map_name",
        "message": "Name of the Existing map",
    },
    {
        "type": "input",
        "name": "input_path",
        "message": "Path where the new learning resources are located",
        "validate": PathValidator,
    },
    {
        "type": "input",
        "name": "field_delimiter",
        "message": "Field Delimiter in the input file",
        "default": ",",
        "when": lambda answers: answers["input_path"].endswith(".csv"),
    },
    {
        "type": "input",
        "name": "content_fields",
        "message": "Field names having the content of the resource. Comma Separated",
        "default": "title,description",
        "when": lambda answers: answers["input_path"].endswith(".csv"),
    },
    {
        "type": "input",
        "name": "output_path",
        "message": "Output Directory where map results are loaded",
        "default": lambda answers: answers["input_path"],
    },
]

map_config_prompt = [
    {
        "type": "rawlist",
        "name": "PREPROCESSOR_TYPE",
        "message": "Preprocessing Pipeline",
        "choices": ["spacy", "nltk"],
    },
    {
        "type": "rawlist",
        "name": "TOPIC_MODEL_TYPE",
        "message": "Topic Model to use",
        "choices": ["lda", "ldamallet", "hierarchical", "top2vec"],
    },
    {
        "type": "confirm",
        "name": "TF_IDF_ENABLED",
        "message": "Apply TF-IDF Filter on Corpus?",
        "default": True,
    },
    {
        "type": "confirm",
        "name": "FILTER_EXTREMES",
        "message": "Filter Extreme Words?",
        "default": True,
    },
    {
        "type": "input",
        "name": "NUM_TOPIC_CLUSTERS",
        "message": "Number of Topic Clusters to evaluate. It could either be a range Eg: 2,10 or a single value",
    },
    {
        "type": "input",
        "name": "NUM_TOPICS",
        "message": "Number of Topics to extract from each cluster",
        "validate": NumberValidator,
        "filter": lambda val: int(val),
    },
    {
        "type": "rawlist",
        "name": "TOPIC_MODEL_EVALUATION_METRIC",
        "message": "Topic Model Evaluation Metric to use",
        "choices": ["cv_score", "umass_score", "log_perplexity"],
    },
    {
        "type": "rawlist",
        "name": "WORD_EMBEDDING_TYPE",
        "message": "Word Embedding Model to use",
        "choices": ["word2vec", "glove", "fasttext"],
        "default": "fasttext",
    },
    {
        "type": "input",
        "name": "WORD_EMBEDDING_DIMENSIONS",
        "message": "Number of Dimensions for Word Embedding Model",
        "default": "300",
        "validate": NumberValidator,
        "filter": lambda val: int(val),
    },
    {
        "type": "rawlist",
        "name": "DOCUMENT_EMBEDDING_TYPE",
        "message": "Document Embedding Model to use",
        "choices": ["doc2vec", "pretrained"],
    },
    {
        "type": "input",
        "name": "DOCUMENT_EMBEDDING_DIMENSIONS",
        "message": "Number of Dimensions for Document Embedding Model",
        "default": "300",
        "validate": NumberValidator,
        "filter": lambda val: int(val),
    },
    {
        "type": "input",
        "name": "PRETRAINED_DOCUMENT_EMBEDDING_PATH",
        "message": "Path of the pretrained model",
        "validate": PathValidator,
        "when": lambda answers: answers["DOCUMENT_EMBEDDING_TYPE"]
        == "pretrained",
    },
    {
        "type": "input",
        "name": "NUM_LEVELS",
        "message": "Number of Levels on Y-Axis",
        "default": "10",
        "validate": NumberValidator,
        "filter": lambda val: int(val),
    },
]


def create_map():
    """Create a New Map."""
    print("Creating a New Competency Map... Specify the details")
    create_parameters = prompt(create_map_prompt, style=style)
    print("Create Map Parameters:")
    pprint(create_parameters)

    if create_parameters["config_type"] == "default":
        map_config = CompetencyMapConfig()
    elif create_parameters["config_type"] == "external":
        map_config = CompetencyMapConfig(create_parameters["config_file"])
    else:
        map_config_parameters = prompt(map_config_prompt, style=style)
        print("Map Config:")
        pprint(map_config_parameters)
        if (
            "PRETRAINED_DOCUMENT_EMBEDDING_PATH"
            not in map_config_parameters.keys()
        ):
            map_config_parameters["PRETRAINED_DOCUMENT_EMBEDDING_PATH"] = None
        map_config = cli_utils.load_map_config(map_config_parameters)

    if "field_delimiter" not in create_parameters.keys():
        create_parameters["field_delimiter"] = None
    if "content_fields" not in create_parameters.keys():
        create_parameters["content_fields"] = None
    else:
        create_parameters["content_fields"] = create_parameters[
            "content_fields"
        ].split(",")

    print(cli_utils.get_map_config(map_config))
    map_path = Path(
        f'{create_parameters["output_path"]}/{create_parameters["map_name"]}'
    )
    preprocessed_texts_path = Path.joinpath(map_path, "preprocessed_texts.obj")
    if os.path.exists(preprocessed_texts_path):
        corpus = None
    else:
        corpus = cli_utils.get_corpus(
            Path(create_parameters["input_path"]),
            create_parameters["field_delimiter"],
        )

    map_object = CompetencyMap(
        create_parameters["map_name"],
        map_config,
        corpus,
        create_parameters["output_path"],
        create_parameters["content_fields"],
        is_debug=False,
    )
    (
        topics,
        resources,
        resource_topic_mapping,
        map_summary,
    ) = map_object.create_map()

    results_directory = Path(
        f"{create_parameters['output_path']}/{create_parameters['map_name']}/cmaps"
    )
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
    return map_summary


def add_resources():
    """Add resources to an existing map"""
    print(
        "Adding Resources to an Existing Competency Map... Specify the details"
    )
    add_parameters = prompt(add_map_prompt, style=style)
    print("Add to Map Parameters:")
    pprint(add_parameters)
    results_directory = Path(
        f"{add_parameters['output_path']}/{add_parameters['map_name']}/cmaps"
    )

    if "field_delimiter" not in add_parameters.keys():
        add_parameters["field_delimiter"] = None
    if "content_fields" not in add_parameters.keys():
        add_parameters["content_fields"] = None
    else:
        add_parameters["content_fields"] = add_parameters[
            "content_fields"
        ].split(",")

    config_path = f"{results_directory}/map_config.json"
    import json

    with open(config_path) as json_file:
        map_config_parameters = json.load(json_file)
        pprint(map_config_parameters)
        if "doc_embedding_path" not in map_config_parameters.keys():
            map_config_parameters["doc_embedding_path"] = None
        map_config = cli_utils.load_map_config(map_config_parameters)

    corpus = cli_utils.get_corpus(
        Path(add_parameters["input_path"]), add_parameters["field_delimiter"]
    )
    map_object = CompetencyMap(
        add_parameters["map_name"],
        map_config,
        corpus,
        add_parameters["output_path"],
        add_parameters["content_fields"],
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


def interactive_main():
    """Starts an interactive command line interface"""
    f = Figlet(font="slant")
    print(f.renderText("Competency Maps"))
    answers = prompt(main_prompt, style=style)
    response = None
    if answers["operation"] == "create":
        response = create_map()
    elif answers["operation"] == "add":
        response = add_resources()
    if response is not None:
        return 0
    else:
        return 1
