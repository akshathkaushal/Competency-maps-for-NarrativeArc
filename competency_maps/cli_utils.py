from competency_maps import data_loaders
from competency_maps.exceptions import exceptions
from competency_maps.utils.map_config import CompetencyMapConfig


def get_corpus(resource_path, delimiter):
    """Read Data From Path and return Dataframe where each row corresponds to a resource

    Args:
        resource_path: Path where the learning resources are located
        delimiter: Delimiter of the fields if the input is a csv file

    Returns:
        An class `pandas.DataFrame` containing the corpus of learning resources

    Raises:
         InvalidInputException: Invalid Input File Specified
    """
    print(f"Input Resources Path: {resource_path}")
    if resource_path.as_posix().endswith(".csv"):
        data_loader_type = data_loaders.grab(
            "csv", path=resource_path, delimiter=delimiter
        )
        corpus = data_loader_type.read_corpus()
    elif resource_path.is_dir():
        data_loader_type = data_loaders.grab("files", path=resource_path)
        corpus = data_loader_type.read_corpus()
    else:
        raise exceptions.InvalidInputException
    return corpus


def load_map_config(map_config_params):
    map_config = CompetencyMapConfig()
    map_config.PREPROCESSOR_TYPE = map_config_params["PREPROCESSOR_TYPE"]
    map_config.TF_IDF_ENABLED = map_config_params["TF_IDF_ENABLED"]
    map_config.FILTER_EXTREMES = map_config_params["FILTER_EXTREMES"]
    map_config.TOPIC_MODEL_TYPE = map_config_params["TOPIC_MODEL_TYPE"]

    topic_clusters = map_config_params["NUM_TOPIC_CLUSTERS"]
    if "," in topic_clusters:
        topic_cluster_range = topic_clusters.split(",", 2)
        map_config.MIN_NUM_TOPIC_CLUSTERS = int(topic_cluster_range[0].strip())
        map_config.MAX_NUM_TOPIC_CLUSTERS = int(topic_cluster_range[1].strip())
    else:
        map_config.MIN_NUM_TOPIC_CLUSTERS = int(topic_clusters.strip())
        map_config.MAX_NUM_TOPIC_CLUSTERS = int(topic_clusters.strip())

    map_config.TOPIC_MODEL_EVALUATION_METRIC = map_config_params[
        "TOPIC_MODEL_EVALUATION_METRIC"
    ]
    map_config.NUM_TOPICS = map_config_params["NUM_TOPICS"]

    map_config.WORD_EMBEDDING_TYPE = map_config_params["WORD_EMBEDDING_TYPE"]
    map_config.WORD_EMBEDDING_DIMENSIONS = map_config_params[
        "WORD_EMBEDDING_DIMENSIONS"
    ]
    map_config.DOCUMENT_EMBEDDING_TYPE = map_config_params[
        "DOCUMENT_EMBEDDING_TYPE"
    ]
    map_config.DOCUMENT_EMBEDDING_DIMENSIONS = map_config_params[
        "DOCUMENT_EMBEDDING_DIMENSIONS"
    ]
    map_config.PRETRAINED_DOCUMENT_EMBEDDING_PATH = map_config_params[
        "PRETRAINED_DOCUMENT_EMBEDDING_PATH"
    ]
    map_config.NUM_LEVELS = map_config_params["NUM_LEVELS"]
    return map_config


def get_map_config(map_config):
    doc_type = (
        "custom"
        if map_config.PRETRAINED_DOCUMENT_EMBEDDING_PATH is None
        else map_config.PRETRAINED_DOCUMENT_EMBEDDING_PATH
    )
    map_config_json = {
        "PREPROCESSOR_TYPE": map_config.PREPROCESSOR_TYPE,
        "TF_IDF_ENABLED": map_config.TF_IDF_ENABLED,
        "FILTER_EXTREMES": map_config.FILTER_EXTREMES,
        "TOPIC_MODEL_TYPE": map_config.TOPIC_MODEL_TYPE,
        "TOPIC_MODEL_EVALUATION_METRIC": map_config.TOPIC_MODEL_EVALUATION_METRIC,
        "NUM_TOPIC_CLUSTERS": f"{map_config.MIN_NUM_TOPIC_CLUSTERS},{map_config.MAX_NUM_TOPIC_CLUSTERS}",
        "NUM_TOPICS": map_config.NUM_TOPICS,
        "WORD_EMBEDDING_TYPE": map_config.WORD_EMBEDDING_TYPE,
        "WORD_EMBEDDING_DIMENSIONS": map_config.WORD_EMBEDDING_DIMENSIONS,
        "DOCUMENT_EMBEDDING_TYPE": map_config.DOCUMENT_EMBEDDING_TYPE,
        "DOCUMENT_EMBEDDING_DIMENSIONS": map_config.DOCUMENT_EMBEDDING_DIMENSIONS,
        "PRETRAINED_DOCUMENT_EMBEDDING_PATH": doc_type,
        "NUM_LEVELS": map_config.NUM_LEVELS,
    }
    return map_config_json
