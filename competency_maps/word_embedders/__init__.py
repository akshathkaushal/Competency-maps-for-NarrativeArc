from importlib import import_module

from competency_maps.word_embedders.word_embedder import WordEmbedderBaseClass


def grab(embedder_type, *args, **kwargs):
    try:
        if "." in embedder_type:
            module_name, class_name = embedder_type.rsplit(".", 1)
        else:
            module_name = embedder_type
            class_name = embedder_type.capitalize() + "Embedder"
        word_embedder_module = import_module(
            f".{module_name}_word_embedder",
            package="competency_maps.word_embedders",
        )
        word_embedder_class = getattr(word_embedder_module, class_name)
        instance = word_embedder_class(*args, **kwargs)
    except (AttributeError, ModuleNotFoundError):
        raise ImportError(
            f"{embedder_type} is not part of our word embedders collection!"
        )
    else:
        if not issubclass(word_embedder_class, WordEmbedderBaseClass):
            raise ImportError(
                "We currently don't have {}, but you are welcome to send in the request for it!".format(
                    word_embedder_class
                )
            )
    return instance
