"""Module to Obtain Document Embedding Model"""
from importlib import import_module

from competency_maps.document_embedders.document_embedder import (
    DocumentEmbedderBaseClass,
)


def grab(embedder_type, *args, **kwargs):
    """ Identify the apprpriate type of Document Embedder Object to be used

    Args:
        embedder_type (str) : Type of Document Embedding Model to use
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        An Object of DocumentEmbedderBaseClass metaclass.

    Raises:
        ImportError: If invalid embedder_type is passed.
    """
    try:
        if "." in embedder_type:
            module_name, class_name = embedder_type.rsplit(".", 1)
        else:
            module_name = embedder_type
            class_name = embedder_type.capitalize() + "Embedder"
        document_embedder_module = import_module(
            f".{module_name}_document_embedder",
            package="competency_maps.document_embedders",
        )
        document_embedder_class = getattr(document_embedder_module, class_name)
        instance = document_embedder_class(*args, **kwargs)
    except (AttributeError, ModuleNotFoundError):
        raise ImportError(
            f"{embedder_type} is not part of our document embedders collection!"
        )
    else:
        if not issubclass(document_embedder_class, DocumentEmbedderBaseClass):
            raise ImportError(
                "We currently don't have {}, but you are welcome to send in the request for it!".format(
                    document_embedder_class
                )
            )
    return instance
