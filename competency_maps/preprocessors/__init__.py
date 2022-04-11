"""Module to preprocess the text in the input learning resources"""
from importlib import import_module

from competency_maps.preprocessors.preprocessor import Preprocessor


def grab(preprocessor_type, *args, **kwargs):
    """ Identify the appropriate type of Preprocessor Object to be used.

    Two types of preprocessor objects are available:
    1. spacy - Uses the spacy library to perform the preprocessing steps.
    2. nltk - Uses the nltk library to perform the preprocessing steps.

    Args:
        preprocessor_type (str) : Type of Preprocessing Pipeline to use
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        An Object of Preprocessor metaclass.

    Raises:
        ImportError: If invalid preprocessor_type is passed.

    """
    try:
        if "." in preprocessor_type:
            module_name, class_name = preprocessor_type.rsplit(".", 1)
        else:
            module_name = preprocessor_type
            class_name = preprocessor_type.capitalize() + "Preprocessor"
        preprocessor_module = import_module(
            f".{module_name}_preprocessor",
            package="competency_maps.preprocessors",
        )
        preprocessor_class = getattr(preprocessor_module, class_name)
        instance = preprocessor_class(*args, **kwargs)
    except (AttributeError, ModuleNotFoundError):
        raise ImportError(
            f"{preprocessor_type} is not part of our topic model collection!"
        )
    else:
        if not issubclass(preprocessor_class, Preprocessor):
            raise ImportError(
                "We currently don't have {}, but you are welcome to send in the request for it!".format(
                    preprocessor_class
                )
            )
    return instance
