"""Module to generate a topic model from the input learning resources"""
from importlib import import_module

from competency_maps.topic_models.topic_model import TopicModel


def grab(topic_model_type, *args, **kwargs):
    """ Identify the appropriate type of Topic Model Object to be used.

        Two types of preprocessor objects are available:
        1. LDA - Uses the LDA Topic Model
        2. LDAMallet - Uses the LDA Mallet Topic Model
        3. Hierarchical - Uses the Hierarchical Topic Model

        Args:
            topic_model_type (str) : Type of Topic Model to use
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            An Object of TopicModel metaclass.

        Raises:
            ImportError: If invalid topic_model_type is passed.

        """
    try:
        if "." in topic_model_type:
            module_name, class_name = topic_model_type.rsplit(".", 1)
        else:
            module_name = topic_model_type
            class_name = topic_model_type.capitalize() + "TopicModel"
            print(f"Class Name: {class_name}")
        topic_model_module = import_module(
            f".{module_name}_topic_model",
            package="competency_maps.topic_models",
        )
        topic_model_class = getattr(topic_model_module, class_name)
        instance = topic_model_class(*args, **kwargs)
    except (AttributeError, ModuleNotFoundError):
        raise ImportError(
            f"{topic_model_type} is not part of our topic model collection!"
        )
    else:
        if not issubclass(topic_model_class, TopicModel):
            raise ImportError(
                "We currently don't have {}, but you are welcome to send in the request for it!".format(
                    topic_model_class
                )
            )
    return instance
