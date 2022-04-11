"""DataLoader Package to read the input learning resources"""
from importlib import import_module

from .data_loader import DataLoader


def grab(data_loader_type, *args, **kwargs):
    """Identify the right DataLoader Class to create as requested.

    Args:
        data_loader_type (str): Type of DataLoader Class to return.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        An Object of DataLoader metaclass.

    Raises:
        ImportError: If invalid data_loader_type is passed.
    """
    try:
        if "." in data_loader_type:
            module_name, class_name = data_loader_type.rsplit(".", 1)
        else:
            module_name = data_loader_type
            class_name = data_loader_type.capitalize() + "DataLoader"
        data_loader_module = import_module(
            "." + module_name + "_data_loader",
            package="competency_maps.data_loaders",
        )
        data_loader_class = getattr(data_loader_module, class_name)
        instance = data_loader_class(*args, **kwargs)
    except (AttributeError, ModuleNotFoundError):
        raise ImportError(
            f"{data_loader_type} is not part of our data loaders collection!"
        )
    else:
        if not issubclass(data_loader_class, DataLoader):
            raise ImportError(
                "We currently don't have {}, but you are welcome to send in the request for it!".format(
                    data_loader_class
                )
            )
    return instance
