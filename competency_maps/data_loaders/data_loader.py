from abc import ABCMeta, abstractmethod


class DataLoader(metaclass=ABCMeta):
    """ Metaclass for defining DataLoaders

    Use this metaclass for defining a DataLoader Class. A DataLoader class is used to load the data containg the
    learning resources.

    Attributes:
        input_path (str) : Path containing the Input Learning Resources. Could be a folder or a CSV File
    """

    def __init__(self, path):
        """ Initialise the class attributes

        Args:
            path (str): Path containing the Input Learning Resources. Could be a folder or a CSV File
        """
        self.input_path = path

    @abstractmethod
    def read_corpus(self):
        """ Read the Data specified in the input_path and return a pandas Dataframe"""
        pass
