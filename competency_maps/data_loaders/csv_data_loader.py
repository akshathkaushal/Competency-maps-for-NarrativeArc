import pandas as pd

from .data_loader import DataLoader


class CsvDataLoader(DataLoader):
    """ Data Loader Class to load data from CSV Files

    If the input learning resources is a single csv file where each row represents a learning resource, then use this
    class to extract the data and the content
    """

    def __init__(self, path, delimiter=","):
        """Initialise the class object

        Args:
            path (Path): Input Path containing the learning resources
            delimiter (str): Field Delimiter used to split the fields in the csv file.
        """
        DataLoader.__init__(self, path)
        self.delimiter = delimiter

    def read_corpus(self):
        """Read the CSV File

        Returns:
              Returns a pandas Dataframe after reading the CSV File
        """
        input_df = pd.read_csv(self.input_path, sep=self.delimiter)
        return input_df
