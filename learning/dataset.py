
# standard imports
from pathlib import Path

# third-party imports
import pandas as pd


class DataSet:
    """Supervised learning dataset.

    Args:
        data_dirpath (str): Data directory path

    Attributes:
        data_directory (pathlib.Path) : Data directory
        input (pandas.DataFrame) : Input vectors dataframe
        output (pandas.DataFrame): output value dataframe

    """

    def __init__(self, data_dirpath="data"):
        self.data_directory = Path(data_dirpath)
        self.input = None
        self.output = None

    def load_input(self, filename):
        """Load dataset input vectors

        Args:
            filename (str): Input vectors filename.

        Returns:
            input (pandas.DataFrame) : Input vectors dataframe
        """
        print("load input")
        self.input = self._load_dataframe(filename)
        return self.input

    def load_output(self, filename, to_serie=True):
        """Load dataset input vectors

        Args:
            filename (str): Output values filename.

        Returns:
            input (pandas.DataFrame) : Output values dataframe
        """
        print("load output")
        self.output = self._load_dataframe(filename)
        if to_serie:
            self.output = self.output[self.output.columns[0]]
        return self.output

    def _load_dataframe(self, filename):
        """Load dataframe from a file name"""
        filepath = self.data_directory / filename
        df = pd.read_csv(filepath, index_col=0)
        return df

    def convert_to_one_hot(self, converter):
        """Convert train and test dataframe columns
        to one-hot vector.

        Args:
            converter (dict): keys columns to convert to one hot
                values prefix of the new columns
        """
        print("convert columns to hot")
        for column, prefix in converter.items():
            one_hot_df = pd.get_dummies(self.input[column], prefix=prefix)
            self.input = self.input.join([one_hot_df])
            self.input = self.input.drop([column], axis="columns")
        return self.input

    def no_null_values(self):
        """Avoid having null values during learing process.

        Returns:
            no_null (bool): True if there's no null
                values in the dataframe
        """
        no_null = not self.input.isnull().values.any()
        return no_null

    def normalize_input(self):
        """Normalize columns values between -1 and 1"""
        print("normalize input values")
        non_boolean_columns = self._find_non_boolean_columns()
        for column in non_boolean_columns:
            self.input[column] = self._normalize_columns(self.input[column])
        return self.input

    def _find_non_boolean_columns(self):
        """Find columns with numerical values (not only zeros and ones)"""
        serie = self.input.isin([0, 1]).all(axis="index")
        columns = serie[serie == False].index.values
        return columns

    def _normalize_columns(self, column):
        return (column - column.mean()) / column.std()
