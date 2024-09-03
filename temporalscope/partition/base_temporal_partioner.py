"""temporalscope/partitioning/base_temporal_partitioner.py

This module defines the BaseTemporalPartitioner class, an abstract base class for all temporal partitioning methods.
Each partitioning method must inherit from this class and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Union
import pandas as pd
import polars as pl
from temporalscope.config import TF_DEFAULT_CFG


class BaseTemporalPartitioner(ABC):
    """Abstract base class for temporal partitioning methods. This class enforces a
    consistent API for all partitioning methods.

    :param data: The dataset to be partitioned.
    :type data: Union[pd.DataFrame, pl.DataFrame]
    :param time_col: The time column name, which will be used for sorting.
    :type time_col: str
    :param id_col: Optional. The column used for grouping (e.g., stock ticker, item ID).
    :type id_col: Optional[str]
    :param backend: The backend to use ('pl' for Polars or 'pd' for Pandas). Default is 'pl'.
    :type backend: str

    .. note::
        The backend parameter controls whether the partitioning is performed using Polars or Pandas. Ensure that the
        data passed to the partitioner is compatible with the specified backend.

    .. rubric:: Examples

    .. code-block:: python

        # Example usage with Pandas DataFrame
        data = pd.DataFrame({'time': pd.date_range(start='2021-01-01', periods=20, freq='D'), 'value': range(20)})
        partitioner = SlidingWindowPartitioner(data=data, time_col='time', backend='pd')

        # Example usage with Polars DataFrame
        data = pl.DataFrame({'time': pl.date_range(start='2021-01-01', periods=20, interval='1d'), 'value': range(20)})
        partitioner = SlidingWindowPartitioner(data=data, time_col='time', backend='pl')
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        time_col: str,
        id_col: str = None,
        backend: str = "pl",
    ):
        self.data = data
        self.time_col = time_col
        self.id_col = id_col

        # Handle both short forms and full names
        backend_lower = backend.lower()
        if backend_lower in TF_DEFAULT_CFG["BACKENDS"]:
            self.backend = TF_DEFAULT_CFG["BACKENDS"][backend_lower]
        elif backend_lower in TF_DEFAULT_CFG["BACKENDS"].values():
            self.backend = backend_lower
        else:
            raise ValueError(
                f"Unsupported backend '{backend}'. Supported backends are: "
                f"{', '.join(TF_DEFAULT_CFG['BACKENDS'].keys())}, "
                f"{', '.join(TF_DEFAULT_CFG['BACKENDS'].values())}"
            )

        # Validate the data type based on the specified backend
        self._check_data_type(self.data, self.backend)

        # Sort data by time_col and id_col (if provided)
        self.data = self._sort_data()

    def _check_data_type(self, data: Any, expected_type: str) -> None:
        """Check the type of the data against the expected type (Pandas or Polars).

        :param data: The data to check.
        :type data: Any
        :param expected_type: The expected type ('pandas' or 'polars').
        :type expected_type: str
        :raises TypeError: If the data type does not match the expected type.
        """
        if expected_type == "pandas" and not isinstance(data, pd.DataFrame):
            raise TypeError("Expected data to be a Pandas DataFrame.")
        elif expected_type == "polars" and not isinstance(data, pl.DataFrame):
            raise TypeError("Expected data to be a Polars DataFrame.")

    def _sort_data(self) -> Union[pd.DataFrame, pl.DataFrame]:
        """Sorts the data by time_col and id_col (if provided).

        :return: The sorted DataFrame.
        :rtype: Union[pd.DataFrame, pl.DataFrame]
        """
        if self.backend == "pandas":
            if self.id_col:
                return self.data.sort_values(
                    by=[self.id_col, self.time_col]
                ).reset_index(drop=True)
            return self.data.sort_values(by=self.time_col).reset_index(drop=True)
        elif self.backend == "polars":
            if self.id_col:
                return self.data.sort([self.id_col, self.time_col])
            return self.data.sort(self.time_col)

        raise TypeError(
            "Unsupported data type. Data must be a Pandas or Polars DataFrame."
        )

    @abstractmethod
    def get_partitions(self) -> List[Tuple[int, int]]:
        """Abstract method that must be implemented by subclasses to generate partitions.

        :return: List of tuples where each tuple represents the start and end indices of a partition.
        :rtype: List[Tuple[int, int]]
        """
        pass

    @abstractmethod
    def apply_partition(
        self, partition: Tuple[int, int]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Abstract method that must be implemented by subclasses to apply a partition to the data.

        :param partition: A tuple representing the start and end indices of the partition.
        :type partition: Tuple[int, int]
        :return: The partitioned DataFrame.
        :rtype: Union[pd.DataFrame, pl.DataFrame]
        """
        pass

    def get_partitioned_data(self) -> List[Union[pd.DataFrame, pl.DataFrame]]:
        """Helper method that returns the data for each partition as a list of DataFrames.

        :return: List of partitioned DataFrames.
        :rtype: List[Union[pd.DataFrame, pl.DataFrame]]
        """
        partitions = self.get_partitions()
        partitioned_data = [self.apply_partition(partition) for partition in partitions]
        self._check_data_type(
            partitioned_data[0],
            "pandas" if self.backend == "pandas" else "polars",
        )
        return partitioned_data

    def get_partition_indices(self) -> List[Tuple[int, int]]:
        """Helper method that returns the indices for each partition.

        :return: List of tuples where each tuple represents the start and end indices of a partition.
        :rtype: List[Tuple[int, int]]
        """
        return self.get_partitions()
