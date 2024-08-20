from typing import Union, Optional
import pandas as pd
import dask.dataframe as dd
import modin.pandas as mpd
import cudf

class TimeSeriesData:
    """
    TimeSeriesData class for handling and processing time series data using multiple backends.

    This class is designed to offer flexibility and scalability by supporting various backends like Pandas, Dask, Modin, and CuDF. 
    The idea is to accommodate different user needs, whether they are working with small in-memory datasets or large datasets 
    requiring distributed computing or GPU acceleration.

    Backends Supported:
    - **Pandas:** The most widely used library for data manipulation in Python. Ideal for small to medium-sized datasets.
    - **Dask:** Scales Pandas workflows for larger-than-memory datasets. Suitable for distributed computing.
    - **Modin:** A drop-in replacement for Pandas, optimized for multi-core processors, using Ray or Dask for parallelism.
    - **CuDF:** A GPU-accelerated DataFrame library, part of the RAPIDS suite. Perfect for high-performance environments.
    - **Polars (Future Work):** A fast, memory-efficient DataFrame library written in Rust.

    Design Considerations:
    1. **Backend Flexibility:** Users can switch between backends depending on their dataset size and computational needs. 
       The class automatically detects the backend based on the input DataFrame or allows users to specify it explicitly.
    2. **Optional Grouping:** Grouping by a specific ID column (e.g., stock ID, city) is optional. If not specified, the data 
       is treated as a single time series. The class ensures no duplicate time entries within groups if grouping is applied.
    3. **Partitioning and Algorithmic Flexibility:** The primary purpose is to enable partitioning schemes (e.g., sliding windows, 
       expanding windows) and apply methods like SHAP or other feature importance tools. These methods will inherit the main class, 
       ensuring robust data checks and flexible usage.

    Example:
        >>> ts_data = TimeSeriesData(df, time_col='timestamp', id_col='stock_id', backend='dask')
        >>> masv_scores = ts_data.run_method(calculate_masv, model=sklearn_model)

    Args:
        df (Union[pd.DataFrame, dd.DataFrame, mpd.DataFrame, cudf.DataFrame]): The input DataFrame.
        time_col (str): The column representing time in the DataFrame.
        id_col (Optional[str]): The column representing the ID for grouping. Default is None.
        backend (str): The backend to use ('pandas', 'dask', 'modin', 'cudf'). Default is 'pandas'.
    """

    def __init__(self, 
                 df: Union[pd.DataFrame, dd.DataFrame, mpd.DataFrame, cudf.DataFrame], 
                 time_col: str, 
                 id_col: Optional[str] = None, 
                 backend: str = 'pandas'):
        self.df = df
        self.time_col = time_col
        self.id_col = id_col
        self.backend = backend.lower()
        self._validate_input()

    def _validate_input(self):
        """Validate the input DataFrame and ensure required columns are present."""
        if self.backend == 'pandas':
            assert isinstance(self.df, pd.DataFrame), "Expected a Pandas DataFrame"
        elif self.backend == 'dask':
            assert isinstance(self.df, dd.DataFrame), "Expected a Dask DataFrame"
        elif self.backend == 'modin':
            assert isinstance(self.df, mpd.DataFrame), "Expected a Modin DataFrame"
        elif self.backend == 'cudf':
            assert isinstance(self.df, cudf.DataFrame), "Expected a CuDF DataFrame"
        else:
            raise ValueError("Unsupported backend. Use 'pandas', 'dask', 'modin', or 'cudf'.")

        if self.time_col not in self.df.columns:
            raise ValueError(f"{self.time_col} must be in the DataFrame columns")
        
        if self.id_col and self.id_col not in self.df.columns:
            raise ValueError(f"{self.id_col} must be in the DataFrame columns if provided")
        
        # Sort by time (and id if provided)
        if self.id_col:
            self.df = self.df.sort_values(by=[self.id_col, self.time_col])
        else:
            self.df = self.df.sort_values(by=self.time_col)

        # Check for duplicate time entries within groups
        if self.id_col:
            duplicates = self.df.duplicated(subset=[self.id_col, self.time_col])
        else:
            duplicates = self.df.duplicated(subset=[self.time_col])
        
        if duplicates.any():
            raise ValueError("Duplicate time entries found within the same group.")

    def groupby(self):
        """Group the DataFrame by the ID column if provided."""
        if self.id_col:
            return self.df.groupby(self.id_col)
        return self.df

    def run_method(self, method, *args, **kwargs):
        """
        Run an analytical method on the data.

        Args:
            method (Callable): The method to run.
            *args, **kwargs: Additional arguments to pass to the method.

        Returns:
            The result of the analytical method.
        """
        return method(self, *args, **kwargs)
