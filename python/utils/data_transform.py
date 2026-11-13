import pandas as pd
import regex as re
import numpy as np

from pathlib import Path
import pyreadstat


def clean_open_file(data=None, 
                    path: Path = None, 
                    output_type: str = 'dataframe', 
                    **kwargs) -> pd.DataFrame:
    """
    Clean a provided DataFrame or open a file from a path, clean it, and return the result.
    Args:
        data (pd.DataFrame, optional): The DataFrame to clean. Defaults to None.
        path (Path, optional): Path to the file to open. Required if data is not provided. Defaults to None.
        output_type (str): The output type, either 'dataframe' or 'numpy'. If 'numpy', unwanted columns will not be removed. Defaults to 'dataframe'.
        **kwargs: Additional arguments to pass to the respective file reading functions.
    Returns:
        pd.DataFrame or np.ndarray: The cleaned DataFrame or NumPy array.
    Raises:
        ValueError: If neither 'data' nor 'path' is provided.
    """
    if data is None and path is None:
        raise ValueError("Either 'data' or 'path' must be provided.")
    
    if data is None :
        file_suffix = path.suffix.lower() # extract suffix (.csv, .txt etc) and convert to lower case

        if file_suffix == '.dta':
            data, meta = open_file(path, output_type = output_type, **kwargs) # if .dta, split the file into data and meta
        else:
            data = open_file(path, output_type = output_type, **kwargs)

    columns_to_drop = ['index', 'Unnamed: 0', 'Unnamed: 0.1','Unnamed: 1', 'level_0'] #columns to drop
    if output_type == 'dataframe':
        data = data.drop(columns=columns_to_drop, errors='ignore')

    if file_suffix == '.dta':
        return data, meta
    else:
        return data
    
def open_file(path: Path, output_type: str = 'dataframe', **kwargs):
    """
    Open a file and return its content as a DataFrame or NumPy array.
    Args:
       path (Path): The path to the file.
       output_type (str): The output type, either 'dataframe' or 'numpy'. Defaults to 'dataframe'.
        **kwargs: Additional arguments to pass to the respective file reading functions.
    Returns:
        pd.DataFrame or np.ndarray: The loaded data.
    Raises:
        ValueError: If the file format is unsupported or output_type is invalid.
    """
    path = Path(path)

    file_suffix = path.suffix.lower()

    match file_suffix:
        case '.csv':
            data = pd.read_csv(path, **kwargs)
        case '.parquet':
            data = pd.read_parquet(path, **kwargs)
        case '.xlsx':
            data = pd.read_excel(path, **kwargs)
        case '.npy':
            data = np.load(path, allow_pickle=True, **kwargs)
        case '.npz':
            with np.load(path, allow_pickle=True, **kwargs) as npz_file:
                data = {key: npz_file[key] for key in npz_file}
        case '.dta':
            data, meta = pyreadstat.read_dta(path, **kwargs)
            meta = vars(meta) # Convert metadata to a dictionary
        case _:
            raise ValueError(f"Unsupported file format: {file_suffix}")

    match output_type:
        case 'dataframe':
            if isinstance(data, pd.DataFrame):
                if file_suffix != '.dta':
                    return data
                else:
                    return data, meta
            elif isinstance(data, np.ndarray):
                if file_suffix != '.dta':
                    return pd.DataFrame(data)
                else:
                    return pd.DataFrame(data), meta
            elif isinstance(data, dict):
                if list(data.keys())==['arr_0']:
                    if file_suffix != '.dta':
                        return pd.DataFrame(data['arr_0'])
                    else:
                        return pd.DataFrame(data['arr_0']), meta
                else:
                    if file_suffix != '.dta':
                        return pd.DataFrame(data)
                    else:
                        return pd.DataFrame(data), meta
            else:
                raise ValueError("Unable to convert data to DataFrame.")
        case 'numpy':
            if isinstance(data, pd.DataFrame):
                if file_suffix != '.dta':
                    return  data.to_numpy()
                else:
                    return  data.to_numpy(), meta
            elif isinstance(data, np.ndarray):
                if file_suffix != '.dta':
                    return data
                else:
                    return data, meta
            elif isinstance(data, dict):
                if file_suffix != '.dta':
                    return  np.column_stack(list(data.values()))
                else:
                    return  np.column_stack(list(data.values())), meta
            else:
                raise ValueError("Unable to convert data to NumPy array.")
        case _:
            raise ValueError(f"Invalid output_type: {output_type}. Must be 'dataframe' or 'numpy'.")
       

def windsorize_column(series:pd.Series, lower_pct:float=0.01, upper_pct:float=0.01):
    """Windsorize a column

    Args:
        series (pd.Series): column to windsorize
        lower_pct (float, optional): lower ratio. Defaults to 0.01.
        upper_pct (float, optional): higher ratio. Defaults to 0.01.

    """
    lower_bound = series.quantile(lower_pct)
    upper_bound = series.quantile(1 - upper_pct)

    return series.clip(lower=lower_bound, upper=upper_bound)
