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

def get_unique_id(df: pd.DataFrame, patient_id:str = 'LOPNR', date:str = 'visit_date')-> pd.DataFrame:
    """Produce a unique ID for each entry (patientID_date)

    Args:
        df (pd.DataFrame): dataframe
        patient_id (str, optional): patiernt ID column name. Defaults to 'LOPNR'.
        date (str, optional): Date column name. Defaults to 'visit_date'.

    Returns:
        pd.DataFrame: dataframe with the column UniqueID
    """
    df[f'{date}_formatted'] = df[date].dt.strftime('%Y%m%d')
    
    df['UniqueID'] = df.apply(lambda row: f"{row[patient_id]}_{row[f'{date}_formatted']}", axis=1)

    df.drop(f'{date}_formatted', inplace=True, axis = 1)

    return df


def remove_special_characters(df:pd.DataFrame, column_name:str, inplace:bool=False)->pd.DataFrame|None:
    """Replace å, ä, ö with a, a, o in df[column_name]

    Args:
        df (pd.DataFrame): dataframe to clean
        column_name (str): name of the column
        inplace (bool, optional): if True, modify the dataframe in place. Defaults to False.

    Returns:
        pd.DataFrame: dataframe with cleaned column
    """
    dict_char = {
        'å': 'a',
        'ä': 'a',
        'ö': 'o',
        'Å': 'A', 
        'Ä': 'A', 
        'Ö': 'O',
    }
    if inplace:
        for char, replacement in dict_char.items():
            df[column_name] = df[column_name].str.replace(char, replacement, regex=False)
        return None
    else:
        df_copy = df.copy()
        for char, replacement in dict_char.items():
            df_copy[column_name] = df_copy[column_name].str.replace(char, replacement, regex=False)
        return df
    

def remove_html_tags(df:pd.DataFrame, column_name:str)->pd.DataFrame:
    """Remove html tags from df[column_name]

    Args:
        df (pd.DataFrame): dataframe to clean
        column_name (str): name of the column

    Returns:
        pd.DataFrame: dataframe with cleaned column
    """
    def tag_replacement(text:str):
        """Remove <i>, <b> and <nl> tags from the text

        Args:
            text (str): text to clean

        """

        if isinstance(text, str):
            tags_list = [r'</?i>', r'</?b>', r'</?nl>']
            for tag in tags_list:
                text = re.sub(tag, '', text)
            return text
        else:
            return text
        
        
    df[column_name] = df[column_name].apply(tag_replacement) # remove html tags
    
    return df


def str_to_bool(string: str):
    """Convert boolean string type to boolean type

    Args:
        string (str): string to convert to boolean

    Returns:
        _type_: _description_
    """
    if string.isin(['True', 'true', '1']):
         return True
    elif string.isin(['False', 'false', '0']):
         return False
    else:
        return np.nan
     
def convert_nan(x: pd.DataFrame):
    """convert all NaN to np.nan

    Args:
        x (pd.DataFrame): DataFrame with potential NaN
    """   
    for p in x.columns:
        x[p] = x[p].fillna(np.nan)
    return x

def time_convert(df : pd.DataFrame,list_columns : list,verbose: bool = False) -> pd.DataFrame:
    """
    Convert object columns to datetime format

    Args:
        df (pd.DataFrame): dataframe containing the column(s)
        list_columns (list): list of the name of the columns to convert

    Returns:
        pd.DataFrame: datafram with converted columns
    """
    list_format = ['%Y-%m-%d %H:%M:%S',
                   '%d/%m/%Y %H:%M',
                   '%d-%m-%Y %H:%M',
                   '%Y-%B-%d %H:%M',
                   '%b %d %Y %I:%M:%S:%f%p',
                   '%b %d %Y %I:%M%p',
                   '%d/%m/%y %H:%M',
                   '%Y-%m-%d %H:%M:%S.%f',
                   '%d%b%Y:%H:%M:%S',
                   '%Y-%m-%dT%H:%M:%S.%f%z',
                   '%d-%m-%Y',
                   '%d%b%Y',
                   '%Y-%m-%d','mixed']
    
    for col in list_columns:
        for form in list_format:
            try:
                df[col] = pd.to_datetime(df[col] ,format=form).astype("datetime64[ms]")
                if verbose:
                    print(f'{col} converted through the format {form}')
            except:
                # If failed casting, try the next format
                continue
            else:
                # If successful casting we want to stop trying for this column
                break
    return df


def diff_time(df:pd.DataFrame, first_time:str = 'min_date', second_time:str = 'max_date', duration:str = 'month', new_column:str = 'diff_date'):
    """Calculate the difference between 2 dates

    Args:
        df (pd.DataFrame): Dataframe
        first_time (str, optional): First time. Defaults to 'min_date'.
        second_time (str, optional): Second time. Defaults to 'max_date'.
        duration (str, optional): Duration type. Defaults to 'month'.
        new_column (str, optional): Name of the new column. Defaults to 'diff_date'.
    """
    if duration == 'month':
        df[new_column] = (df[second_time].dt.year - df[first_time].dt.year) * 12 + \
            (df[second_time].dt.month - df[first_time].dt.month)
    
    return df