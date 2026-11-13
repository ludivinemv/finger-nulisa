# Functions used by several scripts but specific to the project

from pathlib import Path
import sys, ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.data_transform import clean_open_file

from projectspecific.config import DATA_ORIGINAL_PATH, DATA_PROCESSED_PATH, COLORCONTROL, COLORINTER, DEMO_PATH,COLORLOWADHERENCE



def get_data():
    """ - Open data and meta data
        - standardize column names
        - find patients from mITT subgroup
    Return 
        df: Standardized data
        meta: meta data
    """
    # open meta and data files
    data_path = DATA_ORIGINAL_PATH/'dataset_original.csv'

    df = clean_open_file(path=data_path, output_type='dataframe')
    meta = clean_open_file(path=DATA_PROCESSED_PATH/'description'/'columns_description.csv', output_type='dataframe')
    
    # Get the string of a dictionary and convert it to a dictionary
    label_map = ast.literal_eval(meta[meta['column']=='sub_sample_id']['values'].values[0])
    # get Intervention and Control instead of 2 and 3 in df['group'] 
    df['group'] = df['sub_sample_id'].replace(label_map)

    # standardize columns names
    dict_yrs = {r'_24kk$':'_2y',r'_12kk$': '_1y',r'_Scr$' : '_0y', r'_bl1$' : '_0y', 
              r'_BL2$' : '_0y', r'_BL1$' : '_0y', r'_bl2$' : '_0y', r'_0$' : '_0y',
                r'_1$' : '_1y',r'_2$' : '_2y'   }

    for key, value in dict_yrs.items():
        df.columns = df.columns.str.replace(key, value, regex=True)
        meta['column'] = meta['column'].str.replace(key, value, regex=True)

    dict_multilifestyle= {
        'multilifestyle0': 'multilifestyle_0y',
        'multilifestyle1': 'multilifestyle_1y',
        'multilifestyle2': 'multilifestyle_2y',
        }
    
    df.rename(columns=dict_multilifestyle, inplace=True)
    for key,value in dict_multilifestyle.items():
        meta['column'] = meta['column'].str.replace(key, value, regex=True)

    # add age at 1 and 2 years
    df['age_1y'] = df['age_0y'] + 1
    df['age_2y'] = df['age_0y'] + 2
    
    
    # standardize cognition columns names
    col_cognitive = ['mmse_tot_0y','mmse_tot_2y'] + [item for item in df.columns if item.lower().startswith('z')]
    cogn_list = set([s.rsplit('_', 1)[0] for s in col_cognitive])

    for cogn in cogn_list:
        df[f'{cogn}_change'] = df[f'{cogn}_2y'] - df[f'{cogn}_0y']

    
    # get the patients in the mITT subgroup
    df['mITT'] = np.where((df['ztotal_composite_1y'].notna()) | (df['ztotal_composite_2y'].notna()), 1, 0)

    meta = pd.concat([meta, pd.DataFrame([
                                        ['mITT', 'Individuals from the modified intention-to-treat group', np.nan, 'int64', 0,0,0,0],
                                        ['age_1y', 'Age at 1 year', 'years', 'float64', 1,0,1,0],
                                         ['age_2y', 'Age at 2 years', 'years', 'float64', 1,0,0,1],
                                         ['group', 'Intervention group', np.nan, np.nan, 0,0,0,0],
                                         ],

                                        columns=meta.columns)],
                     ignore_index=True)


    nulisa_col = meta[meta['column'].str.startswith('nulisa_')]['column'].tolist()

    for c in nulisa_col:
        meta[meta['column']==c]['description'] = f"Nulisa {c.rsplit('_', 2)[1] } at year {c.rsplit('_', 2)[2].replace('y','')}"

    return df, meta


def get_histogram(df:pd.DataFrame, 
                  column_name:str, 
                  by_group:bool = True, 
                  xlabel:str = '', 
                  ylabel:str = '', 
                  kde:bool = True, 
                  path:Path|str = DEMO_PATH, 
                  bins:int|None = None, 
                  add_name:str|None = None, 
                  verbose:bool = True,
                  group: str = 'group'):
    """ Get histograms of a variable per group or in the full population

    Args:
        df (pd.DataFrame): Dataframe with data
        column_name (str): column name
        by_group (bool, optional): If True, print histogram by group. Defaults to True.
        xlabel (str, optional): x label. Defaults to ''.
        ylabel (str, optional): y label. Defaults to ''.
        kde (bool, optional): If True, print estimated curve. Defaults to True.
        path (Path | str, optional): Path to save. Defaults to DEMO_PATH.
        bins (int, optional): Number of bins. If None, default value. Defaults to None.
        add_name (str | None, optional): Additional string to add the file name. Defaults to None.
        verbose (bool, optional): If True print the histogram. Defaults to True.
    """
    if by_group:
        if group == 'group':
            if bins:
                sns.histplot(df[df['group']=='Intervention'][column_name], kde=kde, stat='density', bins=bins,color = COLORINTER, label = 'Intervention')
                sns.histplot(df[df['group']=='Control'][column_name], kde=kde, stat='density', bins=bins, color = COLORCONTROL, label = 'Control')
            else:
                sns.histplot(df[df['group']=='Intervention'][column_name], kde=kde, stat='density', color = COLORINTER, label = 'Intervention')
                sns.histplot(df[df['group']=='Control'][column_name], kde=kde, stat='density', color = COLORCONTROL, label = 'Control')
        else:     
            if bins:
                sns.histplot(df[df[group]==2][column_name], kde=kde, stat='density', bins=bins,color = COLORINTER, label = 'High adherence')
                sns.histplot(df[df[group]==1][column_name], kde=kde, stat='density', bins=bins,color = COLORLOWADHERENCE, label = 'Low adherence')
                sns.histplot(df[df[group]==0][column_name], kde=kde, stat='density', bins=bins,color = COLORCONTROL, label = 'Control')
            else:
                sns.histplot(df[df[group]==2][column_name], kde=kde, stat='density', color = COLORINTER, label = 'High adherence')
                sns.histplot(df[df[group]==1][column_name], kde=kde, stat='density', color = COLORLOWADHERENCE, label = 'Low adherence')
                sns.histplot(df[df[group]==0][column_name], kde=kde, stat='density', color = COLORCONTROL, label = 'Control')

        plt.title(f'Distribution of {column_name} by intervention group')
        if group == 'group':
            plt.legend(title='Intervention group')
        else:
            plt.legend(title='Adherence group')
    else:
        if bins:
            sns.histplot(df[column_name], kde=True, bins=bins, stat='density', label = 'column_name')
        else:
            sns.histplot(df[column_name], kde=True, stat='density', label = 'column_name')
        plt.title(f'Distribution of {column_name}')
    
    if ylabel == '':
        plt.ylabel('Density')
    else:
        plt.ylabel(ylabel)
    if ylabel == '':
        plt.ylabel(column_name)
    else:
        plt.ylabel(xlabel)

    group_filename = f'_{group}' if by_group else '' #if by_group == True, add _group to the file name, else keep empty
    if add_name:
        group_filename += add_name
    plt.savefig(path/f'hist_{column_name}{group_filename}.png')
    if verbose:
        plt.show()
    plt.close()

    
def get_subset(dict_df:dict, subset:str, windsor:bool) -> pd.DataFrame:
    """Get subset of data

    Args:
        dict_df (dict): dictionary with the dataframe for each subset
        subset (str): subset name
        windsor (bool): If True, use windsorised data

    Returns:
        df (pd.DataFrame): dataframe with data for the selected subset
    """
    if windsor:
        if subset == 'mitt':
            print('df_windsor_mitt')
            df = dict_df['df_windsor_mitt'].copy()
        else:
            print('df_windsor_all')
            df = dict_df['df_windsor_all'].copy()
    else:
        if subset == 'mitt':
            print('df_mitt')
            df = dict_df['df_mitt'].copy()
        else:
            print('df_all')
            df = dict_df['df_all'].copy()

    return df
