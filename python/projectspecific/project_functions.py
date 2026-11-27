# Functions used by several scripts but specific to the project

import ast
import numpy as np
import pandas as pd

from utils.data_transform import clean_open_file

from projectspecific.config import DATA_ORIGINAL_PATH, DATA_PROCESSED_PATH



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
    
    # Get the string of a dictionary and convert it to a dictionary
    label_map = {2: 'Intervention', 3: 'Control'}
    # get Intervention and Control instead of 2 and 3 in df['group'] 
    df['group'] = df['sub_sample_id'].replace(label_map)

    # standardize columns names
    dict_yrs = {r'_24kk$':'_2y',r'_12kk$': '_1y',r'_Scr$' : '_0y', r'_bl1$' : '_0y', 
              r'_BL2$' : '_0y', r'_BL1$' : '_0y', r'_bl2$' : '_0y', r'_0$' : '_0y',
                r'_1$' : '_1y',r'_2$' : '_2y'   }
    
    for key, value in dict_yrs.items():
        df.columns = df.columns.str.replace(key, value, regex=True)
    
    # standardize cognition columns names
    col_cognitive = [item for item in df.columns if item.lower().startswith('z')]
    cogn_list = set([s.rsplit('_', 1)[0] for s in col_cognitive])

    for cogn in cogn_list:
        df[f'{cogn}_change'] = df[f'{cogn}_2y'] - df[f'{cogn}_0y']

    
    # get the patients in the mITT subgroup
    df['mITT'] = np.where((df['ztotal_composite_1y'].notna()) | (df['ztotal_composite_2y'].notna()), 1, 0)


    return df

