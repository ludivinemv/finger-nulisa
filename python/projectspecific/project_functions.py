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

