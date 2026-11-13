##############################
# Purpose: Longitudinal analyses on Nulisa biomarkers alone

# Date creation: 2025-07-18
# Author: ludivinemv

# Input data: 
#       dataset_original.csv

#Output data:

##############################


#! PUBLIC IMPORTS
import os
from tqdm import tqdm

#! PRIVATE IMPORTS
from projectspecific.project_functions import get_data, get_subset
from utils.data_transform import windsorize_column

from utils.analyses_functions import get_lineplot_bygroup

#! GLOBAL VARIABLES
from projectspecific.config import LONG_PATH, COLORINTER, COLORCONTROL
from projectspecific.features_lists import *

#! FUNCTIONS

if __name__ == "__main__":

    #! #################################################################
    #! Options ########################################################
    #! #################################################################
    centered = False
    subset = 'mitt'
    windsor = True
    windsor_ratio = 1

    #! #################################################################
    #! Load and clean the data #########################################
    #! #################################################################
    df_all, meta = get_data()
    
    df_mitt = df_all[df_all['mITT']==1] 
    
    #! Lists ##########################################################
    col_id = ['ID', 'group']

    # List of Nulisa biomarkers columns
    col_nulisa = [item for item in df_all.columns if item.lower().startswith('nulisa')]
    # List of Nulisa biomarkers baseline columns
    col_nulisa_baseline = [s for s in col_nulisa if s.endswith('_0y')]
    # List of Nulisa biomarkers names
    bbm_list = [s.rsplit('_', 1)[0] for s in col_nulisa_baseline]

    col_cofacteurs_base = ['age_0y','sex','campid','educ_years_0y']

    #! Windsorize data
    df_windsor_all= df_all.copy()
    for col in tqdm(col_nulisa):
        df_windsor_all[col] = windsorize_column(df_all[col], windsor_ratio/100, windsor_ratio/100)

    df_windsor_mitt = df_mitt.copy()
    for col in tqdm(col_nulisa):
        df_windsor_mitt[col] = windsorize_column(df_mitt[col], windsor_ratio/100, windsor_ratio/100)

    dict_df = {
        'df_all': df_all,
        'df_mitt': df_mitt,
        'df_windsor_all': df_windsor_all,
        'df_windsor_mitt': df_windsor_mitt,
    }

    print(f'********** {subset}, windsorization: {windsor}')
    df = get_subset(dict_df, subset, windsor)
    
    path_options = LONG_PATH/'bbm'/f'{subset}_windsor_{windsor}'

    if centered:
        path_options += '_centered'

    os.makedirs(path_options, exist_ok=True)


    #! Standardisation

    for bbm in bbm_list:
        mean_0y = df[f'{bbm}_0y'].mean()
        sd_0y = df[f'{bbm}_0y'].std()

        for year in [0, 2]:
            df[f'{bbm}_{year}y_z'] = (
                df[f'{bbm}_{year}y'] - mean_0y
            ) / sd_0y




    #! #################################################################
    #! Visualizing  BBM levels by time and Intervention group (L1) (L7) (L8) ############
    #! #################################################################
    os.makedirs(path_options/'bbm_by_group_time_change_z', exist_ok=True)

    df_name_update = df.copy()
    df_name_update.columns = df_name_update.columns.str.replace(r"^nulisa_", "", regex=True)

    without_nulisa = [x.replace("nulisa_", "", 1) if x.startswith("nulisa_") else x for x in bbm_list]
    
    # change zscore
    for bbm_name in tqdm(without_nulisa):
        get_lineplot_bygroup(df_name_update, outcome = bbm_name, path = path_options/'bbm_by_group_time_change_z', outcome_years = [0, 2], color_interv=COLORINTER, color_control=COLORCONTROL, standardised=True, change = True)
