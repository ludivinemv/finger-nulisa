
##############################
# Purpose: Plot 2d effect of Nulisa biomarkers on cognition over time

# Date creation: 2025-07-18
# Author: Ludivine Morvan

# Input data: 
#       luovutettavat_85_1.csv

#Output data:

##############################


#! PUBLIC IMPORTS
from pathlib import Path
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

#! PRIVATE IMPORTS

sys.path.append('P:/H1_FINGER AD BBM/')
sys.path.append('P:/H1_FINGER AD BBM/Nulisa Biomarkers/Scripts/')
from utils.data_transform import windsorize_column

from projectspecific.project_functions import get_data, get_subset

#! GLOBAL VARIABLES
from projectspecific.config import LONG_PATH
from projectspecific.features_lists import *

#! FUNCTIONS

def standardize_series(series: pd.Series) -> pd.Series:
    """Standardize a series to z-scores."""
    return (series - series.mean()) / series.std()

def clean_bm_name(bm: str) -> str:
    """Remove nulisa_ prefix and _0y, _2y, _0y_z suffixes for display."""
    if bm.startswith('nulisa_'):
        parts = bm.split('_')
        # Remove 'nulisa' and any time suffix (_0y, _2y, _0y_z, etc.)
        cleaned = '_'.join([p for p in parts[1:] if not p.endswith(('0y', '1y', '2y', '0y_z', '1y_z', '2y_z'))])
        return cleaned
    return bm
def rename_nulisa_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns starting with nulisa_XX_0y, nulisa_XX_2y, or nulisa_XX_0y_z to XX."""
    new_columns = {}
    for col in df.columns:
        if col.startswith('nulisa_'):
            # Extract the part after nulisa_ and before _0y, _2y, or _0y_z
            parts = col.split('_')
            if len(parts) >= 3:
                new_col = '_'.join(parts[1:-1])  # Join all parts except the first and last
                new_columns[col] = new_col
            else:
                new_columns[col] = col
        else:
            new_columns[col] = col
    return df.rename(columns=new_columns)

def plot_lineplot_by_bm_level(df:pd.DataFrame, bm:str, outcome:str, 
                              path_options_output:Path|str, 
                              name_centered:str = "", 
                              chosen_thresholds:bool = False, 
                              dict_thr:dict = {},
                              use_standardised: bool = False,
                              use_change: bool = True,
                              outcome_name_y:str = "Global cognitive change"):
    """Plot lineplot of outcome over time by biomarker level (low vs high)

    Args:
        df (pd.DataFrame): Dataframe with data
        bm (str): Biomarker name
        outcome (str): Outcome name
        path_options_output (Path | str): Path to save the plots
        name_centered (str, optional): Name of the file according to the option centered. Defaults to "".
        chosen_thresholds (bool, optional): If True, use thresholds from dict_thr. Defaults to False.
        dict_thr (dict, optional): Dictionary with selected thresholds in the format: 'bm':[threshold_low, threshold_high]. Defaults to {}.
        use_standardised (bool, optional): If True, use standardised biomarker (z-score). Defaults to False.
        use_change (bool, optional): If True, plot change over time instead of raw outcome. Defaults to True.
    
    """

    outcome_folder = outcome + "_change" if use_change else outcome
    os.makedirs(path_options_output/outcome_folder/f'all{name_centered}', exist_ok=True)
    os.makedirs(path_options_output/outcome_folder/f'group{name_centered}', exist_ok=True)
           

    # Clean biomarker name for display
    bm_display = clean_bm_name(bm)

    bm_display = bm_display + "(z score)" if use_standardised else bm_display

    # Select relevant columns    
    bm_col = f"{bm}_0y"

    outcome_cols = [f"{outcome}_0y", f"{outcome}_1y", f"{outcome}_2y"]
    df_prot = df[['ID', 'group', bm_col] + outcome_cols].dropna()


    # Standardize biomarker if requested and not already z-scored
    if use_standardised :
        df_prot[f"{bm}_0y_z"] = standardize_series(df_prot[bm_col])
        bm_col = f"{bm}_0y_z"

    # Define thresholds
    if not chosen_thresholds:
        firstquarter = df_prot[bm_col].quantile(0.25)
        lastquarter = df_prot[bm_col].quantile(0.75)
        name_chosen = ""
    else:
        firstquarter = dict_thr[bm][0]
        lastquarter = dict_thr[bm][1]
        name_chosen = f"_{firstquarter}_{lastquarter}"

    # Split into low and high biomarker groups
    x_prot_low = df_prot[bm_col] <= firstquarter
    x_prot_high = df_prot[bm_col] >= lastquarter

    df_low = df_prot[x_prot_low].copy()
    df_high = df_prot[x_prot_high].copy()


    # Calculate change over time if requested
    if use_change:
        df_low[f'{outcome}_0y_change'] = 0
        df_low[f"{outcome}_1y_change"] = df_low[f"{outcome}_1y"] - df_low[f"{outcome}_0y"]
        df_low[f"{outcome}_2y_change"] = df_low[f"{outcome}_2y"] - df_low[f"{outcome}_0y"]
        df_high[f'{outcome}_0y_change'] = 0
        df_high[f"{outcome}_1y_change"] = df_high[f"{outcome}_1y"] - df_high[f"{outcome}_0y"]
        df_high[f"{outcome}_2y_change"] = df_high[f"{outcome}_2y"] - df_high[f"{outcome}_0y"]
        outcome_cols = [f"{outcome}_0y_change", f"{outcome}_1y_change", f"{outcome}_2y_change"]
    else:
        outcome_cols = [f"{outcome}_0y_change", f"{outcome}_1y", f"{outcome}_2y"]


    # Melt to long format
    df_low = df_low.reset_index(drop=True).melt(
        id_vars=['ID', 'group', bm_col],
        value_vars=outcome_cols,
        var_name='year',
        value_name=outcome
    )
    df_high = df_high.reset_index(drop=True).melt(
        id_vars=['ID', 'group', bm_col],
        value_vars=outcome_cols,
        var_name='year',
        value_name=outcome
    )

    # Rename columns for plotting (only for display)
    df_low = rename_nulisa_columns(df_low)
    df_high = rename_nulisa_columns(df_high)

    # Extract year for plotting
    df_low['year'] = df_low['year'].str.extract(r'(\d)')
    df_high['year'] = df_high['year'].str.extract(r'(\d)')


    # Lineplot for full population
    sns.lineplot(x='year', y=outcome, data=df_low, label=f'Low {bm_display} (≤{round(firstquarter,2)})', color='plum')
    sns.lineplot(x='year', y=outcome, data=df_high, label=f'High {bm_display} (≥{round(lastquarter,2)})', color='indigo')
    plt.legend()
    plt.title(f'{outcome_name_y} over time by {bm_display} level')
    plt.ylabel(outcome_name_y)
    plt.savefig(path_options_output / outcome_folder / f'all{name_centered}' / f'{outcome}_by_{bm}_value_lineplot{name_chosen}.png')
    plt.savefig(path_options_output / outcome_folder / f'all{name_centered}' / f'{outcome}_by_{bm}_value_lineplot{name_chosen}.svg')
    plt.close()

    # Per group without CI
    sns.lineplot(x='year', y=outcome, data=df_low[df_low['group']=='Intervention'], label=f'Intervention - Low {bm_display} (≤{round(firstquarter,2)})', color='plum', ci=None)
    sns.lineplot(x='year', y=outcome, data=df_high[df_high['group']=='Intervention'], label=f'Intervention - High {bm_display} (≥{round(lastquarter,2)})', color='indigo', ci=None)
    sns.lineplot(x='year', y=outcome, data=df_low[df_low['group']=='Control'], label=f'Control - Low {bm_display} (≤{round(firstquarter,2)})', color='silver', ci=None)
    sns.lineplot(x='year', y=outcome, data=df_high[df_high['group']=='Control'], label=f'Control - High {bm_display} (≥{round(lastquarter,2)})', color='#7D7A7A', ci=None)
    plt.legend()
    plt.ylabel(outcome_name_y)
    plt.title(f'{outcome_name_y} over time by {bm_display} level per group')
    plt.savefig(path_options_output / outcome_folder / f'group{name_centered}' / f'{outcome}_by_{bm}_value_lineplot{name_chosen}_group.png')
    plt.savefig(path_options_output / outcome_folder / f'group{name_centered}' / f'{outcome}_by_{bm}_value_lineplot{name_chosen}_group.svg')

    plt.close()

    # Per group with CI
    sns.lineplot(x='year', y=outcome, data=df_low[df_low['group']=='Intervention'], label=f'Intervention - Low {bm_display} (≤{round(firstquarter,2)})', color='plum')
    sns.lineplot(x='year', y=outcome, data=df_high[df_high['group']=='Intervention'], label=f'Intervention - High {bm_display} (≥{round(lastquarter,2)})', color='indigo')
    sns.lineplot(x='year', y=outcome, data=df_low[df_low['group']=='Control'], label=f'Control - Low {bm_display} (≤{round(firstquarter,2)})', color='silver')
    sns.lineplot(x='year', y=outcome, data=df_high[df_high['group']=='Control'], label=f'Control - High {bm_display} (≥{round(lastquarter,2)})', color='#7D7A7A')
    plt.legend()
    plt.ylabel(outcome_name_y)
    plt.title(f'{outcome_name_y} over time by {bm_display} level per group')
    plt.savefig(path_options_output / outcome_folder / f'group{name_centered}' / f'{outcome}_by_{bm}_value_lineplot{name_chosen}_group_ci.png')
    plt.savefig(path_options_output / outcome_folder / f'group{name_centered}' / f'{outcome}_by_{bm}_value_lineplot{name_chosen}_group_ci.svg')

    plt.close()


    # Per group with whiskers
    line_thick = 1.5
    
    plt.figure()
    sns.pointplot(
        x='year', y=outcome,
        data=df_low[df_low['group']=='Intervention'],
        label=f'Intervention - Low {bm_display} (≤{round(firstquarter,2)})',
        color='plum',
        join=True,  # Connect points with a line
        capsize=.08, # Size of the whisker caps
        linewidth=line_thick,  # thinner line
    )
    sns.pointplot(
        x='year', y=outcome,
        data=df_high[df_high['group']=='Intervention'],
        label=f'Intervention - High {bm_display} (≥{round(lastquarter,2)})',
        color='indigo',
        join=True,
        capsize=.08,
        linewidth=line_thick,  # thinner line
    )
    sns.pointplot(
        x='year', y=outcome,
        data=df_low[df_low['group']=='Control'],
        label=f'Control - Low {bm_display} (≤{round(firstquarter,2)})',
        color='silver',
        join=True,
        capsize=.08,
        linewidth=line_thick,  # thinner line
    )
    sns.pointplot(
        x='year', y=outcome,
        data=df_high[df_high['group']=='Control'],
        label=f'Control - High {bm_display} (≥{round(lastquarter,2)})',
        color='#7D7A7A',
        join=True,
        capsize=.08,
        linewidth=line_thick,  # thinner line
    )
    plt.ylabel(outcome_name_y)
    plt.legend()
    plt.title(f'{outcome_name_y} over time by {bm_display} level per group')
    plt.savefig(path_options_output / outcome_folder / f'group{name_centered}' / f'{outcome}_by_{bm}_value_lineplot{name_chosen}_group_whiskers.png')
    plt.savefig(path_options_output / outcome_folder / f'group{name_centered}' / f'{outcome}_by_{bm}_value_lineplot{name_chosen}_group_whiskers.svg')
    plt.close()

if __name__ == "__main__":

    #! #################################################################
    #! Options ########################################################
    #! #################################################################
    centered = False
    subset = 'mitt'
    windsor = True
    windsor_ratio = 1
    new_version = True

    #! #################################################################
    #! Load and clean the data #########################################
    #! #################################################################

    df_all, meta = get_data()
    
    if new_version:
        #! Add adherence
        df_all["adherence_class"] = np.where(
            df_all["sub_sample_id"] == 3,
            "control",
            np.where(
                df_all["multiparti"].isin([7, 8]),
                "high",
                np.where(
                    df_all["multiparti"].isin(range(1, 7)),
                    "low",
                    df_all["multiparti"].astype(str)
                )
            ))
                
    df_mitt = df_all[df_all['mITT']==1] if subset == 'mitt' else df_all  
    

    #! Lists ##########################################################
    # List of Nulisa biomarkers columns
    col_nulisa = [item for item in df_all.columns if item.lower().startswith('nulisa')]
    # List of Nulisa biomarkers baseline columns
    col_nulisa_baseline = [s for s in col_nulisa if s.endswith('_0y')]
    # List of Nulisa biomarkers names
    bbm_list = [s.rsplit('_', 1)[0] for s in col_nulisa_baseline]

    # outcomes tot study
    list_outcomes = [
        'ztotal_composite', 
               # 'zprocessing_speed_domain',
               # 'zabbr_memory_domain',
               # 'zexecutive_funct_domain',
               # 'lumipulse_ptau217', 
               # 'multilifestyle'
                ]

    #! Get data ##########################################################
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
    
    #! Define path ##########################################################
    path_options_output = LONG_PATH/'plot2d_bbm'/f'{subset}_windsor_{windsor}'
    if centered:
        path_options_output += '_centered'

    os.makedirs(path_options_output, exist_ok=True)

    name_folder_bm = f'{subset}_windsor_{windsor}'
    if centered:
        name_folder_bm += '_centered'

    if centered:
        list_b = [f"{b}_0y" for b in bbm_list+['lumipulse_ptau217']]
        df[list_b] = df[list_b].subtract(df[list_b].mean())
        
    #! Plots cognitive values ##########################################################


    name_centered = '_centered' if centered else ''

    for outcome in list_outcomes:
        print(f"{outcome} ********************************")
        
        os.makedirs(path_options_output/outcome/f'all{name_centered}', exist_ok=True)
        os.makedirs(path_options_output/outcome/f'group{name_centered}', exist_ok=True)
           

        for bm in tqdm(bbm_list):      
            plot_lineplot_by_bm_level(df, bm, outcome, path_options_output, name_centered, chosen_thresholds = False)


        #! Specific thresholds
        dict_thr = { 'nulisa_PARK7': [11.5,13],
                    'nulisa_BDNF':[12,14],
                    'nulisa_APOE4':[2,12]}
        
        for bm in dict_thr.keys():
            plot_lineplot_by_bm_level(df, bm, outcome, path_options_output, name_centered, chosen_thresholds = True, dict_thr=dict_thr)

    #! Plot cognitive change #######################

    standardised = True
    outcome_years = [0, 1,2]

    df_name_update = df.copy()
    df_name_update.columns = df_name_update.columns.str.replace(r"^nulisa_", "", regex=True)

    without_nulisa = [x.replace("nulisa_", "", 1) if x.startswith("nulisa_") else x for x in bbm_list]
    
    for bbm in without_nulisa:
        mean_0y = df_name_update[f'{bbm}_0y'].mean()
        sd_0y = df_name_update[f'{bbm}_0y'].std()

        for year in [0,  2]:
            df_name_update[f'{bbm}_{year}y_z'] = (
                df_name_update[f'{bbm}_{year}y'] - mean_0y
            ) / sd_0y


    
    for outcome in list_outcomes:
        print(f"{outcome} ********************************")
        sublist = [#'nulisa_AB38',
                    #'nulisa_ACHE',
                    #'nulisa_APOE4',#
                    #'nulisa_ARSA',
                    'nulisa_BDNF',#
                    'nulisa_CCL2',#
                    #'nulisa_CD40LG',
                    #'nulisa_CNTN2',
                    'nulisa_CRH', #
                    #'nulisa_CRP',
                    'nulisa_GDNF',#
                    #'nulisa_GFAP',
                    'nulisa_HTT',
                    #'nulisa_IGFBP7',
                    #'nulisa_IL18',
                    'nulisa_IL7',#
                    'nulisa_KDR',#
                    #'nulisa_MDH1',
                    #'nulisa_NEFH',
                    #'nulisa_NEFL',
                    #'nulisa_NPTX1',
                    #'nulisa_NPTXR',
                    #'nulisa_NRGN',
                    #'nulisa_PDGFRB',
                    'nulisa_PTN',#
                    'nulisa_RUVBL2',#
                    #'nulisa_SFRP1',
                    'nulisa_SNAP25',#
                    'nulisa_SOD1',
                    'nulisa_TREM2',#
                    #'nulisa_pSNCA129',
                    #'nulisa_pTDP43409',
                    #'nulisa_pTau217'
                    ]

        for bm in tqdm(sublist):      
            plot_lineplot_by_bm_level(df, 
                                      bm, 
                                      outcome, 
                                      path_options_output,
                                      name_centered, 
                                      chosen_thresholds = False,
                                      use_standardised = True,
                                      use_change = True,
                                      outcome_name_y = "Global cognitive change"
                                      )

