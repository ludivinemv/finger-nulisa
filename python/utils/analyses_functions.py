
#! PUBLIC IMPORTS
from pathlib import Path
import pandas as pd
import numpy as np  
import os
from scipy.stats import  ttest_ind, mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.lines as mlines

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_shap_effect(dict_results:dict, path:Path|str, col_nulisa:list):
    """Get SHAP value according to the protein level

    Args:
        dict_results (dict): dictionary with results
        path (Path | str): Path to save
        col_nulisa (list): List of Nulisa bm columns

    """

    x_prot = dict_results['x_train'][col_nulisa]

    firstquarter = x_prot.quantile(0.25, axis = 0)
    lastquarter = x_prot.quantile(0.25, axis = 0)
    x_prot_low = x_prot < firstquarter
    x_prot_high = x_prot >= lastquarter


    # Initialize a dictionary to store the results
    results = {}

    # Iterate over each feature
    for feature in col_nulisa:
        # Get the coefficients and the median indicator for the current feature
        coeffs = dict_results['shap_intervention'][feature]
        med_low = x_prot_low[feature]
        med_high = x_prot_high[feature]

        # Calculate the average coefficient for below and above median
        avg_below = coeffs[med_low].median()
        avg_above = coeffs[med_high].median()


        # Store the results
        results[feature] = {
            'average_coeff_below_median': avg_below,
            'average_coeff_above_median': avg_above
        }

    # Convert the results dictionary to a DataFrame for better visualization
    results_df = pd.DataFrame(results).T
    results_df.index = results_df.index.map(lambda x: x.split('_')[1] if isinstance(x, str) and x.startswith('nulisa') and '_' in x else x)

    results_df.to_csv(path/'shap_effect.csv')

    return results_df
    
def get_all_results(cognition:str, 
                    other_parts:bool, 
                    list_cognition:list, 
                    forest:bool,  
                    path_bbm_baseline_cognition:Path|str, 
                    path_bbm_interv:Path|str, 
                    path_forest:Path|str, 
                    feature_selection:str = "correlation_me",
                    name:str = 'bbm', 
                    nulisa:bool = True,
                    intervention:bool = True):
    """Get all results tables from mixed models and causal forest

    Args:
        cognition (str): Cognition score
        other_parts (bool): If True, include other cognition scores
        list_cognition (list): list of cognition scores
        forest (bool): If True, include causal forest results       
        path_bbm_baseline_cognition (Path | str): Path to mixed model results with baseline bbm and cognition
        path_bbm_interv (Path | str): Path to mixed model results with baseline bbm and intervention
        path_forest (Path | str): Path to causal forest results
        feature_selection (str): Feature selection method used for causal forest. Default: 'correlation_me'. Choices: 'correlation_me', 'bbm_evol', 'forest_pval'.
        name (str, optional): Name of the feature. Defaults to 'bbm'. Choices: 'bbm', 'proteins.
        nulisa (bool, optional): If True, use script adapted to Nulisa project. Defaults to True.
        intervention (bool, optional): True if mixed models with intervention as reference are available. Defaults to True.

    """
    # Open mixed model results
    df_mixed_full = pd.read_csv(path_bbm_baseline_cognition/f"{cognition}_change_mixed_model_{name}_baseline.csv") 
    df_mixed_grp = pd.read_csv(path_bbm_baseline_cognition/f"{cognition}_change_mixed_model_{name}_baseline_group.csv")
    if intervention:
        df_mixed_inter = pd.read_csv(path_bbm_baseline_cognition/f"{cognition}_change_mixed_model_{name}_baseline_group_inter.csv")
    
    df_mixed_full_quadra = pd.read_csv(path_bbm_baseline_cognition/f"{cognition}_change_mixed_model_{name}_baseline_quadra.csv") 
    df_mixed_grp_quadra = pd.read_csv(path_bbm_baseline_cognition/f"{cognition}_change_mixed_model_{name}_baseline_group_quadra.csv")
    if intervention:
        df_mixed_inter_quadra = pd.read_csv(path_bbm_baseline_cognition/f"{cognition}_change_mixed_model_{name}_baseline_group_inter_quadra.csv")

    df_mixed_bbm= pd.read_csv(path_bbm_interv/f"{name}_change_mixed_model_group.csv")

    # Create dictionary with all tables
    dict_tables = {'bm_group': df_mixed_bbm,
                f'{cognition}_full': df_mixed_full,
                f'{cognition}_group': df_mixed_grp
                }
    if intervention:
        dict_tables[f'{cognition}_inter'] = df_mixed_inter
    dict_tables[f'{cognition}_full_quadra'] = df_mixed_full_quadra
    dict_tables[f'{cognition}_group_quadra'] = df_mixed_grp_quadra
    if intervention:
        dict_tables[f'{cognition}_inter_quadra'] = df_mixed_inter_quadra

    if other_parts:
        for cogn in list_cognition:
            if cogn != cognition:
                dict_tables[f'{cogn}_full'] = pd.read_csv(path_bbm_baseline_cognition/f"{cogn}_change_mixed_model_{name}_baseline.csv")
                dict_tables[f'{cogn}_group'] = pd.read_csv(path_bbm_baseline_cognition/f"{cogn}_change_mixed_model_{name}_baseline_group.csv")
                if intervention:
                    dict_tables[f'{cogn}_inter'] = pd.read_csv(path_bbm_baseline_cognition/f"{cogn}_change_mixed_model_{name}_baseline_group_inter.csv")

    if nulisa:
        for table_df in dict_tables.values():
            table_df.rename(columns = {'Unnamed: 0': 'bbm'}, inplace = True)
            table_df['bbm'] = table_df['bbm'].apply(lambda x: x.split('_')[1] if isinstance(x, str) and x.startswith('nulisa') and '_' in x else x  )
    
    if forest:
        #! importance ranking
        rank_features = pd.read_csv(path_forest/f'rank_train_importance.csv')
        rank_features = rank_features.drop(columns = {'Unnamed: 0'}).rename(columns = {'feature': 'bbm'})
        rank_features['bbm'] = rank_features['bbm'].apply(lambda x: x.split('_')[1] if isinstance(x, str) and x.startswith('nulisa') and '_' in x else x)
        rank_features['rank'] = rank_features['total_rank'].rank(ascending=True, method='min')

        #! importance values
        value_features = pd.read_csv(path_forest/f'values_train_importance.csv')
        value_features = value_features.drop(columns = {'Unnamed: 0'}).rename(columns = {'feature': 'bbm'})
        value_features['bbm'] = value_features['bbm'].apply(lambda x: x.split('_')[1] if isinstance(x, str) and x.startswith('nulisa') and '_' in x else x)
        value_features = pd.merge(value_features, dict_tables['bm_group']['bbm'], on = 'bbm', how = 'outer')
        for r in ['Permutation', 'Impurity', 'SHAP_intervention', 'SHAP_control','SHAP_diff']:
            value_features[r] = np.where(value_features[r].isna(), 0,value_features[r])
        value_features['sum_coeff'] = value_features[['Permutation', 'Impurity', 'SHAP_intervention']].sum(axis = 1)

        rank_features = pd.merge(rank_features,value_features[['bbm','sum_coeff']], on = 'bbm', how = 'outer')
        for r in ['Permutation', 'Impurity', 'SHAP_intervention', 'SHAP_control','SHAP_diff', 'total_rank', 'total_above_20_rank', 'rank']:
            rank_features[r] = np.where(rank_features[r].isna(),rank_features[r].max()+1,rank_features[r])
        rank_features['rank_int'] = rank_features['rank'].copy()
        rank_features['rank'] = rank_features['rank'].astype(int).astype(str)
        rank_features['rank'] = np.where(rank_features['sum_coeff']== 0, '-', rank_features['rank'])

        dict_tables['importance_rank'] = rank_features
        dict_tables['importance_value'] = value_features


        #! SHAP values
        shap_intervention = pd.read_csv(path_forest/f'shap_values_intervention_train_featselect_{feature_selection}.csv')
        shap_control = pd.read_csv(path_forest/f'shap_values_control_train_featselect_{feature_selection}.csv')

        shap_diff = shap_intervention - shap_control

        #! proteins values in the training set
        x_train = pd.read_csv(path_forest/f'x_prot_train_featselect_{feature_selection}.csv')

        dict_tables['shap_intervention'] = shap_intervention
        dict_tables['shap_control'] = shap_control
        dict_tables['shap_diff'] = shap_diff
        dict_tables['x_train'] = x_train

        for name_dict in ['shap_intervention','shap_control', 'shap_diff', 'x_train']:
            dict_tables[name_dict].columns = dict_tables[name_dict].columns.map(lambda x: x.split('_')[1] if isinstance(x, str) and x.startswith('nulisa') and '_' in x else x)
    
    return dict_tables
    

def create_tables_resume(dict_effect:dict, 
                         dict_results:dict, 
                         cognition:str, 
                         selected_bm:list, 
                         forest:bool, 
                         other_parts:bool, 
                         list_cognition:list, 
                         path:Path|str, 
                         results_df_shap, 
                         intervention:bool = True, 
                         pval:float = 0.1, 
                         pval_fdr:bool = False):
    """Create a resume table

    Args:
        dict_effect (dict): Dictionary with effects
        dict_results (dict): Dictionary with results
        cognition (str): cognitive score
        selected_bm (list): Selected BM
        forest (bool): If True, include causal forest results       
        other_parts (bool): If True, include other cognition scores
        list_cognition (list): list of cognition scores
        path (Path | str): Path to save
        results_df_shap (_type_): Results with SHAP
        intervention (bool, optional): True if mixed models with intervention as reference are available. Defaults to True.
        pval (float, optional): P-value theshold. Defaults to 0.1.
        pval_fdr (bool, optional): If True, use corrected p-value. Defaults to False.

    """

    columns_list = ['low_bbm_cognition_intervention',
                    'high_bbm_cognition_intervention',
                    'bbm_change_vs_intervention', 
                    f'bbm_baseline_vs_{cognition}_change_full_population', 
                    f'bbm_baseline_vs_{cognition}_change_control', 
                    f'bbm_baseline_vs_{cognition}_change_intervention', 
                    f'bbm_baseline_vs_{cognition}_diff_groups',
                    f'bbm_baseline_vs_{cognition}_curve_change_full_population', 
                    f'bbm_baseline_vs_{cognition}_curve_change_control', 
                    f'bbm_baseline_vs_{cognition}_curve_change_intervention', 
                    f'bbm_baseline_vs_{cognition}_curve_diff_groups']
    
    
    if forest:
        columns_list += ['effect_causal_forest_low_val', 'effect_causal_forest_high_val', 'rank_causal_forest']

    if other_parts:
        for cogn in list_cognition:
            if cogn != cognition:
                columns_list += [f'bbm_baseline_vs_{cogn}_change_full_population', 
                                f'bbm_baseline_vs_{cogn}_change_control', 
                                f'bbm_baseline_vs_{cogn}_diff_groups']

    if not intervention:
        columns_list.remove(f'bbm_baseline_vs_{cognition}_change_intervention')
        columns_list.remove(f'bbm_baseline_vs_{cognition}_curve_change_intervention')

    #! Table with coeffs
    df_table_coeff = pd.DataFrame(index = selected_bm, columns =  columns_list)
    df_table_coeff[f'low_bbm_cognition_intervention'] = dict_effect['low_effect']
    df_table_coeff[f'high_bbm_cognition_intervention'] = dict_effect['high_effect']
    df_table_coeff[f'bbm_baseline_vs_{cognition}_change_full_population'] = dict_results[f'{cognition}_full']['bm:year coeff']
    df_table_coeff[f'bbm_baseline_vs_{cognition}_change_control'] = dict_results[f'{cognition}_group']['bm:year coeff']
    if intervention:
        df_table_coeff[f'bbm_baseline_vs_{cognition}_change_intervention'] = dict_results[f'{cognition}_inter']['bm:year coeff']
    df_table_coeff[f'bbm_baseline_vs_{cognition}_diff_groups'] = dict_results[f'{cognition}_group']['group:bm:year coeff']

    df_table_coeff[f'bbm_baseline_vs_{cognition}_curve_change_full_population'] = dict_results[f'{cognition}_full_quadra']['bm:year2 coeff']
    df_table_coeff[f'bbm_baseline_vs_{cognition}_curve_change_control'] = dict_results[f'{cognition}_group_quadra']['bm:year2 coeff']
    if intervention:
        df_table_coeff[f'bbm_baseline_vs_{cognition}_curve_change_intervention'] = dict_results[f'{cognition}_inter_quadra']['bm:year2 coeff']
    df_table_coeff[f'bbm_baseline_vs_{cognition}_curve_diff_groups'] = dict_results[f'{cognition}_group_quadra']['group:bm:year2 coeff']

    df_table_coeff['bbm_change_vs_intervention'] = dict_results['bm_group']['group:year coeff']
    if forest:
        df_table_coeff['effect_causal_forest_low_val'] = results_df_shap['average_coeff_below_median']
        df_table_coeff['effect_causal_forest_high_val'] = results_df_shap['average_coeff_above_median']
        df_table_coeff['rank_causal_forest'] = dict_results['importance_rank']['rank']
        

    if other_parts:
        for cogn in list_cognition:
            if cogn != cognition:
                df_table_coeff[f'bbm_baseline_vs_{cogn}_change_full_population'] = dict_results[f'{cogn}_full']['bm:year coeff']
                df_table_coeff[f'bbm_baseline_vs_{cogn}_change_control'] = dict_results[f'{cogn}_group']['bm:year coeff']
                df_table_coeff[f'bbm_baseline_vs_{cogn}_diff_groups'] = dict_results[f'{cogn}_group']['group:bm:year coeff']



    selected_bm_target = list(df_table_coeff.index)

    #! Table with pvalues
    df_table_pval = pd.DataFrame(index = selected_bm, columns = columns_list)
    df_table_pval[f'low_bbm_cognition_intervention'] = dict_effect['low_effect_power']
    df_table_pval[f'high_bbm_cognition_intervention'] = dict_effect['high_effect_power']
    df_table_pval[f'bbm_baseline_vs_{cognition}_change_full_population'] = dict_results[f'{cognition}_full']['bm:year pvalue']
    df_table_pval[f'bbm_baseline_vs_{cognition}_change_control'] = dict_results[f'{cognition}_group']['bm:year pvalue']
    if intervention:
        df_table_pval[f'bbm_baseline_vs_{cognition}_change_intervention'] = dict_results[f'{cognition}_inter']['bm:year pvalue']
    df_table_pval[f'bbm_baseline_vs_{cognition}_diff_groups'] = dict_results[f'{cognition}_group']['group:bm:year pvalue']

    df_table_pval[f'bbm_baseline_vs_{cognition}_curve_change_full_population'] = dict_results[f'{cognition}_full_quadra']['bm:year2 pvalue']
    df_table_pval[f'bbm_baseline_vs_{cognition}_curve_change_control'] = dict_results[f'{cognition}_group_quadra']['bm:year2 pvalue']
    if intervention:
        df_table_pval[f'bbm_baseline_vs_{cognition}_curve_change_intervention'] = dict_results[f'{cognition}_inter_quadra']['bm:year2 pvalue']
    df_table_pval[f'bbm_baseline_vs_{cognition}_curve_diff_groups'] = dict_results[f'{cognition}_group_quadra']['group:bm:year2 pvalue']

    df_table_pval['bbm_change_vs_intervention'] = dict_results['bm_group']['group:year pvalue']
    if forest:
        df_table_pval['effect_causal_forest_low_val'] = 0.0
        df_table_pval['effect_causal_forest_high_val'] = 0.0
        df_table_pval['rank_causal_forest'] = 0.0

    if other_parts:
        for cogn in list_cognition:
            if cogn != cognition:
                df_table_pval[f'bbm_baseline_vs_{cogn}_change_full_population'] = dict_results[f'{cogn}_full']['bm:year pvalue']
                df_table_pval[f'bbm_baseline_vs_{cogn}_change_control'] = dict_results[f'{cogn}_group']['bm:year pvalue']
                df_table_pval[f'bbm_baseline_vs_{cogn}_diff_groups'] = dict_results[f'{cogn}_group']['group:bm:year pvalue']

    #! Table with qvalues
    df_table_qval = pd.DataFrame(index = selected_bm, columns = columns_list)
    df_table_qval[f'low_bbm_cognition_intervention'] = dict_effect['low_effect_power']
    df_table_qval[f'high_bbm_cognition_intervention'] = dict_effect['high_effect_power']
    df_table_qval[f'bbm_baseline_vs_{cognition}_change_full_population'] = dict_results[f'{cognition}_full']['bm:year pvalue FDR']
    df_table_qval[f'bbm_baseline_vs_{cognition}_change_control'] = dict_results[f'{cognition}_group']['bm:year pvalue FDR']
    if intervention:
        df_table_qval[f'bbm_baseline_vs_{cognition}_change_intervention'] = dict_results[f'{cognition}_inter']['bm:year pvalue FDR']
    df_table_qval[f'bbm_baseline_vs_{cognition}_diff_groups'] = dict_results[f'{cognition}_group']['group:bm:year pvalue FDR']

    df_table_qval[f'bbm_baseline_vs_{cognition}_curve_change_full_population'] = dict_results[f'{cognition}_full_quadra']['bm:year2 pvalue FDR']
    df_table_qval[f'bbm_baseline_vs_{cognition}_curve_change_control'] = dict_results[f'{cognition}_group_quadra']['bm:year2 pvalue FDR']
    if intervention:
        df_table_qval[f'bbm_baseline_vs_{cognition}_curve_change_intervention'] = dict_results[f'{cognition}_inter_quadra']['bm:year2 pvalue FDR']
    df_table_qval[f'bbm_baseline_vs_{cognition}_curve_diff_groups'] = dict_results[f'{cognition}_group_quadra']['group:bm:year2 pvalue FDR']

    df_table_qval['bbm_change_vs_intervention'] = dict_results['bm_group']['group:year pvalue FDR']
    if forest:
        df_table_qval['effect_causal_forest_low_val'] = 0.0
        df_table_qval['effect_causal_forest_high_val'] = 0.0
        df_table_qval['rank_causal_forest'] = 0.0

    if other_parts:
        for cogn in list_cognition:
            if cogn != cognition:
                df_table_qval[f'bbm_baseline_vs_{cogn}_change_full_population'] = dict_results[f'{cogn}_full']['bm:year pvalue FDR']
                df_table_qval[f'bbm_baseline_vs_{cogn}_change_control'] = dict_results[f'{cogn}_group']['bm:year pvalue FDR']
                df_table_qval[f'bbm_baseline_vs_{cogn}_diff_groups'] = dict_results[f'{cogn}_group']['group:bm:year pvalue FDR']

    df_table_coeff.to_csv(path/f'resume_schema_table_coeff_{pval}_fdr_{pval_fdr}.csv')
    df_table_pval.to_csv(path/f'resume_schema_table_pval_{pval}_fdr_{pval_fdr}.csv')
    df_table_qval.to_csv(path/f'resume_schema_table_qval_{pval}_fdr_{pval_fdr}.csv')

    return df_table_coeff, df_table_pval, df_table_qval, selected_bm_target, columns_list


def get_bm_cognitive_change(df:pd.DataFrame, cognition:str, bbm_list:list, nulisa:bool = True):
    """Get cognitive change according to the bm level (low or high)

    Args:
        df (pd.DataFrame): Dataframe with data
        cognition (str): cognitive test
        bbm_list (list): List of biomarkers
        nulisa (bool, optional): If True, use script adapted to Nulisa project. Defaults to True.

    """
    if nulisa:
        cognitive_change = df[f'{cognition}_2y'] - df[f'{cognition}_0y']
        x_prot = df[[f'{b}_0y' for b in bbm_list]]
    else:
        cognitive_change = df['cognitive_change']
        x_prot = df[df['visit']==0][bbm_list]
        
    firstquarter = x_prot.quantile(0.25, axis = 0)
    lastquarter = x_prot.quantile(0.25, axis = 0)

    x_prot_low = x_prot < firstquarter
    x_prot_high = x_prot >= lastquarter
    if nulisa:
        inter = (df['group'] == 1)
        control = (df['group'] == 0)
    else:
        inter = (df['group'] == 'Intervention')
        control = (df['group'] == 'Control')


    low_inter = x_prot_low.copy()
    low_inter.loc[~inter, :] = False
    low_inter = low_inter.mul(cognitive_change, axis=0).replace(0, np.nan)
    low_control = x_prot_low.copy()
    low_control.loc[~control, :] = False
    low_control = low_control.mul(cognitive_change, axis=0).replace(0, np.nan)
    low_effect = low_inter.mean(axis = 0) - low_control.mean(axis = 0)
    
    high_inter = x_prot_high.copy()
    high_inter.loc[~inter, :] = False
    high_inter = high_inter.mul(cognitive_change, axis=0).replace(0, np.nan)
    high_control = x_prot_high.copy()
    high_control.loc[~control, :] = False
    high_control = high_control.mul(cognitive_change, axis=0).replace(0, np.nan)
    high_effect = high_inter.mean(axis = 0) - high_control.mean(axis = 0)

    low_effect_power = pd.Series(np.where((low_effect < -0.01) | (low_effect > 0.01), 0.001, 0.5), index=low_effect.index)
    high_effect_power = pd.Series(np.where((high_effect < -0.01) | (high_effect > 0.01), 0.001, 0.5), index=high_effect.index)

    dict_effect = {'low_effect' : low_effect, 
                   'high_effect' : high_effect, 
                   'low_effect_power' : low_effect_power, 
                   'high_effect_power' : high_effect_power
                   }
    
    return dict_effect


def classify_volcano(row, coeff_name):
    """Classify points in the volcano plot

    Args:
        row (_type_): row
        coeff_name (_type_): name of the coefficient 
    """
    if row['significant']:
        if row[coeff_name] > 0:
            return 'Up'
        else:
            return 'Down'
    else:
        return 'Not significant'
        

def get_boxplot_bygroup(df:pd.DataFrame, 
                        outcome:str, 
                        path:Path|str, 
                        col_id:list = ['ID', 'group'],
                        outcome_years:list = [0, 2],
                        nulisa = True):
    """Get boxplot of outcome by group

    Args:
        df (pd.DataFrame): Dataframe with data
        outcome (str): Outcome name
        path (Path | str): Path to save
        col_id (list, optional): List of columns with ID and group. Defaults to ['ID', 'group'].
        outcome_years (list, optional): List with outcome years. Defaults to [0, 2].
        nulisa (bool, optional): If True, use script adapted to Nulisa project. Defaults to True.    
    """

    if nulisa:
        # list of columns
        col_c = []
        for y in outcome_years:
            col_c += [f'{outcome}_{y}y']

        # convert table with one column per year per cognitive score to one column for each cognitive score and one column "year" containing values 0, 1 or 2 
        df_long = df[col_id + col_c]
        df_cogn = df_long.melt(id_vars = col_id, value_vars = col_c,
                    var_name='year', value_name=outcome)
        df_cogn['year'] = df_cogn['year'].str.extract(r'_(\d+)y$').astype(int)
    else:
        df_cogn = df.copy()[['ID', outcome, 'visit', 'group']]
        df_cogn = df_cogn[df_cogn['visit'].isin(outcome_years)]
        df_cogn.rename(columns={'visit':'year'}, inplace = True)
    # boxplot cognitive score vs time vs group  
    sns.boxplot(x='year', y=outcome, hue='group', data=df_cogn)
    plt.title(f'{outcome} over time by Intervention group')
    plt.savefig(path/f'{outcome}_by_time.png')
    plt.close()


def get_lineplot_bygroup(df:pd.DataFrame, 
                         outcome:str, 
                         path:Path|str, 
                        outcome_years:list = [0, 2],
                        nulisa = True,
                        color_control = '#4D4D4D',
                        color_interv = '#CC79A7',
                         standardised:bool = False,
                         change:bool = True,
                         group:str = 'group'):
    """Get lineplot of outcome by group

    Args:
        df (pd.DataFrame): Dataframe with data
        outcome (str): Outcome name
        path (Path | str): Path to save
        col_id (list, optional): List of columns with ID and group. Defaults to ['ID', 'group'].
        outcome_years (list, optional): List with outcome years. Defaults to [0, 2].
        nulisa (bool, optional): If True, use script adapted to Nulisa project. Defaults to True.    
    """
    col_id = ['ID', group]
    add_z = '_z' if standardised else ''

    if group == 'adherence_class':
        df = df[df[group].isin(['high','control'])]
    if nulisa:
        if change:
            # list of columns
            col_c = []
            for y in outcome_years:
                col_c += [f'{outcome}_{y}y{add_z}']

            df_change = df[['ID',group]+col_c].copy()
            df_change[f'{outcome}_Change_0'] = 0
            if 1 in outcome_years:
                df_change[f'{outcome}_Change_1'] = df_change[f'{outcome}_1y{add_z}'] - df_change[f'{outcome}_0y{add_z}']
            df_change[f'{outcome}_Change_2'] = df_change[f'{outcome}_2y{add_z}'] - df_change[f'{outcome}_0y{add_z}']

            col_cchange =[f"{outcome}_Change_{i}" for i in outcome_years]


            df_change = df_change.melt(id_vars=col_id, value_vars=col_cchange,
                        var_name='year', value_name=outcome)
            df_change['year'] = df_change['year'].str.extract(r'_Change_(\d+)$').astype(int)
    
        else:
            # List of biomarker columns at each time point
            col_c = [f'{outcome}_{y}y{add_z}' for y in outcome_years]

            # Keep ID, group, and biomarker z-scores
            df_plot = df[['ID', group] + col_c].copy()

            # Reshape from wide to long format
            df_plot = df_plot.melt(
                id_vars=['ID', group],
                value_vars=col_c,
                var_name='year',
                value_name=outcome
            )

            # Extract year from column name
            df_plot['year'] = df_plot['year'].str.extract(r'_(\d+)y').astype(int)

    else:
        df_cogn = df.copy()[['ID', outcome, 'visit', group]]
        df_cogn = df_cogn[df_cogn['visit'].isin(outcome_years)]
        df_cogn.rename(columns={'visit':'year'}, inplace = True)
        # Pivot to wide form temporarily to compute differences easily
        df_wide = df_cogn.pivot(index=col_id, columns='year', values=outcome)
        
        # Compute changes relative to year 0
        df_change = pd.DataFrame(index=df_wide.index)
        for y in outcome_years:
            if y == 0:
                df_change[f'{outcome}_Change_{y}'] = 0
            else:
                if 0 in df_wide.columns and y in df_wide.columns:
                    df_change[f'{outcome}_Change_{y}'] = df_wide[y] - df_wide[0]
                else:
                    df_change[f'{outcome}_Change_{y}'] = pd.NA
        
        # Melt back to long format
        df_change = df_change.reset_index().melt(
            id_vars=col_id, 
            var_name='year', 
            value_name=outcome
        )
    palette = {
        "Control": color_control,
        "control": color_control,
        "Intervention": color_interv,
        "high": color_interv,
        "low": color_interv
    }

    if change:
        y_legend = f"{outcome} (normalized NPQ) change"
        sns.lineplot(x='year', y=outcome, hue=group, data=df_change, palette=palette)
        #plt.title(f'{outcome} over time by Intervention group')
        plt.ylabel(y_legend, fontsize=14, fontweight='bold')
        plt.xlabel('Time (years)', fontsize=12)

        plt.savefig(path/f'{outcome}_by_time_lineplot{add_z}.png', 
                    dpi=600,
                    bbox_inches='tight',
        )
    else:
        y_legend = f"{outcome} (normalized NPQ)"
        # Plot mean z-score over time
        sns.lineplot(
            x='year',
            y=outcome,
            hue=group,
            data=df_plot,
            palette=palette,
            estimator='mean',
            errorbar=('ci', 95)
        )

        plt.ylabel(y_legend, fontsize=14, fontweight='bold')
        plt.xlabel('Time (years)', fontsize=12)

        plt.savefig(
            path / f'{outcome}_by_time_lineplot{add_z}_mean.png',
            dpi=600,
            bbox_inches='tight'
        )
    plt.close()



def get_multicolinearity(df:pd.DataFrame, variables:list):
    """Get multicolinearity

    Args:
        df (pd.DataFrame): Dataframe
        variables (list): list of variables

    """
    df_set = df[variables].dropna()
    vif = pd.DataFrame()
    vif["feature"] = df_set.columns
    vif["VIF"] = [variance_inflation_factor(df_set.values, i) for i in range(df_set.shape[1])]

    vif.sort_values(by="VIF", inplace=True)
    print(vif)
    return vif

def get_cognitive_score_per_year(df:pd.DataFrame, 
                                 cogn_list:list, 
                                 path_options:str|Path, 
                                 name:str = 'cognitive', 
                                 year_1:bool = True,
                                 standardised:bool = False,
                                 ):
    """Get cognitive score per year

    Args:
        df (pd.DataFrame): Dataframe with data
        cogn_list (list): list of cognitive scores
        path_options (str | Path): Path to save
        name (str, optional): name for the file to save. Defaults to 'cognitive'.
        year_1 (bool, optional): If True, list of years is [0, 1, 2] else [0, 2]. Defaults to True.

    """
    add_z = '_z' if standardised else ''
    list_years = [0,1,2] if year_1 else [0, 2]

    col_desc = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    df_stat_year_cogn = pd.DataFrame(columns=[name, 'Group', 'year'] + col_desc + ['p-value'], index = np.arange(len(cogn_list)*3*(len(list_years)+1)))

    temp_index = 0
    for cogn in tqdm(cogn_list):
        df[f'{cogn}_Change'] = df[f'{cogn}_2y{add_z}'] - df[f'{cogn}_0y{add_z}']
        df_stat_year_cogn.loc[temp_index, name] = cogn
        df_stat_year_cogn.loc[temp_index, 'Group'] = 'Total'
        
        
        for year in list_years:
            df_stat_year_cogn.loc[temp_index, 'year'] = year
            if year != 1  or cogn != 'mmse_tot':
                df_stat_year_cogn.loc[temp_index,col_desc] = df[f'{cogn}_{year}y'].describe() 

        temp_index += 1
    df_stat_year_cogn.loc[temp_index, 'year'] = 'change_0_2'
    df_stat_year_cogn.loc[temp_index, col_desc] =df[f'{cogn}_Change'].describe() 
    
    group1 = df[df['group']=='Intervention'][f'{cogn}_Change'].dropna()
    group2 = df[df['group']=='Control'][f'{cogn}_Change'].dropna()

    if cogn == 'mmse_tot' or cogn == 'zabbr_memory_domain' or name == 'bbm':  
      _, p = mannwhitneyu(group1, group2, alternative='two-sided')
    else:
      _, p = ttest_ind(group1, group2, equal_var=False)  # Welch’s t-test (safer)
    
    df_stat_year_cogn.loc[temp_index, 'p-value'] = p

    temp_index += 1
    for group in ['Intervention', 'Control']:
      df_group = df[df['group']==group]
      df_stat_year_cogn.loc[temp_index, 'Group'] =  group
      
      for year in list_years:
        df_stat_year_cogn.loc[temp_index, 'year'] = year
        if year != 1  or cogn != 'mmse_tot':
          df_stat_year_cogn.loc[temp_index,col_desc] = df_group[f'{cogn}_{year}y'].describe() 

        temp_index += 1
      df_stat_year_cogn.loc[temp_index, 'year'] = 'change_0_2'
      df_stat_year_cogn.loc[temp_index, col_desc] =df_group[f'{cogn}_Change'].describe() 
      temp_index += 1
  
    for c in col_desc:
        df_stat_year_cogn[c] = round(df_stat_year_cogn[c].astype(float),4)
    df_stat_year_cogn['p-value'] = round(pd.to_numeric(df_stat_year_cogn['p-value']),4)

    df_stat_year_cogn.to_csv(path_options/f'{name}_stat_year{add_z}.csv', index=False)

    return df_stat_year_cogn


def mixedeffect_multiple_outcomes(df:pd.DataFrame,
                          outcomes_list:list,
                          predictor:str,
                          second_predictor:str|None = None,
                          cofactors:None|list = None,
                          quadratic:bool = False,
                          cofactors_choice:str = 'basic',
                          outcome_years:list = [0, 1, 2],
                          predictor_years:list|None = None,
                          name_file:str = 'bbm',
                          horizontal:bool = True,
                          path:str|Path = './mixed_effect', 
                          group_ref:str = 'Control'
                          ):
    """Run Mixed effect models for several outcomes

    Args:
        df (pd.DataFrame): Dataframe with data
        outcomes_list (list): List of outcomes
        predictor (str): predictor
        second_predictor (str | None, optional): Second predictor. Defaults to None.
        cofactors (None | list, optional): List of cofactors. Defaults to None.
        quadratic (bool, optional): If True, non-linear model. Defaults to False.
        cofactors_choice (str, optional): Cofactors version. Defaults to 'basic'.
        outcome_years (list, optional): List of years for the outcome. Defaults to [0, 1, 2].
        predictor_years (list | None, optional): list of years for the predictor. Defaults to None.
        name_file (str, optional): File name. Defaults to 'bbm'.
        horizontal (bool, optional): If True, use data in the format with one row with all visits, else one row per visit. Defaults to True.
        path (str | Path, optional): Path to save. Defaults to './mixed_effect'.
        group_ref (str, optional): Reference for group. Defaults to 'Control'.

    """
    if second_predictor:
        if quadratic:        
            col_table = [second_predictor, predictor,  f'{second_predictor}:{predictor}', 'year',f'{second_predictor}:year', f'{predictor}:year',  f'{second_predictor}:{predictor}:year', 'year2', f'{second_predictor}:year2', f'{predictor}:year2',  f'{second_predictor}:{predictor}:year2' ]
        else:        
            col_table = [second_predictor, predictor,  f'{second_predictor}:{predictor}', 'year',f'{second_predictor}:year', f'{predictor}:year',  f'{second_predictor}:{predictor}:year']
    else:
        if quadratic:                    
            col_table = [predictor, 'year',  f'{predictor}:year', 'year2',  f'{predictor}:year2']
        else:
            col_table = [predictor, 'year',  f'{predictor}:year']

    col_desc = ['Intercept coeff','Intercept std']
    for c in col_table:
        col_desc += [f"{c} coeff", f"{c} std", f"{c} pvalue"]

    df_mixedeffect= pd.DataFrame(index = outcomes_list, columns=col_desc)

    for bm_name in tqdm(outcomes_list):
        df_long = df.copy()
        if horizontal:
            df_long = df_long.dropna(subset=[f"{bm_name}_0y"])
        else:
            df_long = df_long[['ID','group', 'visit',bm_name] + cofactors]

        if bm_name in ['sex','diabetes_ever_no_0y']:
            df_long[bm_name] = df_long[bm_name].astype("category")

        if cofactors:
            list_table_coeff = ['Intercept'] + col_table + cofactors
        else:
            list_table_coeff = ['Intercept'] + col_table
        
        if horizontal:
            result_2, _ = mixlm_indiv_hor(df_input = df_long, 
                                            col_id = ['ID'], 
                                            predictor = predictor, 
                                            second_predictor = second_predictor, 
                                            outcome = bm_name, 
                                            save_status = False, 
                                            cofactors = cofactors, 
                                            verbose = False, 
                                            quadratic = quadratic,
                                            cofactors_choice = cofactors_choice,
                                            outcome_years = outcome_years,
                                            predictor_years = predictor_years,
                                            group_ref = group_ref,
                                            )
        else:
            result_2, _ = mixlm_indiv_vert(df_input = df_long, 
                                             col_id = ['ID'], 
                                             predictor = predictor, 
                                             second_predictor = second_predictor, 
                                             outcome = bm_name, 
                                             save_status = False, 
                                             cofactors = cofactors, 
                                             verbose = False, 
                                             quadratic = quadratic,
                                             cofactors_choice = cofactors_choice,
                                             outcome_years = outcome_years,
                                             predictor_years = predictor_years,
                                             )
        
        df_mixedeffect.loc[bm_name,[f'{c} coeff' for c in list_table_coeff]] = result_2.fe_params.values.reshape(1, -1)[0,0:len(list_table_coeff)]
        df_mixedeffect.loc[bm_name,[f'{c} std' for c in ['Intercept'] + col_table]] = result_2.bse_fe.values.reshape(1, -1)[0,0:len(['Intercept'] + col_table)]
        df_mixedeffect.loc[bm_name,[f'{c} pvalue' for c in col_table]] = result_2.pvalues.values.reshape(1, -1)[0,1:len(['Intercept'] + col_table)]
    
    for c in col_table:
        _, df_mixedeffect[f'{c} pvalue FDR'], _, _ = multipletests(df_mixedeffect[f'{c} pvalue'], alpha=0.1, method='fdr_bh')

    for c in col_desc:
        df_mixedeffect[c] = round(df_mixedeffect[c].astype(float), 4)

    # definition of the file name according to options
    txt_second = f'_{second_predictor}' if second_predictor else ''
    if txt_second == '_group' and group_ref == 'Intervention':
        txt_second += '_inter'
    txt_quadra = f'_quadra' if quadratic else ''

    if predictor_years:
        if len(predictor_years > 1):
            txt_evol = '_change'
        else:
            txt_evol = f"_baseline" 
    else:
        txt_evol = ''
    txt_predictor = predictor
    if txt_predictor == 'group' and group_ref == 'Intervention':
        txt_predictor += '_inter'


    df_mixedeffect.to_csv(path/f'{name_file}_change_mixed_model_{txt_predictor}{txt_evol}{txt_second}{txt_quadra}.csv')

    return df_mixedeffect

def mixedeffect_multiple_predictors(df:pd.DataFrame,
                          outcome:str,
                          predictors_list:list,
                          second_predictor:str|None = None,
                          cofactors:None|list = None,
                          quadratic:bool = False,
                          cofactors_choice:str = 'basic',
                          outcome_years:list = [0, 1, 2],
                          predictor_years:list|None = None,
                          name_file:str = 'bbm',
                          horizontal:bool = True,
                          path:str|Path = './mixed_effect', 
                          group_ref:str = 'Control'
                          ):
    """Run Mixed effect models for several predictors

    Args:
        df (pd.DataFrame): Dataframe with data
        outcomes_list (str): Outcome
        predictor (list): List of predictors
        second_predictor (str | None, optional): Second predictor. Defaults to None.
        cofactors (None | list, optional): List of cofactors. Defaults to None.
        quadratic (bool, optional): If True, non-linear model. Defaults to False.
        cofactors_choice (str, optional): Cofactors version. Defaults to 'basic'.
        outcome_years (list, optional): List of years for the outcome. Defaults to [0, 1, 2].
        predictor_years (list | None, optional): list of years for the predictor. Defaults to None.
        name_file (str, optional): File name. Defaults to 'bbm'.
        horizontal (bool, optional): If True, use data in the format with one row with all visits, else one row per visit. Defaults to True.
        path (str | Path, optional): Path to save. Defaults to './mixed_effect'.
        group_ref (str, optional): Reference for group. Defaults to 'Control'.

    """
    if second_predictor:
        if quadratic:        
            col_table = [second_predictor, 'bm',  f'{second_predictor}:bm', 'year',f'{second_predictor}:year', 'bm:year',  f'{second_predictor}:bm:year', 'year2', f'{second_predictor}:year2', 'bm:year2',  f'{second_predictor}:bm:year2' ]
        else:        
            col_table = [second_predictor, 'bm',  f'{second_predictor}:bm', 'year',f'{second_predictor}:year', 'bm:year',  f'{second_predictor}:bm:year']
    else:
        if quadratic:                    
            col_table = ['bm', 'year',  'bm:year', 'year2',  'bm:year2']
        else:
            col_table = ['bm', 'year',  'bm:year']

    col_desc = ['Intercept coeff','Intercept std']
    for c in col_table:
        col_desc += [f"{c} coeff", f"{c} std", f"{c} pvalue"]

    df_mixedeffect= pd.DataFrame(index = predictors_list, columns=col_desc)

    for bm_name in tqdm(predictors_list):
        df_long = df.copy()
        if horizontal:
            if predictor_years:
                df_long = df_long.dropna(subset=[f"{bm_name}_{predictor_years[0]}y"])
            else:
                df_long = df_long.dropna(subset=[bm_name])
        else:
            df_long = df_long[['ID','group', 'visit', bm_name, outcome] + cofactors]
            df_long.columns = df_long.columns.astype(str)

        if bm_name in ['sex','diabetes_ever_no_0y']:
            df_long[bm_name] = df_long[bm_name].astype("category")

        if cofactors:
            list_table_coeff = ['Intercept'] + col_table + cofactors
        else:
            list_table_coeff = ['Intercept'] + col_table
        
        if horizontal:
            result_2, _ = mixlm_indiv_hor(df_input = df_long, 
                                            col_id = ['ID'], 
                                            predictor = bm_name, 
                                            second_predictor = second_predictor, 
                                            outcome = outcome, 
                                            save_status = False, 
                                            cofactors = cofactors, 
                                            verbose = False, 
                                            quadratic = quadratic,
                                            cofactors_choice = cofactors_choice,
                                            outcome_years = outcome_years,
                                            predictor_years = predictor_years, 
                                            group_ref = group_ref,
                                            )
        else:
            result_2, _ = mixlm_indiv_vert(df_input = df_long, 
                                             col_id = ['ID'], 
                                             predictor = bm_name, 
                                             second_predictor = second_predictor, 
                                             outcome = outcome, 
                                             save_status = False, 
                                             cofactors = cofactors, 
                                             verbose = False, 
                                             quadratic = quadratic,
                                             cofactors_choice = cofactors_choice,
                                             outcome_years = outcome_years,
                                             predictor_years = predictor_years,
                                             )
        
        df_mixedeffect.loc[bm_name,[f'{c} coeff' for c in list_table_coeff]] = result_2.fe_params.values.reshape(1, -1)[0,0:len(list_table_coeff)]
        df_mixedeffect.loc[bm_name,[f'{c} std' for c in ['Intercept'] + col_table]] = result_2.bse_fe.values.reshape(1, -1)[0,0:len(['Intercept'] + col_table)]
        df_mixedeffect.loc[bm_name,[f'{c} pvalue' for c in col_table]] = result_2.pvalues.values.reshape(1, -1)[0,1:len(['Intercept'] + col_table)]
    
    for c in col_table:
        _, df_mixedeffect[f'{c} pvalue FDR'], _, _ = multipletests(df_mixedeffect[f'{c} pvalue'], alpha=0.1, method='fdr_bh')

    for c in col_desc:
        df_mixedeffect[c] = round(df_mixedeffect[c].astype(float), 4)

    # definition of the file name according to options
    txt_second = f'_{second_predictor}' if second_predictor else ''
    if txt_second == '_group' and group_ref == 'Intervention':
        txt_second += '_inter'
    txt_quadra = f'_quadra' if quadratic else ''

    if predictor_years:
        if len(predictor_years) > 1:
            txt_evol = '_change'
        else:
            txt_evol = f"_baseline" 
    else:
        txt_evol = ''

    df_mixedeffect.to_csv(path/f'{outcome}_change_mixed_model_{name_file}{txt_evol}{txt_second}{txt_quadra}.csv')

    return df_mixedeffect

def mixlm_indiv_hor(df_input:pd.DataFrame, 
                    col_id:list = ['ID'], 
                    predictor:str = 'group', 
                    second_predictor = None,
                    outcome:str = 'ztotal_composite',
                    cofactors = None,
                    save_status:bool = True,
                    verbose:bool = False,
                    path_options:str|Path = './mixed_effect/', 
                    quadratic:bool = False,
                    cofactors_choice:str = 'basic',
                    outcome_years:list = [0, 1, 2],
                    predictor_years:list|None = None, 
                    group_ref:str = 'Control',
                    vertical:bool = False):
    """
    Vertical = False: If no visit column and one column per year per cogntion and proteins
    Vertical = True:  If visit column and only one column for cogntion, and one column for each protein

    formula = "outcome ~ predictor * year  * second_predictor + cofactors"

    Args:        
            df (pd.DataFrame): dataframe with protein to analyse
            col_id (list): list of columns used for patient identification. Default: ['ID']
            predictor (str, optional): Predictor. Defaults to 'group'.
            second_predictor (_type_, optional): Second predictor. Defaults to None.
            outcome (str, optional): _description_. Defaults to 'ztotal_composite'.
            cofactors (_type_, optional): _description_. Defaults to None.
            save_status (bool, optional): _description_. Defaults to True.
    """
    df = df_input.copy()
    if vertical:    
        df.rename(columns={'visit':'year'}, inplace=True)

    os.makedirs(path_options, exist_ok=True)

    if 'group' in df.columns:
        if group_ref == 'Control':
            df['group'] = pd.Categorical(df['group'], categories=['Control', 'Intervention'], ordered=True)
        else:
            df['group'] = pd.Categorical(df['group'], categories=['Intervention', 'Control'], ordered=True)

    outcome_list = []
    for i in outcome_years:
        outcome_list.append(f'{outcome}_{i}y')

    predictor_list = []
    if predictor_years:
        for i in predictor_years:
            predictor_list.append(f'{predictor}_{i}y')
    else:
        predictor_list.append(predictor)

    column_list = col_id.copy()

    if len(predictor_list) == 1:
        column_list += predictor_list

    if second_predictor :
        column_list.append(second_predictor)

    df = df.dropna(subset = [f'{outcome}_0y'] + column_list )
    
    if len(predictor_list) > 1:
        df = df.dropna(subset = [f'{predictor}_0y'] )
            
    df_outcome = df[[col_id[0]] + outcome_list].dropna()

    df_outcome = df_outcome.melt(id_vars=col_id, value_vars=outcome_list,
                    var_name=f'year', value_name=outcome)
    df_outcome['year'] = df_outcome['year'].str.extract(r'_(\d+)y$').astype(int)


    if len(predictor_list) > 1:
        df_predictor = df[[col_id[0]] + predictor_list].dropna()
        df_predictor = df_predictor.melt(id_vars=col_id, value_vars=predictor_list,
                    var_name=f'year', value_name=predictor)
        df_predictor['year'] = df_predictor['year'].str.extract(r'_(\d+)y$').astype(int)
    
        df_outcome = pd.merge(df_outcome, df_predictor, on=[col_id[0],'year'], how = 'outer')
        
    if len(predictor_list) > 1:
        ids_to_remove = df_outcome[(df_outcome[predictor].isna()) & (df_outcome['year'] == 0)][col_id[0]].unique()
        df_outcome = df_outcome[~df_outcome[col_id[0]].isin(ids_to_remove)]
        df_outcome = df_outcome.dropna(subset = [outcome, predictor] )

    if cofactors is None:
        df_outcome = pd.merge(df_outcome, df[column_list], on=[col_id[0]], how='left')
    else:
        df_outcome = pd.merge(df_outcome, df[column_list + cofactors], on=[col_id[0]], how='left')
        df_outcome = df_outcome.dropna(subset = cofactors )
        


    if predictor_years and  len(predictor_years) == 1:
        predictor_name = f"{predictor}_{predictor_years[0]}y"
    else:
        predictor_name = predictor

    formula = f"{outcome} ~ {predictor_name}"

    if quadratic:
        df_outcome['year_centered'] = (df_outcome['year'] - df_outcome['year'].mean()) 
        formula += " * year_centered"
    else:    
        formula += " * year"
        
    if second_predictor:
        formula += f" * {second_predictor}"

    
    if quadratic:
        df_outcome['year_centered'] = df_outcome['year'] - df_outcome['year'].mean()  # optional, helps reduce multicollinearity
        df_outcome['year_squared'] = df_outcome['year_centered'] ** 2

        formula += " + year_squared"
        formula = (
        f"{formula} + "
        f" {predictor_name}:year_squared"
        )
        if second_predictor:
            formula = (
                f"{formula} + "
                f" {second_predictor}:year_squared + {predictor_name}:{second_predictor}:year_squared"
                ) 

    if cofactors is not None:
        for co in cofactors:
            formula += f" + {co}"

    if quadratic:
        model = smf.mixedlm(formula, df_outcome, groups=df_outcome["ID"], re_formula="~year_centered")
    else:   
        model = smf.mixedlm(formula, df_outcome, groups=df_outcome["ID"], re_formula="~year")
    result = model.fit()

    if verbose:
        print(result.summary())

    summary_df = pd.DataFrame({
        "coef": result.params,
        "stderr": result.bse,
        "z": result.tvalues,
        "pvalue": result.pvalues,
        "ci_lower": result.conf_int()[0],
        "ci_upper": result.conf_int()[1]
    })
    if predictor_years:
        if len(predictor_years) > 1:
            text_evol = '_change'
        else:
            text_evol = f"_baseline" 
    else:
            text_evol = ''

    text_quadratic = '_quadra' if quadratic else ''

    text_cofactors = f"_cofactors_{cofactors_choice}" if cofactors is not None else ''
    
    txt_second = f'_{second_predictor}' if second_predictor else ''
    if txt_second == '_group' and group_ref == 'Intervention':
        txt_second += '_inter'

    txt_predictor = predictor
    if txt_predictor == 'group' and group_ref == 'Intervention':
        txt_predictor += '_inter'

    if save_status:
        summary_df.to_csv(path_options/f'{outcome}_mixed_effects_{txt_predictor}{text_evol}{txt_second}{text_cofactors}{text_quadratic}.csv', index=True)

    return result, summary_df

def mixlm_indiv_vert(df_input:pd.DataFrame, 
                    col_id:list = ['ID'], 
                    predictor:str = 'group', 
                    second_predictor = None,
                    outcome:str = 'ztotal_composite',
                    cofactors = None,
                    save_status:bool = True,
                    verbose:bool = False,
                    quadratic:bool = False,
                    path_options:str|Path = './mixed_effect/', 
                    cofactors_choice:str = 'basic',
                    outcome_years:list = [0, 1, 2],
                    predictor_years:list|None = None, 
                    group_ref:str = 'Control'
                    ):
    """
    If visit column and only one column for cogntion, and one column for each protein

    formula = "outcome ~ predictor * year  * second_predictor + cofactors"

    Args:
        df (pd.DataFrame): dataframe with protein to analyse
        col_id (list): list of columns used for patient identification. Default: ['ID']
        predictor (str, optional): Predictor. Defaults to 'group'.
        second_predictor (_type_, optional): Second predictor. Defaults to None.
        outcome (str, optional): _description_. Defaults to 'ztotal_composite'.
        cofactors (_type_, optional): _description_. Defaults to None.
        save_status (bool, optional): _description_. Defaults to True.
    """
    os.makedirs(path_options, exist_ok=True)

    dft = df_input.copy()

    if 'group' in dft.columns:
        if group_ref == 'Control':
            dft['group'] = pd.Categorical(dft['group'], categories=['Control', 'Intervention'], ordered=True)
        else:
            dft['group'] = pd.Categorical(dft['group'], categories=['Intervention', 'Control'], ordered=True)

    dft.rename(columns={'visit':'year'}, inplace=True)

    all_col = col_id.copy()

    if second_predictor :
        all_col.append(second_predictor)

    if cofactors :
        all_col += cofactors

    if predictor == 'group': 
        all_col.append('group')

    df_temp = dft[all_col]

    df_outcome = dft[['ID',outcome,'year']]
    df_outcome = df_outcome[df_outcome['year'].isin(outcome_years)]

    if predictor != 'group':
        df_pred = dft[['ID',predictor,'year']]
        df_pred = df_pred[df_pred['year'].isin(predictor_years)]
        if len(predictor_years) == 1:
            df_pred = df_pred[['ID',predictor]]
            df_merge = pd.merge(df_outcome, df_pred, on=['ID'], how = 'outer')
        else:
            df_merge = pd.merge(df_outcome, df_pred, on=['ID','year'], how = 'outer')

        df_merge = pd.merge(df_merge, df_temp, on=['ID'], how = 'outer')
    else:
        df_merge = pd.merge(df_outcome, df_temp, on=['ID'], how = 'outer')

    df_merge = df_merge.dropna(subset = all_col)

    if (predictor_years != None) and (len(predictor_years) == 1):
        df_merge = df_merge.dropna(subset = [predictor])
        
    formula = f"Q('{outcome}') ~ Q('{predictor}')"

    if quadratic:
        df_merge['year_centered'] = (df_merge['year'] - df_merge['year'].mean()) 
        formula += " * year_centered"
    else:    
        formula += " * year"
        
    if second_predictor:
        formula += f" * {second_predictor}"

    
    if quadratic:
        df_merge['year_centered'] = df_merge['year'] - df_merge['year'].mean()  # optional, helps reduce multicollinearity
        df_merge['year_squared'] = df_merge['year_centered'] ** 2

        formula += " + year_squared"
        formula = (
        f"{formula} + "
        f" Q('{predictor}'):year_squared"
        )
        if second_predictor:
            formula = (
                f"{formula} + "
                f" {second_predictor}:year_squared + Q('{predictor}'):{second_predictor}:year_squared"
                ) 

    if cofactors is not None:
        for co in cofactors:
            formula += f" + {co}"

    if quadratic:
        model = smf.mixedlm(formula, df_merge, groups=df_merge["ID"], re_formula="~year_centered")
    else:   
        model = smf.mixedlm(formula, df_merge, groups=df_merge["ID"], re_formula="~year")
    result = model.fit()

    if verbose:
        print(result.summary())

    summary_df = pd.DataFrame({
        "coef": result.params,
        "stderr": result.bse,
        "z": result.tvalues,
        "pvalue": result.pvalues,
        "ci_lower": result.conf_int()[0],
        "ci_upper": result.conf_int()[1]
    })
    if (predictor_years != None) and len(predictor_years)>1:
        text_evol = '_evol' 
    elif predictor != 'group':
        text_evol = '_baseline'
    else: 
        text_evol = ''
    
    text_quadratic = '_quadra' if quadratic else ''

    if save_status:
        if cofactors is None:
            summary_df.to_csv(path_options/f'{outcome}_mixed_effects_{predictor}{text_evol}_{second_predictor}{text_quadratic}_{len(outcome_years)}points.csv', index=True)
        else:
            summary_df.to_csv(path_options/f'{outcome}_mixed_effects_{predictor}{text_evol}_{second_predictor}_cofactors_{cofactors_choice}{text_quadratic}_{len(outcome_years)}points.csv', index=True)

    return result, summary_df



def get_symbol_and_color(coeff:float, qval:float, col:str):
    """Get symbol for the resume shema

    Args:
        coeff (float): coefficient
        qval (float): pvalue or qvalue
        col (str): column name

    """
    if col == 'rank_causal_forest':
        return None, None, None  # Handled separately
    else:
        if coeff == 0:
            return '■', 'gray', 'gray'
        elif coeff > 0:
            if qval <= 0.05:
                return '▲', 'green', 'green'  # Text color matches symbol color
            elif qval <= 0.1:
                return '▲', 'lightgreen', 'lightgreen'
            elif qval > 0.1:
                return '▲', 'gray', 'gray'
            else:
                return '-', 'gray', 'gray'
            
        elif coeff < 0:  # coeff < 0
            if qval <= 0.05:
                return '▼', 'red', 'red'  # Text color matches symbol color
            elif qval <= 0.1:
                return '▼', 'salmon', 'salmon'
            elif qval > 0.1:
                return '▼', 'gray', 'gray'
            else:
                return '-', 'gray', 'gray'
        else:
            return '-', 'gray', 'gray'
    
def create_schema(cognition, 
                  columns_list, 
                  selected_bm_target, 
                  df_table_coeff, 
                  df_table_pval, 
                  path_options_output, 
                  list_cognition, 
                  other_parts, 
                  forest, 
                  forest_order, 
                  feature_selection, 
                  pval_type = 'pval', 
                  pval:float = 0.1, 
                  nulisa = True, 
                  intervention = True):
    dict_columns_list_text = {
                'low_bbm_cognition_intervention':'Effect of intervention on z total\nwhen low baseline bbm',
                'high_bbm_cognition_intervention':'Effect of intervention on z total\nwhen high baseline bbm',
                'bbm_change_vs_intervention':'Association between bbm change\nand intervention', 
                f'bbm_baseline_vs_{cognition}_change_full_population':'Effect of bbm baseline\non z total change(full population)', 
                f'bbm_baseline_vs_{cognition}_change_control':'Effect of bbm baseline\non z total change(control)', 
                f'bbm_baseline_vs_{cognition}_change_intervention':'Effect of bbm baseline\non z total change(inter)', 
                f'bbm_baseline_vs_{cognition}_diff_groups':'Difference of effect of\nbbm baseline on z total change\n in intervention group', 
                f'bbm_baseline_vs_{cognition}_curve_change_full_population':'Effect of bbm baseline\non z total curve change (full population)', 
                f'bbm_baseline_vs_{cognition}_curve_change_control':'Effect of bbm baseline\non z total curve change (control)',  
                f'bbm_baseline_vs_{cognition}_curve_change_intervention':'Effect of bbm baseline\non z total curve change (inter)', 
                f'bbm_baseline_vs_{cognition}_curve_diff_groups':'Difference of effect of\nbbm baseline on z total curve change\n in intervention group', 
                
                }
    if forest:
        dict_columns_list_text['effect_causal_forest_low_val'] = 'Effect of low bbm baseline\nlevel in causal forest'
        dict_columns_list_text['effect_causal_forest_high_val'] = 'Effect of high bbm baseline\nlevel in causal forest'
        dict_columns_list_text['rank_causal_forest'] = 'Rank in causal forest'
        
    
        
    if other_parts:
        for cogn in list_cognition:
            if cogn != cognition:
                dict_columns_list_text[f'bbm_baseline_vs_{cogn}_change_full_population'] = f'Effect of bbm baseline\non {cogn} (full population)'
                dict_columns_list_text[f'bbm_baseline_vs_{cogn}_change_control'] = f'Effect of bbm baseline\non {cogn} (control)'
                dict_columns_list_text[f'bbm_baseline_vs_{cogn}_diff_groups'] = f'Difference of effect of\nbbm baseline on {cogn}\n in intervention group'

    if not intervention:
        del dict_columns_list_text[f'bbm_baseline_vs_{cognition}_change_intervention']
        del dict_columns_list_text[f'bbm_baseline_vs_{cognition}_curve_change_intervention']


    # if coeff positive, qval <= 0.1 -> triangle pointing up green  
    # if coeff negative, qval <= 0.1  -> triangle pointing down red
    # if coeff positive, qval <= 0.05 -> triangle pointing up green with bold edges
    # if coeff negative, qval <= 0.05  -> triangle pointing down red with bold edges
    # if coeff positive, qval > 0.1 -> triangle pointing up gray 
    # if coeff negative, qval > 0.1  -> triangle pointing down gray
    # if coeff at 0  -> square gray

    # shema resuming effects of proteins on cognition change and interaction with intervention


    # Create table data with symbols
    table_data = []
    for bbm in selected_bm_target:
        row = [bbm]
        for col in columns_list:
            if col == 'rank_causal_forest':
                row.append(df_table_coeff.loc[bbm, 'rank_causal_forest'])
            else:
                coeff = df_table_coeff.loc[bbm, col]
                qval = df_table_pval.loc[bbm, col]
                symbol, _, _ = get_symbol_and_color(coeff, qval, col)
                row.append(symbol if symbol is not None else '')
        table_data.append(row)

    # Add column headers
    table_data.insert(0, ['BBM'] + list(dict_columns_list_text.values()))

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Hide axes
    ax.axis('off')

    # Create the table
    table = ax.table(
        cellText=table_data,
        cellLoc='center',
        rowLoc='center',
        loc='center',
        edges='open'  # Remove table edges
    )

    table.scale(1, 2)  # Scale rows vertically by 1.5x
    # Set colors and font weight for each cell
    for i, row in enumerate(table_data[1:], start=1):
        for j, cell in enumerate(row[1:], start=1):
            col = list(dict_columns_list_text.keys())[j-1]
            if col == 'rank_causal_forest':
                table[i, j].set_facecolor('white')
                table[i, j].get_text().set_color('black')
            else:
                _, bg_color, text_color = get_symbol_and_color(
                    df_table_coeff.loc[row[0], list(dict_columns_list_text.keys())[j-1]],
                    df_table_pval.loc[row[0], list(dict_columns_list_text.keys())[j-1]], col
                )
                table[i, j].set_facecolor(bg_color)
                table[i, j].get_text().set_color(text_color)
                table[i, j].get_text().set_fontweight('bold')

    # Set header colors and font weight
    for j, col in enumerate(list(dict_columns_list_text.values()), start=1):
        table[0, j].get_text().set_fontweight('bold')
        table[0, j].set_facecolor('lightgray')
        table[0, j].get_text().set_color('black')
    table[0, 0].get_text().set_fontweight('bold')
    table[0, 0].set_facecolor('lightgray')
    table[0, 0].get_text().set_color('black')

    # Auto-scale columns
    table.auto_set_column_width(col=list(range(len(list(dict_columns_list_text.values())) + 1)))

    #plt.subplots_adjust(bottom=0.2)  # add space at bottom for legend

    #plt.title('Protein Effects Schema')
    plt.savefig(path_options_output/f'bbm_effects_schema_featselect_forestorder_{forest_order}_{feature_selection}_forest{forest}_otherparts_{other_parts}_{len(selected_bm_target)}_{str(pval)[2:]}_{pval_type}.png', bbox_inches='tight', dpi=300)
    plt.show()


    # Define legend handles
    legend_handles = [        
        # Positive coeff, qval <= 0.05 (bold edges)
        mlines.Line2D([], [], color='green', marker='^', linestyle='None', markersize=10,  label=f'Coefficient > 0, {pval_type} ≤ 0.05'),
        # Positive coeff, qval <= 0.1
        mlines.Line2D([], [], color='lightgreen', marker='^', linestyle='None', markersize=10, label=f'Coefficient > 0, {pval_type} ≤ 0.1'),
        # Positive coeff, qval > 0.1
        mlines.Line2D([], [], color='gray', marker='^', linestyle='None', markersize=10, label=f'Coefficient > 0, {pval_type} > 0.1'),
        
        # Negative coeff, qval <= 0.1
        mlines.Line2D([], [], color='salmon', marker='v', linestyle='None', markersize=10, label=f'Coefficient < 0, {pval_type} ≤ 0.1'),
        # Negative coeff, qval <= 0.05 (bold edges)
        mlines.Line2D([], [], color='red', marker='v', linestyle='None', markersize=10, label=f'Coefficient < 0, {pval_type} ≤ 0.05'),
        # Negative coeff, qval > 0.1
        mlines.Line2D([], [], color='gray', marker='v', linestyle='None', markersize=10, label=f'Coefficient < 0, {pval_type} > 0.1'),
        # Coeff at 0
        mlines.Line2D([], [], color='gray', marker='s', linestyle='None', markersize=10, label='Coefficient = 0'),
    ]

    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.axis('off')    
    leg = plt.legend(handles=legend_handles, loc='center',)
    
    fig.add_artist(leg)
    plt.savefig(path_options_output/f'bbm_effects_schema_legend_{pval_type}.png', bbox_inches='tight', dpi=300)
    plt.show()
