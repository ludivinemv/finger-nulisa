##############################
# Purpose: Elastic Net

# Date creation: 2025-10-18
# Author: ludivinemv

# Input data: 
#       dataset_original.csv

#Output data:

##############################


#! PUBLIC IMPORTS

import os, sys, ast
import logging, re
from tqdm import tqdm
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm


from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RandomizedSearchCV
from scipy.stats import loguniform, uniform


#! PRIVATE IMPORTS
from utils.log_functions import setup_logger
from utils.data_transform import windsorize_column, clean_open_file


#! GLOBAL VARIABLES
from projectspecific.config import LOG_PATH, DATA_PROCESSED_PATH, LONG_PATH, DATA_PROCESSED_PATH

VERSION = "2-0"
GROUP_CLASS = "intervention"


#! FUNCTIONS


def get_data():
    """ - Open data and meta data
        - standardize column names
        - find patients from mITT subgroup
    Return 
        df: Standardized data
        meta: meta data
    """
    # open meta and data files
    df = clean_open_file(path=DATA_PROCESSED_PATH/'dataset_original.csv', output_type='dataframe')
    meta = clean_open_file(path=DATA_PROCESSED_PATH/'description'/'columns_description.csv', output_type='dataframe')
    
    # Get the string of a dictionary and convert it to a dictionary
    label_map = ast.literal_eval(meta[meta['column']=='sub_sample_id']['values'].values[0])
    # get Intervention and Control instead of 2 and 3 in df['group'] 
    df['group'] = df['sub_sample_id'].replace(label_map)
    
    # standardize cognition columns names
    col_cognitive = [item for item in df.columns if item.lower().startswith('z')]
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

    for c in ["nulisa_APOE4", "nulisa_CHIT1", "nulisa_GDF15"]:
        # Remove {c}_0y and {c}_2y
        df.drop(columns=[f"{c}_0y", f"{c}_2y"], inplace=True, errors='ignore')

        # Rename {c}_0_halfmin to {c}_0y and {c}_2_halfmin to {c}_2y
        df.rename(columns={
            f"{c}_0_halfmin": f"{c}_0y",
            f"{c}_2_halfmin": f"{c}_2y"
        }, inplace=True, errors='ignore')

    df['diabetes_0y'] = np.where(df['diabetes_ever_no_0y'] == 1, 0, 1)
    df = df[df['mITT']==1]

    #! Lists ##########################################################
    # basic cofactors
    col_cofacteurs_base = ['age_0y','sex','educ_years_0y','campid']

    # cofactors to evaluate
    cofactors_to_test = ['sex',
                         'age_0y',
                         'educ_years_0y',
                        'bmi_0y',
                        'campid',
                ]
    
    # List of Nulisa biomarkers columns
    col_nulisa = [item for item in df.columns if item.lower().startswith('nulisa')]


    #! Windsorize data
    for col in tqdm(col_nulisa):
        df[col] = windsorize_column(df[col], windsor_ratio/100, windsor_ratio/100)



    col_nulisa_baseline = [s for s in col_nulisa if s.endswith('_0y')]

    return df, col_cofacteurs_base, cofactors_to_test, col_nulisa, col_nulisa_baseline





def scaling(X, feature_names):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    return X_scaled

def get_graphs(
    X_scaled, y, feature_names, path_options_output_model, group,
    train_dict=None, test_dict=None
    ):
    # --- Color setup ---
    tab20 = cm.get_cmap('tab20', 20)
    tab20b = cm.get_cmap('tab20b', 20)
    tab20c = cm.get_cmap('tab20c', 20)
    colors_tab20 = [tab20(i) for i in range(20)]
    colors_tab20b = [tab20b(i) for i in range(20)]
    colors_tab20c = [tab20c(i) for i in range(20)]
    all_colors = colors_tab20 + colors_tab20b + colors_tab20c
    custom_colors = all_colors[:25]


    # --- Open outputs ---
    with open(path_options_output_model /group/ f'elasticnet_all_folds_outputs_{group}.pkl', 'rb') as f:
        fold_outputs = pickle.load(f)
    

    metrics_df = pd.read_csv(path_options_output_model / group/f'summary_metrics_all_folds_{group}.csv')
    metrics_all_folds = metrics_df.set_index("Fold").to_dict("index")


    # --- Initialize outputs ---
    
    feature_coef_ranges = {}
    feature_rank_ranges = {}

    # --- Loop over folds ---
    for fold in range(n_folds):
        print(f"\n--- Processing Fold {fold} ---")

        with open(path_options_output_model /group/ f'random_search_fold{fold}_{group}.pkl', 'rb') as f:
            random_search = pickle.load(f)
        results = pd.DataFrame(random_search.cv_results_)

        best_model = fold_outputs[fold]['best_model']
        coef_table = fold_outputs[fold]['coef_table']
        coef_table_all = fold_outputs[fold]['coef_table_all']
        shrink_alpha_df = fold_outputs[fold]['shrink_alpha']
       
        best_alpha = best_model.alpha



        # --- Plot 1: Train and Validation MSE vs. Alpha (from CV) ---
        plt.figure(figsize=(10, 6))
        plt.scatter(
            np.log10(results['param_alpha']),
            -results['mean_train_score'],
            alpha=0.5, color='blue', label='Train MSE (CV)'
        )
        plt.scatter(
            np.log10(results['param_alpha']),
            -results['mean_test_score'],
            alpha=0.5, color='red', label='Validation MSE (CV)'
        )
        plt.scatter(
            np.log10(best_alpha), -random_search.best_score_,
            color='green', s=100, label=f'Best Model (alpha={best_alpha:.2e})'
        )
        plt.xlabel('log10(alpha)')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title(f'Fold {fold}: Train vs Validation Average MSE vs. Alpha (CV)')
        plt.grid(True)
        plt.legend()
        plt.savefig(path_options_output_model /group/ f'fold_{fold}_cv_mse_vs_alpha_{group}.png', bbox_inches='tight')

        plt.savefig(
            path_options_output_model / group / f'fold_{fold}_cv_mse_vs_alpha_{group}.svg',
            bbox_inches='tight'
        )
        plt.close()



        # Load data for this fold
        train_ids = train_dict[fold]
        test_ids = test_dict[fold]
        X_train = pd.merge(X_scaled,train_ids, how = 'right')
        X_train, y_train = X_scaled[X_scaled['ID'].isin(list(train_ids))], y[y['ID'].isin(train_ids)]
        X_test, y_test = X_scaled[X_scaled['ID'].isin(test_ids)], y[y['ID'].isin(test_ids)]

        X_train.drop('ID',axis = 1, inplace = True)
        y_train.drop('ID',axis = 1, inplace = True)
        y_train = y_train.values.tolist()
        X_test.drop('ID',axis = 1, inplace = True)
        y_test.drop('ID',axis = 1, inplace = True)
        y_test = y_test.values.tolist()
            

        # --- Plot 2: Coefficient Shrinkage Paths ---
        alpha_values_to_plot = np.logspace(-4, 0, 75)
        l1_ratio_to_plot = best_model.l1_ratio
        all_coefs = []
        all_alpha = []

        for alpha_val in alpha_values_to_plot:
            temp_model = ElasticNet(
                alpha=alpha_val, l1_ratio=l1_ratio_to_plot,
                max_iter=10000, random_state=42, selection='random'
            )
            temp_model.fit(X_train, y_train)
            all_coefs.append(temp_model.coef_)
            all_alpha.append(alpha_val)

        coef_df = pd.DataFrame(all_coefs, columns=feature_names)
        coef_df['alpha'] = all_alpha
        coef_df['log10_alpha'] = np.log10(coef_df['alpha'])

        shrink_threshold = 1e-8
        shrink_alpha = {}
        for feature in feature_names:
            max_abs_coef = np.abs(coef_df[feature]).max()
            if max_abs_coef > 0:
                threshold = shrink_threshold * max_abs_coef
                shrink_point = coef_df[np.abs(coef_df[feature]) >= threshold]['log10_alpha'].max()
                shrink_alpha[feature] = shrink_point
            else:
                shrink_alpha[feature] = coef_df['log10_alpha'].min()


        all_features_ordered = sorted(feature_names, key=lambda x: shrink_alpha[x], reverse=True)

        
        N_features_to_plot = 25
        top_N_features = all_features_ordered[:N_features_to_plot]
        plot_df = coef_df[top_N_features + ['log10_alpha']].copy()

        plot_df.columns = [x.removeprefix("nulisa_").removesuffix("_0y").replace("_0y_x_group", "_group") for x in plot_df.columns]
        top_N_features = [x.removeprefix("nulisa_").removesuffix("_0y").replace("_0y_x_group", "_group") for x in top_N_features]


        plt.figure(figsize=(12, 8))
        for i, feature in enumerate(top_N_features):
            plt.plot(
                plot_df['log10_alpha'], plot_df[feature],
                label=feature, linewidth=2, color=custom_colors[i]
            )
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.axvline(np.log10(metrics_all_folds[0]['Best_Alpha']),  color='green', linestyle='--', label='Best model')
        plt.xlabel('log10(alpha)')
        plt.ylabel('Coefficient Value')
        plt.title(f'Coefficient shrinkage paths for fold {fold}.')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Features')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path_options_output_model /group/ f'fold_{fold}_alpha_vs_coefficients_25top_{group}.png', bbox_inches='tight')
        plt.savefig(path_options_output_model /group/ f'fold_{fold}_alpha_vs_coefficients_25top_{group}.svg', bbox_inches='tight')
        plt.close()

        # --- Plot 2b: Coefficient Shrinkage Paths (Only Best Model Features, Ordered by Shrink Alpha) ---
        # Get features with non-zero coefficients in the best model
        best_model_features = [f for f, coef in zip(feature_names, best_model.coef_) if coef != 0]

        # Order by shrink_alpha (highest first = shrinks slowest)
        best_model_features_ordered = sorted(best_model_features, key=lambda x: shrink_alpha[x], reverse=True)

        best_model_plot_df = coef_df[best_model_features_ordered + ['log10_alpha']].copy()
        best_model_plot_df.columns = [x.removeprefix("nulisa_").removesuffix("_0y").replace("_0y_x_group", "_group") for x in best_model_plot_df.columns]

        best_model_features_ordered =[x.removeprefix("nulisa_").removesuffix("_0y").replace("_0y_x_group", "_group") for x in best_model_features_ordered]

        plt.figure(figsize=(12, 8))
        for i, feature in enumerate(best_model_features_ordered):
            plt.plot(
                best_model_plot_df['log10_alpha'],
                best_model_plot_df[feature],
                label=feature,
                linewidth=2,
                color=custom_colors[i % len(custom_colors)]  # Cycle colors if needed
            )
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.xlabel('log10(alpha)')
        plt.ylabel('Coefficient Value')
        plt.axvline(np.log10(metrics_all_folds[0]['Best_Alpha']),  color='green', linestyle='--', label='Best model')

        plt.title(f'Coefficient shrinkage paths for the features selected in the best model for fold {fold}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Features')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path_options_output_model / group/f'fold_{fold}_alpha_vs_coefficients_best_model_{group}.png', bbox_inches='tight')
        plt.savefig(path_options_output_model / group/f'fold_{fold}_alpha_vs_coefficients_best_model_{group}.svg', bbox_inches='tight')
        plt.close()

        # --- Table: Features + Coefficients for the best model ---
        coef_table = coef_table_all[coef_table_all[f'Coefficient_fold_{fold}'] != 0].sort_values(by=f'Coefficient_fold_{fold}', ascending=False, key=abs)
        coef_table.reset_index(inplace=True, drop = True)


        # --- Update feature_coef_ranges and feature_rank_ranges ---
        coef_table['Feature'] = [x.removeprefix("nulisa_").removesuffix("_0y").replace("_0y_x_group", "_group") for x in coef_table['Feature']]
        
        shrink_alpha_df['Feature'] = [x.removeprefix("nulisa_").removesuffix("_0y").replace("_0y_x_group", "_group") for x in shrink_alpha_df['Feature']]

        for _, row in  coef_table.iterrows():
            feature = row['Feature']
            if feature not in feature_coef_ranges:
                feature_coef_ranges[feature] = []
                feature_rank_ranges[feature] = []
            feature_coef_ranges[feature].append(row[f'Coefficient_fold_{fold}'])
            feature_rank_ranges[feature].append(shrink_alpha_df[shrink_alpha_df['Feature']==feature][f'shrink_rank_fold_{fold}'].iloc[0])  # Rank is 1-based index in the sorted table

    # --- Generate summary graph: coefficient ranges and rank ranges ---
    features_with_non_zero_coef = sorted(feature_coef_ranges.keys())
    features_with_non_zero_coef =[x.removeprefix("nulisa_").removesuffix("_0y").replace("_0y_x_group", "_group") for x in features_with_non_zero_coef]
    coef_min, coef_max = {}, {}
    rank_min, rank_max = {}, {}
    for feature in features_with_non_zero_coef:
        coefs = feature_coef_ranges[feature]
        ranks = feature_rank_ranges[feature]
        coef_min[feature] = min(coefs)
        coef_max[feature] = max(coefs)
        rank_min[feature] = min(ranks)
        rank_max[feature] = max(ranks)

    # --- Figure 1: Coefficient Ranges ---
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(features_with_non_zero_coef))
    plt.barh(
        y_pos,
        [coef_max[f] - coef_min[f] for f in features_with_non_zero_coef],
        left=[coef_min[f] for f in features_with_non_zero_coef],
        color='skyblue', alpha=0.6, label='Coefficient Range'
    )
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title('Coefficient Ranges Across Folds')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yticks(y_pos, features_with_non_zero_coef)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path_options_output_model /group/ f'summary_coef_ranges_{group}.png', bbox_inches='tight')
    plt.savefig(path_options_output_model /group/ f'summary_coef_ranges_{group}.svg', bbox_inches='tight')
    plt.close()

    # --- Figure 2: Rank Ranges ---
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(features_with_non_zero_coef))
    plt.barh(
        y_pos,
        [rank_max[f] - rank_min[f] for f in features_with_non_zero_coef],
        left=[rank_min[f] for f in features_with_non_zero_coef],
        color='salmon', alpha=0.6, label='Rank Range'
    )
    plt.xlabel('Rank')
    plt.ylabel('Feature')
    plt.title('Rank Ranges Across Folds')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yticks(y_pos, features_with_non_zero_coef)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path_options_output_model /group/ f'summary_rank_ranges_{group}.png', bbox_inches='tight')
    plt.savefig(path_options_output_model /group/ f'summary_rank_ranges_{group}.svg', bbox_inches='tight')
    plt.close()

    # --- Calculate mean absolute coefficients and appearance counts ---
    feature_coef_means = {}
    feature_appearance_count = {}

    for fold in fold_outputs:
        fold_outputs[fold]['coef_table']['Feature'] = [x.removeprefix("nulisa_").removesuffix("_0y").replace("_0y_x_group", "_group") for x in fold_outputs[fold]['coef_table']['Feature']]
        coef_table = fold_outputs[fold]['coef_table']
        for _, row in coef_table.iterrows():
            feature = row['Feature']
            coef = row['Coefficient']
            # Update mean absolute coefficient
            if feature not in feature_coef_means:
                feature_coef_means[feature] = []
            feature_coef_means[feature].append(coef)
            # Update appearance count
            if feature not in feature_appearance_count:
                feature_appearance_count[feature] = 0
            feature_appearance_count[feature] += 1

    # Calculate mean absolute coefficient for each feature
    feature_mean_abs_coef = {
        f: np.mean(np.abs(coefs))
        for f, coefs in feature_coef_means.items()
    }

    # Sort features by mean absolute coefficient (descending)
    features_with_non_zero_coef = sorted(
        feature_coef_means.keys(),
        key=lambda x: feature_mean_abs_coef[x],
        reverse=True
    )

    # --- Figure 1: Coefficient Ranges with Crosses and Rank Text ---
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(features_with_non_zero_coef))
    ytick_labels = [f"{f} (n={feature_appearance_count[f]})" for f in features_with_non_zero_coef]

    # Plot coefficient ranges (bars)
    width = [coef_max[f] - coef_min[f] if (coef_max[f] - coef_min[f]) != 0 else 0.001 for f in features_with_non_zero_coef]
    bars = plt.barh(
        y_pos,
        width,
        left=[coef_min[f] for f in features_with_non_zero_coef],
        color='skyblue', alpha=0.6, label='Coefficient Range'
    )

    # Add crosses and rank text for each coefficient in each fold
    for i, feature in enumerate(features_with_non_zero_coef):
        for coef, rank in zip(feature_coef_ranges[feature], feature_rank_ranges[feature]):
            plt.scatter(
                coef, i,
                marker='x', color='red', s=50, alpha=0.8, linewidths=1
            )
            # Add rank as text next to the cross
            plt.text(
                coef, i,
                str(int(rank)),
                ha='center', va='center',
                fontsize=8,
                color='black',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )


    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature (n = Appearance Count)')
    plt.title('Coefficient ranges across Folds (Ordered by Mean coefficient magnitude)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yticks(y_pos, ytick_labels)
    plt.legend(handles=[bars], loc='upper right')
    plt.tight_layout()
    plt.savefig(path_options_output_model /group/ f'summary_coef_ranges_{group}.png', bbox_inches='tight')
    plt.savefig(path_options_output_model /group/ f'summary_coef_ranges_{group}.svg', bbox_inches='tight')
    plt.close()
    
    # --- Figure 1: Coefficient Ranges with Crosses and Rank Text ---
    #! at least in 2 models
    
    features_with_non_zero_coef_above_2 = []
    feature_appearance_count_above_2 = {}
    feature_rank_ranges_above_2 = {}
    coef_max_2 = {}
    coef_min_2 = {}
    feature_coef_ranges_above_2 ={}
    
    for feat in features_with_non_zero_coef:
        if feature_appearance_count[feat] >=2:
            features_with_non_zero_coef_above_2.append(feat)
            feature_appearance_count_above_2[feat] = feature_appearance_count[feat]
            feature_rank_ranges_above_2[feat] = feature_rank_ranges[feat]
            feature_coef_ranges_above_2[feat] = feature_coef_ranges[feat]
            coef_min_2[feat] = coef_min[feat]
            coef_max_2[feat] = coef_max[feat]

    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(features_with_non_zero_coef_above_2))
    ytick_labels = [f"{f} (n={feature_appearance_count_above_2[f]})" for f in features_with_non_zero_coef_above_2]

    # Plot coefficient ranges (bars)
    width = [coef_max_2[f] - coef_min_2[f] if (coef_max_2[f] - coef_min_2[f]) != 0 else 0.001 for f in features_with_non_zero_coef_above_2]
    bars = plt.barh(
        y_pos,
        width,
        left=[coef_min[f] for f in features_with_non_zero_coef_above_2],
        color='skyblue', alpha=0.6, label='Coefficient Range'
    )

    # Add crosses and rank text for each coefficient in each fold
    for i, feature in enumerate(features_with_non_zero_coef_above_2):
        for coef, rank in zip(feature_coef_ranges_above_2[feature], feature_rank_ranges_above_2[feature]):
            plt.scatter(
                coef, i,
                marker='x', color='red', s=50, alpha=0.8, linewidths=1
            )
            # Add rank as text next to the cross
            plt.text(
                coef, i,
                str(int(rank)),
                ha='center', va='center',
                fontsize=8,
                color='black',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )

    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature (n = Appearance Count)')
    plt.title('Coefficient ranges across Folds (Ordered by Mean coefficient magnitude)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yticks(y_pos, ytick_labels)
    plt.legend(handles=[bars], loc='upper right')
    plt.tight_layout()
    plt.savefig(path_options_output_model /group/ f'summary_coef_ranges_{group}_atleast2models.png', bbox_inches='tight')
    plt.savefig(path_options_output_model /group/ f'summary_coef_ranges_{group}_atleast2models.svg', bbox_inches='tight')
    plt.close()

def apply_elasticnet(
    X_scaled, y, feature_names, path_options_output_model, group,
    train_dict=None, test_dict=None, n_folds=4, rerun_model = True, rerun_opti = True,
):
    # --- Color setup ---
    tab20 = cm.get_cmap('tab20', 20)
    tab20b = cm.get_cmap('tab20b', 20)
    tab20c = cm.get_cmap('tab20c', 20)
    colors_tab20 = [tab20(i) for i in range(20)]
    colors_tab20b = [tab20b(i) for i in range(20)]
    colors_tab20c = [tab20c(i) for i in range(20)]
    all_colors = colors_tab20 + colors_tab20b + colors_tab20c
    custom_colors = all_colors[:25]

    # --- Initialize outputs ---
    fold_outputs = {}
    metrics_all_folds = []
    feature_coef_ranges = {}
    feature_rank_ranges = {}



    # --- Loop over folds ---
    for fold in range(n_folds):
        print(f"\n--- Processing Fold {fold} ---")

        # Load data for this fold
        train_ids = train_dict[fold]
        test_ids = test_dict[fold]
        X_train = pd.merge(X_scaled,train_ids, how = 'right')
        X_train, y_train = X_scaled[X_scaled['ID'].isin(list(train_ids))], y[y['ID'].isin(train_ids)]
        X_test, y_test = X_scaled[X_scaled['ID'].isin(test_ids)], y[y['ID'].isin(test_ids)]

        X_train.drop('ID',axis = 1, inplace = True)
        y_train.drop('ID',axis = 1, inplace = True)
        y_train = y_train.values.tolist()
        X_test.drop('ID',axis = 1, inplace = True)
        y_test.drop('ID',axis = 1, inplace = True)
        y_test = y_test.values.tolist()

        # --- ElasticNet with RandomizedSearchCV ---
        if rerun_model and rerun_opti:
            model = ElasticNet(max_iter=10000, random_state=42, tol=1e-4, selection='random')
            param_distributions = {
                'alpha': loguniform(1e-4, 1e2),
                'l1_ratio': uniform(0.1, 0.9)
            }
            kf_inner = KFold(n_splits=3, shuffle=True, random_state=42)
            random_search = RandomizedSearchCV(
                model, param_distributions, n_iter=30, cv=kf_inner,
                scoring='neg_mean_squared_error', n_jobs=-1, random_state=42, verbose=1,
                return_train_score=True  # Enable to get train scores
            )
            random_search.fit(X_train, y_train)
            
            with open(path_options_output_model /group/ f'random_search_fold{fold}_{group}.pkl', 'wb') as f:
                pickle.dump(random_search, f)
        else:
            with open(path_options_output_model /group/ f'random_search_fold{fold}_{group}.pkl', 'rb') as f:
                random_search = pickle.load(f)
            

        # --- Filter candidates to keep only those with > 5 non-zero features ---
        results = pd.DataFrame(random_search.cv_results_)
        sorted_indices = results['mean_test_score'].argsort()[::-1]  # Sort by best score first
        best_model = None
        best_alpha = None
        best_l1_ratio = None

        for idx in sorted_indices:
            params = results['params'].iloc[idx]
            temp_model = ElasticNet(**params, max_iter=10000, random_state=42, tol=1e-4, selection='random')
            temp_model.fit(X_train, y_train)
            non_zero_count = np.sum(temp_model.coef_ != 0)
            if np.sum(temp_model.coef_ != 0) <= 5:
                print(f"Selected model with {non_zero_count} non-zero features. Search for another one")

            if non_zero_count > 5:
                best_model = temp_model
                best_alpha = temp_model.alpha
                best_l1_ratio = temp_model.l1_ratio
                print(f"Selected model with {non_zero_count} non-zero features (alpha={best_alpha:.2e}, l1_ratio={best_l1_ratio:.2f})")
                break

        if best_model is None:
            print("No model with > 5 non-zero features found. Using best model from RandomizedSearchCV.")
            best_model = random_search.best_estimator_
            best_alpha = best_model.alpha
            best_l1_ratio = best_model.l1_ratio

        # --- Evaluate on train and validation sets ---
        train_pred = best_model.predict(X_train)
        val_pred = best_model.predict(X_test)
        train_mse = np.mean((y_train - train_pred) ** 2)
        Test_MSE = np.mean((y_test - val_pred) ** 2)

        # --- Save metrics for this fold ---
        metrics_all_folds.append({
            'Fold': fold,
            'Train_MSE': train_mse,
            'Test_MSE': Test_MSE,
            'Best_Alpha': best_alpha,
            'Best_L1_Ratio': best_l1_ratio,
            'Non_Zero_Features': np.sum(best_model.coef_ != 0)
        })


        # --- Plot 1: Train and Validation MSE vs. Alpha (from CV) ---
        plt.figure(figsize=(10, 6))
        plt.scatter(
            np.log10(results['param_alpha']),
            -results['mean_train_score'],
            alpha=0.5, color='blue', label='Train MSE (CV)'
        )
        plt.scatter(
            np.log10(results['param_alpha']),
            -results['mean_test_score'],
            alpha=0.5, color='red', label='Validation MSE (CV)'
        )
        #plt.axvline(np.log10(metrics_all_folds[0]['Best_Alpha']),  color='green', linestyle='--', label='Best model')
        plt.scatter(
            np.log10(best_alpha), -random_search.best_score_,
            color='green', s=100, label=f'Best Model (alpha={best_alpha:.2e})'
        )
        plt.xlabel('log10(alpha)')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title(f'Fold {fold}: Train vs Validation Average MSE vs. Alpha (CV)')
        plt.grid(True)
        plt.legend()
        plt.savefig(path_options_output_model /group/ f'fold_{fold}_cv_mse_vs_alpha_{group}.png', bbox_inches='tight')
        plt.close()

        # --- Plot 2: Coefficient Shrinkage Paths ---
        alpha_values_to_plot = np.logspace(-4, 0, 75)
        l1_ratio_to_plot = best_model.l1_ratio
        all_coefs = []
        all_alpha = []

        for alpha_val in alpha_values_to_plot:
            temp_model = ElasticNet(
                alpha=alpha_val, l1_ratio=l1_ratio_to_plot,
                max_iter=10000, random_state=42, selection='random'
            )
            temp_model.fit(X_train, y_train)
            all_coefs.append(temp_model.coef_)
            all_alpha.append(alpha_val)

        coef_df = pd.DataFrame(all_coefs, columns=feature_names)
        coef_df['alpha'] = all_alpha
        coef_df['log10_alpha'] = np.log10(coef_df['alpha'])

        shrink_threshold = 1e-8
        shrink_alpha = {}
        for feature in feature_names:
            max_abs_coef = np.abs(coef_df[feature]).max()
            if max_abs_coef > 0:
                threshold = shrink_threshold * max_abs_coef
                shrink_point = coef_df[np.abs(coef_df[feature]) >= threshold]['log10_alpha'].max()
                shrink_alpha[feature] = shrink_point
            else:
                shrink_alpha[feature] = coef_df['log10_alpha'].min()

        shrink_alpha_df = pd.DataFrame.from_dict(shrink_alpha,orient="index")
        shrink_alpha_df.columns = [f'shrink_alpha_fold_{fold}']
        shrink_alpha_df = shrink_alpha_df.sort_values(by=f'shrink_alpha_fold_{fold}',ascending=False)
        shrink_alpha_df['feature'] = shrink_alpha_df.index
        shrink_alpha_df.reset_index(inplace=True, drop = True )

        rank_shrink = shrink_alpha_df[f'shrink_alpha_fold_{fold}'].drop_duplicates()
        rank_shrink = pd.DataFrame(rank_shrink)
        rank_shrink['shrink_rank'] = rank_shrink.index + 1

        shrink_alpha_df = pd.merge(shrink_alpha_df,rank_shrink, on =  [f'shrink_alpha_fold_{fold}'])
        


        all_features_ordered = sorted(feature_names, key=lambda x: shrink_alpha[x], reverse=True)
        N_features_to_plot = 25
        top_N_features = all_features_ordered[:N_features_to_plot]
        plot_df = coef_df[top_N_features + ['log10_alpha']].copy()

        plt.figure(figsize=(12, 8))
        for i, feature in enumerate(top_N_features):
            plt.plot(
                plot_df['log10_alpha'], plot_df[feature],
                label=feature, linewidth=2, color=custom_colors[i]
            )
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.axvline(np.log10(metrics_all_folds[0]['Best_Alpha']),  color='green', linestyle='--', label='Best model')
        plt.xlabel('log10(alpha)')
        plt.ylabel('Coefficient Value')
        plt.title(f'Fold {fold}: Coefficient shrinkage paths')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Features')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path_options_output_model /group/ f'fold_{fold}_alpha_vs_coefficients_25top_{group}.png', bbox_inches='tight')
        plt.close()

        # --- Plot 2b: Coefficient Shrinkage Paths (Only Best Model Features, Ordered by Shrink Alpha) ---
        # Get features with non-zero coefficients in the best model
        best_model_features = [f for f, coef in zip(feature_names, best_model.coef_) if coef != 0]

        # Order by shrink_alpha (highest first = shrinks slowest)
        best_model_features_ordered = sorted(best_model_features, key=lambda x: shrink_alpha[x], reverse=True)

        if len(best_model_features_ordered) > 0:
            best_model_plot_df = coef_df[best_model_features_ordered + ['log10_alpha']].copy()
            plt.figure(figsize=(12, 8))
            for i, feature in enumerate(best_model_features_ordered):
                plt.plot(
                    best_model_plot_df['log10_alpha'],
                    best_model_plot_df[feature],
                    label=feature,
                    linewidth=2,
                    color=custom_colors[i % len(custom_colors)]  # Cycle colors if needed
                )
            plt.axhline(0, color='black', linestyle='--', linewidth=1)
            plt.xlabel('log10(alpha)')
            plt.ylabel('Coefficient Value')
            plt.axvline(np.log10(metrics_all_folds[0]['Best_Alpha']),  color='green', linestyle='--', label='Best model')

            plt.title(f'Coefficient shrinkage paths for the features selected in the best model for fold {fold}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Features')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(path_options_output_model / group/f'fold_{fold}_alpha_vs_coefficients_best_model_{group}.png', bbox_inches='tight')
            plt.close()

        # --- Table: Features + Coefficients for the best model ---
        coefs = best_model.coef_
        coef_table_all = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
        coef_table_all = coef_table_all.sort_values(by='Coefficient', ascending=False, key=abs)
        coef_table_all.reset_index(inplace=True, drop = True)

        coef_table = coef_table_all[coef_table_all['Coefficient'] != 0].sort_values(by='Coefficient', ascending=False, key=abs)
        coef_table.reset_index(inplace=True, drop = True)
        coef_table.to_csv(path_options_output_model /group/ f'fold_{fold}_best_model_coefficients_{group}.csv', index=False)


        # --- Store outputs for this fold ---
        fold_outputs[fold] = {
            'best_model': best_model,
            'shrink_alpha': shrink_alpha,
            'coef_table': coef_table,
            'coef_table_all': coef_table_all,
            "shrink_alpha": shrink_alpha_df,
            'metrics': {'Train_MSE': train_mse, 'Test_MSE': Test_MSE}
        }

        fold_outputs[fold]['shrink_alpha'].columns = [f'shrink_alpha_fold_{fold}', 'Feature', f'shrink_rank_fold_{fold}']
        fold_outputs[fold]['coef_table_all'].columns = [f'Feature', f'Coefficient_fold_{fold}']

        # --- Update feature_coef_ranges and feature_rank_ranges ---
        for idx, row in  coef_table.iterrows():
            feature = row['Feature']
            if feature not in feature_coef_ranges:
                feature_coef_ranges[feature] = []
                feature_rank_ranges[feature] = []
            feature_coef_ranges[feature].append(row['Coefficient'])
            feature_rank_ranges[feature].append(shrink_alpha_df[shrink_alpha_df['feature']==feature]['shrink_rank'].iloc[0])  # Rank is 1-based index in the sorted table

    # --- Save summary table of metrics for all folds ---
    metrics_df = pd.DataFrame(metrics_all_folds)
    metrics_df.to_csv(path_options_output_model / group/f'summary_metrics_all_folds_{group}.csv', index=False)

    # --- Generate summary graph: coefficient ranges and rank ranges ---
    features_with_non_zero_coef = sorted(feature_coef_ranges.keys())
    coef_min, coef_max = {}, {}
    rank_min, rank_max = {}, {}
    for feature in features_with_non_zero_coef:
        coefs = feature_coef_ranges[feature]
        ranks = feature_rank_ranges[feature]
        coef_min[feature] = min(coefs)
        coef_max[feature] = max(coefs)
        rank_min[feature] = min(ranks)
        rank_max[feature] = max(ranks)

    # --- Figure 1: Coefficient Ranges ---
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(features_with_non_zero_coef))
    plt.barh(
        y_pos,
        [coef_max[f] - coef_min[f] for f in features_with_non_zero_coef],
        left=[coef_min[f] for f in features_with_non_zero_coef],
        color='skyblue', alpha=0.6, label='Coefficient Range'
    )
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title('Coefficient Ranges Across Folds')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yticks(y_pos, features_with_non_zero_coef)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path_options_output_model /group/ f'summary_coef_ranges_{group}.png', bbox_inches='tight')
    plt.close()

    # --- Figure 2: Rank Ranges ---
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(features_with_non_zero_coef))
    plt.barh(
        y_pos,
        [rank_max[f] - rank_min[f] for f in features_with_non_zero_coef],
        left=[rank_min[f] for f in features_with_non_zero_coef],
        color='salmon', alpha=0.6, label='Rank Range'
    )
    plt.xlabel('Rank')
    plt.ylabel('Feature')
    plt.title('Rank Ranges Across Folds')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yticks(y_pos, features_with_non_zero_coef)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path_options_output_model /group/ f'summary_rank_ranges_{group}.png', bbox_inches='tight')
    plt.close()

    # --- Calculate mean absolute coefficients and appearance counts ---
    feature_coef_means = {}
    feature_appearance_count = {}

    for fold in fold_outputs:
        coef_table = fold_outputs[fold]['coef_table']
        for _, row in coef_table.iterrows():
            feature = row['Feature']
            coef = row['Coefficient']
            # Update mean absolute coefficient
            if feature not in feature_coef_means:
                feature_coef_means[feature] = []
            feature_coef_means[feature].append(coef)
            # Update appearance count
            if feature not in feature_appearance_count:
                feature_appearance_count[feature] = 0
            feature_appearance_count[feature] += 1

        fold_outputs[fold]["appearance_count"] = feature_appearance_count
        fold_outputs[fold]["rank"] = feature_rank_ranges
    # Calculate mean absolute coefficient for each feature
    feature_mean_abs_coef = {
        f: np.mean(np.abs(coefs))
        for f, coefs in feature_coef_means.items()
    }

    # Sort features by mean absolute coefficient (descending)
    features_with_non_zero_coef = sorted(
        feature_coef_means.keys(),
        key=lambda x: feature_mean_abs_coef[x],
        reverse=True
    )

    # --- Figure 1: Coefficient Ranges with Crosses and Rank Text ---
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(features_with_non_zero_coef))
    ytick_labels = [f"{f} (n={feature_appearance_count[f]})" for f in features_with_non_zero_coef]

    # Plot coefficient ranges (bars)
    width = [coef_max[f] - coef_min[f] if (coef_max[f] - coef_min[f]) != 0 else 0.001 for f in features_with_non_zero_coef]
    bars = plt.barh(
        y_pos,
        width,
        left=[coef_min[f] for f in features_with_non_zero_coef],
        color='skyblue', alpha=0.6, label='Coefficient Range'
    )

    # Add crosses and rank text for each coefficient in each fold
    for i, feature in enumerate(features_with_non_zero_coef):
        for coef, rank in zip(feature_coef_ranges[feature], feature_rank_ranges[feature]):
            plt.scatter(
                coef, i,
                marker='x', color='red', s=50, alpha=0.8, linewidths=1
            )
            # Add rank as text next to the cross
            plt.text(
                coef, i,
                str(int(rank)),
                ha='center', va='center',
                fontsize=8,
                color='black',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )

    # Add a note explaining the plot
    plt.text(
        0.02, 0.98,
        'Red crosses: Fold-specific coefficients. Associated number: Importance rank according to Shrinking alpha',
        transform=plt.gca().transAxes,
        ha='left', va='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )

    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature (n = Appearance Count)')
    plt.title('Coefficient ranges across Folds (Ordered by Mean coefficient magnitude)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yticks(y_pos, ytick_labels)
    plt.legend(handles=[bars], loc='upper right')
    plt.tight_layout()
    plt.savefig(path_options_output_model /group/ f'summary_coef_ranges_{group}.png', bbox_inches='tight')
    plt.close()

    all_shrink = fold_outputs[0]['shrink_alpha'].copy()
    all_shrink = all_shrink[['Feature','shrink_alpha_fold_0','shrink_rank_fold_0']]
    for f in range(1,n_folds):
        all_shrink = pd.merge(all_shrink, fold_outputs[f]['shrink_alpha'], on =['Feature'], how = 'right')
    all_shrink.columns = ['Feature'] + [ col for f in range(0,n_folds)  for col in (f'shrink_alpha_fold_{f}', f'shrink_rank_fold_{f}')]
    all_shrink = round(all_shrink,4)
    shrink_range = all_shrink[[f"shrink_rank_fold_{f}" for f in range(n_folds)]]
    
    all_shrink['min shrink rank'] = np.min(shrink_range, axis=1)
    all_shrink['median shrink rank'] = np.median(shrink_range, axis=1)
    all_shrink['max shrink rank'] = np.max(shrink_range, axis=1)
    all_shrink["min/median/max shrink rank"] = [
        f"{mn} / {md} / {mx}"
        for mn, md, mx in zip(all_shrink['min shrink rank'], all_shrink['median shrink rank'], all_shrink['max shrink rank'])
    ]    


    
    
    # --- Save all outputs ---
    if rerun_model:
        with open(path_options_output_model /group/ f'elasticnet_all_folds_outputs_{group}.pkl', 'wb') as f:
            pickle.dump(fold_outputs, f)
    else:
        with open(path_options_output_model /group/ f'elasticnet_all_folds_outputs_{group}.pkl', 'rb') as f:
            fold_outputs = pickle.load(f)
            




    
    
    all_coeff_fin = fold_outputs[0]['coef_table_all'].copy()
    for f in range(1,n_folds):
        all_coeff_fin = pd.merge(all_coeff_fin, fold_outputs[f]['coef_table_all'], on =['Feature'], how = 'right')
    all_coeff_fin = round(all_coeff_fin,4)

    all_coeff_fin_range = all_coeff_fin[[f"Coefficient_fold_{f}" for f in range(n_folds)]]
        
    # Ignore coefficients equal to 0
    coeff_nonzero = all_coeff_fin_range.replace(0, np.nan)

    all_coeff_fin['min Coeff'] = coeff_nonzero.min(axis=1, skipna=True)
    all_coeff_fin['median Coeff'] = coeff_nonzero.median(axis=1, skipna=True)
    all_coeff_fin['max Coeff'] = coeff_nonzero.max(axis=1, skipna=True)

    all_coeff_fin["min/median/max Coeff"] = [
        f"{mn} / {md} / {mx}"
        for mn, md, mx in zip(all_shrink['min shrink rank'], all_shrink['median shrink rank'], all_shrink['max shrink rank'])
    ]    
    

    table_final =  all_coeff_fin[["Feature",'min Coeff','median Coeff','max Coeff',"min/median/max Coeff"]]
    all_shrink = all_shrink[["Feature",'min shrink rank','median shrink rank','max shrink rank',"min/median/max shrink rank"]]
    
    table_final = pd.merge(table_final,all_shrink , on = 'Feature', how = 'right')
    
    appar = pd.DataFrame.from_dict(fold_outputs[0]['appearance_count'],orient="index")
    appar['Feature'] = appar.index
    appar.reset_index(inplace=True, drop=True)
    appar.columns = ['Apparence count', 'Feature']
    table_final = pd.merge(table_final,appar , on = 'Feature', how = 'right')

    table_final.sort_values(inplace=True, by=['Apparence count'],ascending=False)
    
    
    table_final.to_csv(path_options_output_model /group/ f'table_final_{group}.csv', index=False)



    print("All folds processed successfully. Summary tables and graphs saved.")
    return fold_outputs, metrics_df
if __name__ == "__main__":
     
    #########################
    # Arguments
    ########################
    centered = False
    windsor_ratio = 1

    rerun_split = False
    rerun_model = False
    rerun_opti = False

    cross_val = True
    n_folds = 4

    version_model = 'v1'
    cognition = 'ztotal_composite'  

    #! #################################################################
    #! Load and clean the data #########################################
    #! #################################################################

    df, col_cofacteurs_base, cofactors_to_test, col_nulisa, col_nulisa_baseline = get_data() 
    df.reset_index(drop = True, inplace=True)
    
    name_folder = f'mitt_windsor_halmin'
    if centered:
        name_folder += '_centered'

    logger = setup_logger(name='ElasticNet', log_dir=LOG_PATH, subfolder_name='machine learning R version', log_file=f'causal_forest_{version_model}.log', level=logging.INFO, mode='w')


    logger.info(f" Options: centered: {centered}, subset: mITT, windsor: True, windsor_ratio: {windsor_ratio}, cofactors_choice: age + sex + education + BMI + site, rerun_split: {rerun_split}, rerun_model: {rerun_model}, cross_val: {cross_val}, rerun_opti: {rerun_opti}, cognition: {cognition} ")

    path_options_output_data = LONG_PATH/'ElasticNet'/name_folder/'data'
    path_options_output_model = LONG_PATH/'ElasticNet'/name_folder/version_model

    os.makedirs(path_options_output_data, exist_ok=True)
    os.makedirs(path_options_output_model, exist_ok=True)
    os.makedirs(path_options_output_model/'data_backup', exist_ok=True)

    
    dict_outputs = {}

    #! #################################################################
    #! Prepare the target variable (cognitive change)
    #! #################################################################
    df[f"{cognition}_change"] = df[f"{cognition}_2y"] - df[f"{cognition}_0y"]

    columns_to_keep = list(
        set(['ID', 'group', 'ztotal_composite_change', f"{cognition}_2y", f"{cognition}_1y", f"{cognition}_0y"] +
            col_nulisa_baseline + cofactors_to_test + col_cofacteurs_base)
    )
    df = df[columns_to_keep]
    df.dropna(inplace=True)

    dict_outputs['df'] = df
    
    #! #################################################################
    #! Prepare for cross-validation 
    #! #################################################################

    dataset_version = os.listdir(path_options_output_data)
    if dataset_version == []:
        os.makedirs(path_options_output_data/"dataset_v0", exist_ok=True)
        
    # Filter out only the directories and match the version pattern
    version_folders = []
    pattern = re.compile(r'dataset_v(\d+)')

    # Filter out only the directories
    for entry in dataset_version:
        match = pattern.match(entry)
        if match:
            version_folders.append((entry, int(match.group(1))))

    
    latest_folder, latest_version = max(version_folders, key=lambda x: x[1])

    train_dict = {}
    test_dict = {}

    if rerun_split or not os.path.isfile(path_options_output_data/f'train_list_fold_0_over_{n_folds-1}.csv'):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            train_ids = df.iloc[train_idx]['ID']
            test_ids = df.iloc[val_idx]['ID']
            train_ids.to_csv(path_options_output_data / f'train_list_fold_{fold}_over_{n_folds-1}.csv', index=False)
            test_ids.to_csv(path_options_output_data / f'test_list_fold_{fold}_over_{n_folds-1}.csv', index=False)
            train_dict[fold] = train_ids
            test_dict[fold] = test_ids
    else:
        for fold in range(n_folds):
            train_dict[fold] = pd.read_csv(
                path_options_output_data / f'train_list_fold_{fold}_over_{n_folds-1}.csv'
            )['ID']
            test_dict[fold] = pd.read_csv(
                path_options_output_data / f'test_list_fold_{fold}_over_{n_folds-1}.csv'
            )['ID']

    #! #################################################################
    #! Select features #########################################
    #! #################################################################

    df.reset_index(inplace = True, drop = True)
    feature_names = col_nulisa_baseline + cofactors_to_test
    
    interaction_feature_names = [f"{f}_x_group" for f in feature_names]
    all_feature_names = feature_names + interaction_feature_names


    print("Full cohort *************")

    X = df[feature_names + ['ID']]
    y = df[['ztotal_composite_change'] + ['ID']]
    X_scaled = X.copy()
    X_scaled[feature_names] = scaling(X[feature_names], feature_names)
    

    if rerun_model:
        fold_outputs, metrics_df = apply_elasticnet(
            X_scaled=X_scaled,
            y=y,
            feature_names=feature_names,
            path_options_output_model=path_options_output_model,
            group="full",
            train_dict = train_dict,
            test_dict = test_dict,
            n_folds=n_folds,
            rerun_model = rerun_model
        )

    get_graphs(
        X_scaled=X_scaled,
        y=y,
        feature_names=feature_names,
        path_options_output_model=path_options_output_model,
        group="full",
        train_dict = train_dict,
        test_dict = test_dict
        )



    print("Intervention cohort *************")

    df_intervention = df.copy()
    df_intervention = df_intervention[df_intervention['group'] == "Intervention"]
    df_intervention.reset_index(drop = True, inplace=True)
    X_intervention = df_intervention[feature_names + ['ID']]
    y_intervention = df_intervention[['ztotal_composite_change'] + ['ID']]
    X_intervention_scaled = X_intervention.copy()
    X_intervention_scaled[feature_names] = scaling(X_intervention[feature_names], feature_names)

    if rerun_model:
        fold_outputs_intervention , metrics_df_intervention  = apply_elasticnet(
            X_scaled=X_intervention_scaled,
            y=y_intervention,
            feature_names=feature_names,
            path_options_output_model=path_options_output_model,
            group="intervention",
            train_dict = train_dict,
            test_dict = test_dict,
            n_folds=n_folds,
            rerun_model = rerun_model
        )

    get_graphs(
        X_scaled=X_intervention_scaled,
        y=y_intervention,
        feature_names=feature_names,
        path_options_output_model=path_options_output_model,
        group="intervention",
        train_dict = train_dict,
        test_dict = test_dict
        )


    print("Control cohort *************")

    df_control = df[df['group'] == "Control"]
    df_control.reset_index(drop = True, inplace=True)
    X_control = df_control[feature_names + ['ID']]
    y_control = df_control[['ztotal_composite_change'] + ['ID']]
    X_control_scaled = X_control.copy()
    X_control_scaled[feature_names] = scaling(X_control_scaled[feature_names], feature_names)

    if rerun_model:
        fold_outputs_control , metrics_df_control  = apply_elasticnet(
            X_scaled=X_control_scaled,
            y=y_control,
            feature_names=feature_names,
            path_options_output_model=path_options_output_model,
            group="control",
            train_dict = train_dict,
            test_dict = test_dict,
            n_folds=n_folds,
            rerun_model = rerun_model
        )

    get_graphs(
        X_scaled=X_control_scaled,
        y=y_control,
        feature_names=feature_names,
        path_options_output_model=path_options_output_model,
        group="control",
        train_dict = train_dict,
        test_dict = test_dict
        )

    print("Full cohort with interaction terms *************")


    df_interaction = df.copy()
    df_interaction['group_binary'] = np.where(df_interaction['group']=='Intervention',1,0)
    X_interaction = df_interaction[feature_names + ['ID']]
    y_interaction = df_interaction[['ztotal_composite_change'] + ['ID']]
    X_interaction_scaled = X_interaction.copy()
    X_interaction_scaled[feature_names] = scaling(X_interaction_scaled[feature_names], feature_names)
    for feature in feature_names:
        X_interaction_scaled[f"{feature}_x_group"] = X_scaled[feature] * df_interaction['group_binary']

    if rerun_model:
        fold_outputs_interaction , metrics_df_interaction  = apply_elasticnet(
            X_scaled=X_interaction_scaled,
            y=y_interaction,
            feature_names=feature_names + [f"{feature}_x_group" for feature in feature_names],
            path_options_output_model=path_options_output_model,
            group="full_interaction",
            train_dict = train_dict,
            test_dict = test_dict,
            n_folds=n_folds,
            rerun_model = rerun_model
        )

    get_graphs(
        X_scaled=X_interaction_scaled,
        y=y_interaction,
        feature_names=feature_names + [f"{feature}_x_group" for feature in feature_names],
        path_options_output_model=path_options_output_model,
        group="full_interaction",
        train_dict = train_dict,
        test_dict = test_dict
        )




