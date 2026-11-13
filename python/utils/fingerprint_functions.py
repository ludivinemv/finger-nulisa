##############################
# Purpose: Causal forest functions

# Date creation: 2025-11-10
# Author: Ludivine Morvan


##############################


#! PUBLIC IMPORTS
import os, sys
from typing import List
import pandas as pd
import numpy as np  

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from pathlib import Path
from tqdm import tqdm
import shap
import ast, logging
from joblib import dump, load
import networkx as nx
from itertools import product

import statsmodels.api as sm
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error

from scipy.stats import spearmanr
from scipy.sparse import hstack

from causalml.inference.tree import CausalRandomForestRegressor
from causalml.metrics import qini_score

import warnings
warnings.filterwarnings('ignore')

#! PRIVATE IMPORTS
sys.path.append('P:/H1_FINGER AD BBM/')

import utils.analyses_functions as af
from utils.statistics import calculate_anot_spearman


def cate_quantile_rank_score(y:pd.Series|np.ndarray, treatment:pd.Series|np.ndarray, cate_pred:pd.Series|np.ndarray, n_groups:int=4):
    """ Calculate Spearman coefficient and p-value per quantile between average predicted CATE and observed effect.

    Args:
        y (pd.Series | np.ndarray): Outcomes (cognitive change)
        treatment (pd.Series | np.ndarray): Treatment / Intervention group. Values must be 1 or 0.
        cate_pred (pd.Series | np.ndarray): predicted CATE
        n_groups (int, optional): Number of groups/quantiles. Defaults to 4.

    Returns:
        spearman_corr (float): Spearman coefficient
        pval (float): p-value
    """

    df = pd.DataFrame({'y': y, 'treatment': treatment, 'cate_pred': cate_pred})
    df['quantile'] = pd.qcut(df['cate_pred'], q=n_groups, labels=False, duplicates='drop')

    avg_pred = []
    obs_effect = []

    for q in range(df['quantile'].nunique()): # for each quantile
        g = df[df['quantile'] == q]
        treated = g[g['treatment'] == 1]
        control = g[g['treatment'] == 0]
        if len(treated) == 0 or len(control) == 0:
            continue
        avg_pred.append(g['cate_pred'].mean()) # average predicted CATE
        obs_effect.append(treated['y'].mean() - control['y'].mean()) #difference of average cognitive change between treatment groups

    if len(avg_pred) < 2:
        return float('nan')  # not enough data to compute correlation

    spearman_corr, pval = spearmanr(avg_pred, obs_effect) # get Spearman coefficient and p-value

    return spearman_corr, pval


def cate_calibration_score(y:pd.Series|np.ndarray, treatment:pd.Series|np.ndarray, cate_pred:pd.Series|np.ndarray, n_groups:int=4):
    """
    Compute CATE calibration R² (predicted vs observed effect by quantiles).
    Returns R² and RMSE.

    Args:
        y (pd.Series | np.ndarray): Outcomes (cognitive change)
        treatment (pd.Series | np.ndarray): Treatment / Intervention group. Values must be 1 or 0.
        cate_pred (pd.Series | np.ndarray): predicted CATE
        n_groups (int, optional): Number of groups/quantiles. Defaults to 4.

    Returns:
        r2 (float): r2 coefficient
        rmse (float): RMSE value
    """
    df = pd.DataFrame({'y': y, 'treatment': treatment, 'cate_pred': cate_pred})
    df['group'] = pd.qcut(df['cate_pred'], q=n_groups, labels=False, duplicates='drop')

    results = []
    for g in range(df['group'].nunique()):
        g_data = df[df['group'] == g]
        treated = g_data[g_data['treatment'] == 1]
        control = g_data[g_data['treatment'] == 0]
        if len(treated) == 0 or len(control) == 0:
            continue
        obs_effect = treated['y'].mean() - control['y'].mean()
        pred_mean = g_data['cate_pred'].mean()
        results.append((pred_mean, obs_effect))

    if len(results) < 2:
        return np.nan, np.nan  # Not enough bins to compute meaningful correlation

    preds, obs = zip(*results)
    r2 = r2_score(obs, preds)
    mse = mean_squared_error(obs, preds)
    rmse = np.sqrt(mse)

    return r2, rmse


def get_qini_score(ite:np.ndarray, y:np.ndarray, treatment:np.ndarray, path:Path = None, plot:bool = False, setname:str = 'test'):
    """Get qini score and curve

    Args:
        ite (np.ndarray): predicted ITE
        y (np.ndarray): outcome (cognitive change)
        treatment (np.ndarray): treatment/intervention group.
        path (Path, optional): Path to saved figure. Defaults to None.
        plot (bool, optional): If True, plot Qini figure. Defaults to False.
        setname (str, optional): Name of the dataset. Defaults to 'test'.

    Returns:
        score_val (float): Qini Score
    """
    ite_val_series = pd.DataFrame(ite, columns=["ite"])
    input_score_val = pd.concat([
        pd.DataFrame(y, columns=["outcome"]),
        ite_val_series,
        pd.DataFrame(treatment, columns=["treatment"])
    ], axis=1)

    #calculate qini score
    score_val = qini_score(
        df=input_score_val,
        outcome_col="outcome",
        treatment_col="treatment"
    ).iloc[0]


    # Optionally plot Qini curve
    if plot:
        
        # Sort individuals by predicted uplift (ITE)
        df_sorted = input_score_val.sort_values("ite", ascending=False).reset_index(drop=True)

        # Cumulative number of samples
        n = len(df_sorted)
        x = np.arange(1, n + 1) / n  # proportion of population targeted

        # Compute incremental gains (Qini curve)
        treated = df_sorted["treatment"]
        outcome = df_sorted["outcome"]

        # Actual uplift = outcomes in treated - outcomes in control, cumulatively
        cum_treated = np.cumsum(treated * outcome) / max(treated.sum(), 1)
        cum_control = np.cumsum((1 - treated) * outcome) / max((1 - treated).sum(), 1)
        qini_y = cum_treated - cum_control

        # Plot curve
        plt.figure(figsize=(6, 5))
        plt.plot(x, qini_y, label=f"{setname} Qini curve")
        plt.xlabel("Proportion of population targeted")
        plt.ylabel("Incremental gains (Qini)")
        plt.title(f"Qini Curve ({setname}) — Score: {score_val:.4f}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        if path:
            plt.savefig(path/f"qini_curve_{setname}", bbox_inches='tight')
        plt.show()

    return score_val


def get_tsne(crforest:CausalRandomForestRegressor, leaf_hot:np.ndarray, X_sel:pd.DataFrame|np.ndarray, df:pd.DataFrame, 
             feature_selection:str, path_options_output:Path|str, version:str = 'lastleaf', set_version:str = 'train') -> pd.DataFrame:
    """Get t-sne from embedding space

    Args:
        crforest (CausalRandomForestRegressor): Causal Forest model
        leaf_hot (np.ndarray): embedding space after One-Hot encoding
        X_sel (pd.DataFrame|np.ndarray): X data after feature selection
        df (pd.DataFrame): dataframe with intervention group and outcome values
        feature_selection (str): version of the feature selection
        path_options_output (Path | str): Path to save outputs
        version (str, optional): version of the embedding space extraction. Defaults to 'lastleaf'.
        set_version (str, optional): dataset version. Defaults to 'train'.

    Returns:
        df_plot (pd.DataFrame): Dataframe with embedding space after TSNE
    """
    S = (leaf_hot*leaf_hot.transpose()).todense() # similarity matrix
    D = 1 - S/S.max() # convert similarity into distance
    embed = TSNE( perplexity=30).fit_transform(np.array(D))
    
    preds_train_1 = crforest.predict(X_sel)

    # Convert to DataFrame 
    df_plot = pd.DataFrame({
        'Embed1': embed[:, 0],
        'Embed2': embed[:, 1],
        'PredITE': preds_train_1, 
        'group': df['group'].values,
        'cognition_change': df['ztotal_composite_change'].values,
    })

    # Plot embedding space after TSNE (2D) colored by Predicted ITE
    fig, ax = plt.subplots(figsize=(8,6))
    scatter = sns.scatterplot(
        data=df_plot,
        x='Embed1',
        y='Embed2',
        hue='PredITE',          # color by predicted ITE
        palette='vlag',     
        s=50,                   
        alpha=0.8,
        legend=False,
        ax=ax   
    )
    norm = mpl.colors.Normalize(vmin=df_plot['PredITE'].min(), vmax=df_plot['PredITE'].max())
    sm = mpl.cm.ScalarMappable(cmap='vlag', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Predicted ITE')
    plt.xlabel('Embedding 1')
    plt.ylabel('Embedding 2')
    plt.title('Causal Forest Embedding colored by Predicted ITE')
    plt.savefig(path_options_output/f'causal_forest_embedding_featselect_{feature_selection}_predicted_ite_{version}_{set_version}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


    # Plot embedding space after TSNE (2D) for the intervention group colored by cognitive change
    df_plot_inter = df_plot[df_plot['group']=='Intervention']
    fig, ax = plt.subplots(figsize=(8,6))
    scatter = sns.scatterplot(
        data=df_plot_inter,
        x='Embed1',
        y='Embed2',
        hue='cognition_change',          # color by cognitive change
        palette='vlag',    
        s=50,                   
        alpha=0.8,
        legend=False,
        ax=ax   
    )
    norm = mpl.colors.Normalize(vmin=df_plot_inter['cognition_change'].min(), vmax=df_plot_inter['cognition_change'].max())
    sm = mpl.cm.ScalarMappable(cmap='vlag', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Cognitive change')
    plt.xlabel('Embedding 1')
    plt.ylabel('Embedding 2')
    plt.title('Causal Forest Embedding colored by cognitive change when intervention')
    plt.savefig(path_options_output/f'causal_forest_embedding_featselect_{feature_selection}_intervention_cognitive_change_{version}_{set_version}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return df_plot


def get_embedding_causal_forest(crforest:CausalRandomForestRegressor, X_sel:pd.DataFrame|np.ndarray, df:pd.DataFrame, 
                                path_options_output:Path|str, feature_selection:str, version:str = 'lastleaf', 
                                set_version:str = 'train'):
    """Get embedding space from the Causal forest model for individuals with profile from X_sel.

    Args:
        crforest (CausalRandomForestRegressor):  Causal Forest model        
        X_sel (pd.DataFrame|np.ndarray): X data after feature selection
        df (pd.DataFrame): dataframe with intervention group and outcome values
        path_options_output (Path | str): Path to save outputs
        feature_selection (str): version of the feature selection
        version (str, optional): version of the embedding space extraction. If 'lastleaf', 
                        the embedding space will be the leaf in which fall the individual in each tree and 
                        the size of the embedding space will be equal to the number of trees.
                        If 'fullpath': the embedding space will depend on all the leafs the individual fell.
                        Defaults to 'lastleaf'.
        set_version (str, optional): dataset version. Defaults to 'train'.

    Returns:
        df_plot (pd.DataFrame): dataframe with embedding space after Tsne
        leaf_matrix (np.ndarray): Embedding space before One-Hot encoding
        leaf_hot_dense (np.ndarray): Embedding space after One-Hot encoding
    """
    all_leaf_indices = []
    for tree in crforest.estimators_:  
        if version == 'lastleaf':
            leaf_idx = tree.apply(X_sel.values)  # shape = (n_samples,)
        else:
            leaf_idx = tree.decision_path(X_sel.values)  # (n_samples, n_nodes_in_tree)
        all_leaf_indices.append(leaf_idx)
 
    if version == 'lastleaf':
        leaf_matrix = np.column_stack(all_leaf_indices)
    else:
        leaf_matrix = hstack(all_leaf_indices).toarray()  # (n_samples, sum(n_nodes_in_tree))

    encoder = OneHotEncoder(handle_unknown='ignore')
    leaf_hot = encoder.fit_transform(leaf_matrix)
    leaf_hot_dense = leaf_hot.toarray()

    df_plot = get_tsne(crforest, leaf_hot, X_sel, df, feature_selection, path_options_output, version, set_version)
    
    pd.DataFrame(leaf_hot_dense).to_csv(path_options_output/f'leaf_hot_dense_featselect_{feature_selection}_{version}_{set_version}.csv', index = False)
    pd.DataFrame(leaf_matrix).to_csv(path_options_output/f'leaf_matrix_featselect_{feature_selection}_{version}_{set_version}.csv', index = False)
    
    return df_plot, leaf_matrix, leaf_hot_dense


def optimize_model(logger:logging.Logger, 
                   X_train:pd.DataFrame, y_train:np.ndarray, treatment_train:np.ndarray, 
                   X_val:pd.DataFrame, y_val:np.ndarray, treatment_val:np.ndarray, 
                   path:Path|str, rerun_opti:bool = False, 
                   min_feat:int = 15, max_feat:int = 123, 
                   metric:str = 'cate_spearman', num_q:int = 4):
    """Find hyperparameters and optimal features

    Args:
        logger (logging.Logger): logger
        X_train (pd.DataFrame): Profile of individuals in the training set
        y_train (np.ndarray): Cognitive change in the training set
        treatment_train (np.ndarray): Intervention group in the training set
        X_val (pd.DataFrame): Profile of individuals in the validation set
        y_val (np.ndarray):  Cognitive change in the validation set
        treatment_val (np.ndarray): Intervention group in the validation set
        path (Path | str): Path to save outputs
        rerun_opti (bool, optional): If False, use previous optimisation parameters. Defaults to False.
        min_feat (int, optional): Minimal accepted number of features. Defaults to 15.
        max_feat (int, optional): Maximal accepted number of features. Defaults to 123.
        metric (str, optional): Metric for optimisation. Defaults to 'cate_spearman'.
        num_q (int, optional): Number of quantiles. Defaults to 4.

    Returns:
        best_params (np.ndarray): optimal hyperparameters
        best_model: Best model
        results (pd.DataFrame): Results for all hyperparameters combinations
        best_features (np.ndarray): Optimal/selected features
    """
    # hyperparameters to test
    param_grid = {
    'n_estimators': [50,200],#[10, 50,200],
    'max_depth': [10,15,20], #[5,10]
    'min_samples_leaf': [10] # [50, 100]
    }

    if rerun_opti or not os.path.isfile(path/'hyperparameters_optimisation.csv'):
        # Initialisation
        best_score = -float('inf')
        best_model, best_params, best_features = [None] * 3
        results = []

        # for each combination of hyperparameters
        for n_estimators, max_depth, min_samples_leaf in tqdm(product(*param_grid.values())):
            print(f"Optimizing for n_estimators={n_estimators}, max_depth={max_depth}, min_samples_leaf={min_samples_leaf}")
        
            # Step 1: Recursive Feature Elimination based on permutation importance 
            selected_features, res_rfe = recursive_feature_selection(logger, X_train, treatment_train, y_train, X_val, treatment_val, y_val,n_estimators, max_depth, min_samples_leaf, features = X_train.columns,
                                                            min_feat = min_feat, max_feat = max_feat, metric = metric, num_q = num_q)
            X_train_sel = X_train[selected_features].values
            X_val_sel = X_val[selected_features].values

            # Step 2: Refit model on selected features
            model_sel = CausalRandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                criterion="causal_mse",
                random_state=42
            )
            model_sel.fit(X_train_sel, treatment_train, y_train)

            # get predictions and metrics for the training set
            ite_train = model_sel.predict(X_train_sel)

            qini_train  = get_qini_score(ite_train, y_train, treatment_train)
            qini_train = qini_train if isinstance(qini_train, np.float64) else qini_train.iloc[0]
            cate_r2_train, cate_rmse_train = cate_calibration_score(y_train, treatment_train, ite_train)
            cate_spearman_train, pval_spearman_train = cate_quantile_rank_score(y_train, treatment_train, ite_train,n_groups=num_q)

            # get predictions and metrics for the validation set
            ite_val = model_sel.predict(X_val_sel)
            
            qini_val  = get_qini_score(ite_val, y_val, treatment_val)
            qini_val = qini_val if isinstance(qini_val, np.float64) else qini_val.iloc[0]
            cate_r2_val, cate_rmse_val = cate_calibration_score(y_val, treatment_val, ite_val)            
            cate_spearman_val, pval_spearman_val = cate_quantile_rank_score(y_val, treatment_val, ite_val,n_groups=num_q)
            
            # Select the value of the optimisation metric
            if metric == 'cate_spearman':
                score_val = cate_spearman_val
            elif metric == 'cate_r2':
                score_val = cate_r2_val
            else:  
                score_val = qini_val 

            print(f"Params: {n_estimators}, {max_depth}, {min_samples_leaf} → "
                    f"CATE R²: {cate_r2_val:.4f} | Qini: {qini_val:.4f}| CATE Spearman: {cate_spearman_val:.4f}")

            # Save metrics for each number of features saved in res_fre, calculated during RFE
            for i in range(len(res_rfe)):
                results.append({
                    'n_estimators': n_estimators,
                    'max_depth': max_depth if max_depth is not None else -1,
                    'min_samples_leaf': min_samples_leaf,
                    'num_feat_kept': len(selected_features),
                    'rfe_num_features': res_rfe.iloc[i]['n_features'],
                    'rfe_qini_score_val': res_rfe.iloc[i].get('qini_score_val', np.nan),
                    'rfe_cate_r2_val': res_rfe.iloc[i].get('cate_r2_val', np.nan),
                    'rfe_cate_rmse_val': res_rfe.iloc[i].get('cate_rmse_val', np.nan),
                    'qini_score_val': qini_val,
                    'qini_score_train': qini_train,
                    'cate_r2_val': cate_r2_val,
                    'cate_rmse_val': cate_rmse_val,
                    'cate_r2_train': cate_r2_train,
                    'cate_rmse_train': cate_rmse_train,
                    'cate_spearman_val': cate_spearman_val,
                    'cate_spearman_train': cate_spearman_train,
                    'cate_spearman_pval_val': pval_spearman_val,
                    'cate_spearman_pval_train': pval_spearman_train,
                })

            # Save best model and hyperparameters
            if score_val > best_score:
                best_score = score_val
                best_model = model_sel
                best_params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_leaf': min_samples_leaf,
                    'selected_features': selected_features,
                    'num_feat_kept': len(selected_features),
                    'cate_r2_val': cate_r2_val,
                    'cate_rmse_val': cate_rmse_val,
                    'qini_score_val': qini_val,
                    'qini_score_train': qini_train,
                    'cate_spearman_val': cate_spearman_val,
                    'cate_spearman_train': cate_spearman_train,
                    'cate_spearman_pval_val': pval_spearman_val,
                    'cate_spearman_pval_train': pval_spearman_train,
                }
                best_features = selected_features
            #except Exception as e:
            #    print(f"Optimization failed for {n_estimators}, {max_depth}, {min_samples_leaf} → {e}")
            #    continue
        # Save results
        results = pd.DataFrame(results)
        results.to_csv(path/'results_optimisation.csv', index=False)

        # Save best parameters
        best_params = pd.DataFrame([best_params])
        best_params.to_csv(path/'hyperparameters_optimisation.csv', index = False)

        # Save best model
        dump(best_model, path/'best_model_optimisation.joblib')
    else : 
        # If we don't rerun the optimisation, get the saved parameters, model and features
        best_params = pd.read_csv(path/'hyperparameters_optimisation.csv')
        results = pd.read_csv(path/'results_optimisation.csv')
        best_model = load(path/'best_model_optimisation.joblib')
        best_features = best_params['selected_features'].values[0]

        if isinstance(best_features, str):
            best_features = ast.literal_eval(best_features) #If best_features is in the format "[...]" convert it to [...]

    return best_params, best_model, results, best_features


def recursive_feature_selection(logger:logging.Logger,  
                                X_train:pd.DataFrame, treatment_train:np.ndarray, y_train:np.ndarray, 
                                X_val:pd.DataFrame, treatment_val:np.ndarray, y_val:np.ndarray, 
                                n_estimators:int, max_depth:int, min_samples_leaf:int,
                                features:List[str], min_feat:int = 15, max_feat:int = 80, 
                                random_state=42, metric:str="cate_spearman", num_q:int = 4):
    """Recursive Feature Elimination using permutation importance for CausalRandomForestRegressor.
    Keeps the subset of features that gives the highest Qini score on the validation set.

    Args:
        logger (logging.Logger): logger
        X_train (pd.DataFrame): Profile of individuals in the training set
        treatment_train (np.ndarray): Intervention group in the training set
        y_train (np.ndarray): Cognitive change in the training set
        X_val (pd.DataFrame): Profile of individuals in the validation set
        treatment_val (np.ndarray): Intervention group in the validation set
        y_val (np.ndarray):  Cognitive change in the validation set
        n_estimators (int): Number of trees
        max_depth (int): Maximal depth
        min_samples_leaf (int): Minimal number of sample per leaf
        features (List[str]): List of features
        min_feat (int, optional): Minimal accepted number of features. Defaults to 15.
        max_feat (int, optional): Maximal accepted number of features. Defaults to 80.
        random_state (int, optional): Random state. Defaults to 42.
        metric (str, optional): Metric for optimisation. Defaults to 'cate_spearman'.
        num_q (int, optional): Number of quantiles. Defaults to 4.

    """
    selected_features = list(features)
    best_score = -float('inf')
    best_features = selected_features.copy()

    # number of interations 
    n_iter = (len(selected_features) - 50)//5 + 50 - min_feat + 1

    results =  []
    # for each iteration
    for i in tqdm(range(n_iter)):
        # Fit causal forest on current features
        model = CausalRandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            criterion="causal_mse",
            random_state=random_state
        )
        model.fit(X_train[selected_features].values, treatment_train, y_train)

        # get predictions for validation set, removing nan
        ite_val = model.predict(X_val[selected_features])
        nan_mask = np.isnan(ite_val)
        if nan_mask.any():
            logger.info(f"Dropping {nan_mask.sum()} NaN predictions.")
            X_val_filtered = X_val.loc[~nan_mask, selected_features]
            y_val_filtered = y_val[~nan_mask]
            treatment_val_filtered = treatment_val[~nan_mask]
            ite_val = ite_val[~nan_mask]
        else:
            X_val_filtered = X_val[selected_features]
            y_val_filtered = y_val
            treatment_val_filtered = treatment_val

        # If less than the max_feat features selected
        if len(selected_features) < max_feat:
            # Compute the metrics
            cate_r2, cate_rmse = cate_calibration_score(y_val_filtered, treatment_val_filtered, ite_val)
            qini_score_value = get_qini_score(ite_val, y_val_filtered, treatment_val_filtered)
            cate_spearman_val, pval_spearman_val = cate_quantile_rank_score(y_val_filtered, treatment_val_filtered, ite_val,n_groups=num_q)
            
            # select the metric
            if metric == "qini":
                score_val = qini_score_value
            elif metric == "cate_r2":
                score_val = cate_r2
            elif metric == "cate_spearman":
                score_val = cate_spearman_val
            else:
                raise ValueError("metric must be either 'qini' or 'cate_r2'")

            # keep the best
            if score_val > best_score:
                best_score = score_val
                best_features = selected_features.copy()
                logger.info(f"New best {metric} = {best_score:.4f} with {len(best_features)} features")
            # save results
            results.append([len(selected_features), qini_score_value, cate_r2, cate_rmse, cate_spearman_val, pval_spearman_val])
        

        # Permutation importance
        least_important_feats = [None]
        importances = None
        try:
            perm_importance = permutation_importance(
                model,
                X_val_filtered.values,
                y_val_filtered,
                n_repeats=3,
                random_state=random_state,
                scoring=None
            )
            importances = pd.Series(perm_importance.importances_mean, index=selected_features)

            #least_important = importances.idxmin()
            #print(f"Removing least important feature (perm): {least_important}")

        
        except ValueError as e:
            logger.info(f"Permutation importance failed: {e}")
        
            # --- Fallback A: model.feature_importances_ ---
            if importances is None or importances.isna().all():
                if hasattr(model, "feature_importances_"):
                    importances = pd.Series(model.feature_importances_, index=selected_features)
                    importances = importances.fillna(0)
                    logger.info(" Using model.feature_importances_ as fallback.")

            # --- Fallback B: use variance or correlation if still invalid ---
            if importances is None or importances.isna().all() or importances.sum() == 0:
                logger.info(" Using variance/correlation fallback.")
                # Compute correlation with outcome
                corr = X_train[selected_features].corrwith(pd.Series(y_train, index=X_train.index)).abs().fillna(0)
                if corr.sum() == 0:  # all zero correlations → fallback to variance
                    importances = X_train[selected_features].var()
                else:
                    importances = corr
            
        # Remove least important feature           
        if len(selected_features) > 50:
            k = 5  # remove 5 each step until 50 features left
        else: 
            k = 1 # then remove only one per step
        #selected_features.remove(least_important)

        least_important_feats = importances.nsmallest(k).index.tolist()
        selected_features = [f for f in selected_features if f not in least_important_feats]


    return best_features, pd.DataFrame(results, columns=['n_features','qini_score_val', 'cate_r2_val', 'cate_rmse_val', 'cate_spearman_val', 'cate_spearman_pval_val'])


def get_cate_histo(logger:logging.Logger, cate_predicted:np.ndarray, name:str, path:Path, feature_selection:str):
    """Get histogram of predicted CATE.

    Args:
        cate_predicted (np.ndarray): predicted CATE
        name (str): name of the set
        path (Path): Path to save the histograms
        feature_selection (str): Feature pre-selection version
    """

    logger.info(name)
    logger.info("Mean predicted treatment effect (ATE):", cate_predicted.mean())
    logger.info("Std of predicted treatment effect:", cate_predicted.std())

    sns.histplot(cate_predicted, bins=30, kde=True)
    plt.title("Predicted Treatment Effects (CATE)")
    plt.xlabel("Estimated Treatment Effect on Cognitive Change")
    plt.savefig(path/f'predictions_{name}_featselect_{feature_selection}.png', dpi=300, bbox_inches='tight')
    plt.show()


def get_cate_quantile(cate_predicted:np.ndarray, y:np.ndarray, treatment:np.ndarray, name:str, path:Path, feature_selection:str, num_q:int = 5):
    """Get figures 
        - Observed vs. Predicted Treatment Effect by CATE Quantile
        - Cumulative Sensitivity Curve: Observed Treatment Effect vs. Proportion Targeted
        - Cumulative Gain Curve: Proportion of Total Treatment Effect Captured

    Args:
        cate_predicted (np.ndarray): predicted CATE
        y (np.ndarray): cognitive change
        treatment (np.ndarray): treatments
        name (str): name of the set
        path (Path): Path to save the histograms
        feature_selection (str): Feature pre-selection version
        num_q (int, optional): Number of quantiles. Defaults to 5.

    """
    
    df_eval = pd.DataFrame({
        'cate_pred': cate_predicted,
        'outcome': y,
        'treatment': treatment
    })
    # Sort the DataFrame by predicted CATE in descending order
    df_eval = df_eval.sort_values('cate_pred', ascending=False).reset_index(drop=True)

    # Calculate cumulative observed treatment effect
    df_eval['cumulative_treated'] = df_eval['treatment'].cumsum()
    df_eval['cumulative_control'] = (1 - df_eval['treatment']).cumsum()
    df_eval['cumulative_treated_outcome'] = (df_eval['outcome'] * df_eval['treatment']).cumsum()
    df_eval['cumulative_control_outcome'] = (df_eval['outcome'] * (1 - df_eval['treatment'])).cumsum()

    # Calculate cumulative observed treatment effect
    df_eval['cumulative_observed_effect'] = (
        df_eval['cumulative_treated_outcome'] / df_eval['cumulative_treated'] -
        df_eval['cumulative_control_outcome'] / df_eval['cumulative_control']
    )

    # Replace NaN/Inf values (e.g., when cumulative_treated or cumulative_control is zero)
    df_eval['cumulative_observed_effect'] = df_eval['cumulative_observed_effect'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Calculate cumulative proportion of population targeted
    df_eval['cumulative_proportion'] = np.arange(1, len(df_eval) + 1) / len(df_eval)
    
    # Calculate total possible treatment effect
    total_treated_outcome = df_eval['outcome'][df_eval['treatment'] == 1].sum()
    total_control_outcome = df_eval['outcome'][df_eval['treatment'] == 0].sum()
    total_possible_effect = (total_treated_outcome / df_eval['treatment'].sum()) - (total_control_outcome / (len(df_eval) - df_eval['treatment'].sum()))

    # Calculate cumulative gain
    df_eval['cumulative_gain'] = (
        (df_eval['cumulative_treated_outcome'] / df_eval['cumulative_treated']) -
        (df_eval['cumulative_control_outcome'] / df_eval['cumulative_control'])
    ) / total_possible_effect


    df_eval['cate_quantile'] = pd.qcut(df_eval['cate_pred'], q=num_q, labels=False)
    results = []
    for q in sorted(df_eval['cate_quantile'].unique()):
        group = df_eval[df_eval['cate_quantile'] == q]
        treated = group[group['treatment'] == 1]['outcome']
        control = group[group['treatment'] == 0]['outcome']

        if len(treated) > 0 and len(control) > 0:
            observed_effect = treated.mean() - control.mean()
        else:
            observed_effect = np.nan  # insufficient data
        
        results.append({
            'Quantile': q,
            'Observed Treatment Effect': observed_effect,
            'Avg Predicted CATE': group['cate_pred'].mean(),
            'Size': len(group)
        })
    df_results = pd.DataFrame(results)

    # Plot observed treatment effect by predicted CATE quantile
    plt.figure(figsize=(8, 5))
    plt.plot(df_results['Quantile'], df_results['Observed Treatment Effect'], marker='o', label='Observed Effect')
    plt.plot(df_results['Quantile'], df_results['Avg Predicted CATE'], marker='x', linestyle='--', label='Avg Predicted CATE')
    plt.title("Observed vs. Predicted Treatment Effect by CATE Quantile")
    plt.xlabel("Predicted CATE Quantile (0 = lowest, 4 = highest)")
    plt.ylabel("Treatment Effect on Cognitive Change")
    plt.legend()
    plt.grid(True)
    plt.savefig(path/f'quarter_split_{name}_featselect_{feature_selection}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


    # Plot cumulative sensitivity curve
    plt.figure(figsize=(8, 5))
    plt.plot(df_eval['cumulative_proportion'], df_eval['cumulative_observed_effect'], label='Cumulative Observed Effect', color='blue')
    plt.axhline(y=0, color='gray', linestyle='--')  # Baseline (no effect)
    plt.title("Cumulative Sensitivity Curve: Observed Treatment Effect vs. Proportion Targeted")
    plt.xlabel("Proportion of Population Targeted (by Predicted CATE)")
    plt.ylabel("Cumulative Observed Treatment Effect")
    plt.legend()
    plt.grid(True)
    plt.savefig(path/f'cumulative_sensitivity_{name}_featselect_{feature_selection}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot cumulative gain curve
    plt.figure(figsize=(8, 5))
    plt.plot(df_eval['cumulative_proportion'], df_eval['cumulative_gain'], label='Cumulative Gain', color='green')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Baseline (Random Targeting)')
    plt.title("Cumulative Gain Curve: Proportion of Total Treatment Effect Captured")
    plt.xlabel("Proportion of Population Targeted (by Predicted CATE)")
    plt.ylabel("Proportion of Total Treatment Effect Captured")
    plt.legend()
    plt.grid(True)
    plt.savefig(path/f'cumulative_gain_{name}_featselect_{feature_selection}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return df_results


def get_impurity_importance(model:CausalRandomForestRegressor, feature_names:List[str], path:Path, name:str, 
                            feature_selection:str, top_n:int = 25 ):
    """Get impurity importance

    Args:
        model (CausalRandomForestRegressor): Causal forest model
        feature_names (List[str]): List of features
        path (Path): Path to save outputs
        name (str): name of the dataset
        feature_selection (str): List of features
        top_n (int, optional): Number of features to print. Defaults to 25.

    """

    #! Impurity based importance
    df_importances = pd.DataFrame({'forest': model.feature_importances_,
                            'feature': feature_names})
    df_importances['std'] = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

    # select the top_n features
    df_importances_forest = df_importances.sort_values('forest', ascending=False).head(top_n)

    # plot the impurity importance for the top_n features
    fig, ax = plt.subplots()
    df_importances_forest['forest'].plot.bar(yerr=df_importances_forest['std'] , ax=ax)
    ax.set_title("Causal Forest feature importances")
    ax.set_ylabel("Mean decrease in impurity")
    ax.set_xticklabels(df_importances_forest['feature'], rotation=45, fontsize=6)
    plt.savefig(path/f'impurity_features_{name}_featselect_{feature_selection}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_importances[['feature','forest']].rename(columns={'forest':'Impurity'})


def get_permutation_importance(model:CausalRandomForestRegressor, X_train, y_train, feature_names:List[str], path:Path, name:str, 
                            feature_selection:str, top_n:int = 25, n_repeats:int = 50 ):
    """Get permutation importance

    Args:
        model (CausalRandomForestRegressor): Causal forest model
        X_train (_type_): data
        y_train (_type_): cognitive change
        feature_names (List[str]): List of features
        path (Path): Path to save outputs
        name (str): name of the dataset
        feature_selection (str): List of features
        top_n (int, optional): Number of features to print. Defaults to 25.
        n_repeats (int, optional): Number of repeats. Defaults to 50.

    """

    # Permutation-based feature importance
    imp = permutation_importance(model, X_train, y_train,
                                n_repeats=n_repeats,
                                random_state=0)
    
    df_imp = pd.DataFrame(np.array([imp['importances_mean'],imp['importances_std'], feature_names]).T, columns=['importances_mean','importances_std', 'features'])
    df_imp['importances_mean'] = df_imp['importances_mean'].astype('float')
    df_imp['importances_std'] = df_imp['importances_std'].astype('float')
    df_imp['importance_mean_abs'] = abs(df_imp['importances_mean'])

    
    # select the top_n features
    imp_selected = df_imp.sort_values('importance_mean_abs', ascending=False).head(top_n)

    # plot the permutation importance for the top_n features
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_title(f"Causal Forest permutation feature importances")
    ax.set_ylabel("Mean decrease in impurity")
    plt.bar(imp_selected['features'],- imp_selected['importances_mean'].astype('float'), yerr=imp_selected['importances_std'].astype('float'))
    ax.set_xticklabels(imp_selected['features'], rotation=45, fontsize=8)
    plt.savefig(path/f'permutation_features_{name}_featselect_{feature_selection}.png', dpi=300, bbox_inches='tight')
    plt.show()
         
    return df_imp[['features','importances_mean']].rename(columns={'importances_mean':'Permutation','features':'feature'})


def get_shap(model:CausalRandomForestRegressor, X, path:Path, name:str, feature_selection:str):
    """Get SHAP importance

    Args:
        model (CausalRandomForestRegressor): model
        X (_type_): data
        path (Path): Path to save outputs
        name (str): name of the dataset
        feature_selection (str): List of features

    """
    cforest_explainer = shap.TreeExplainer(model)

    # Expected values for treatment=0 and treatment=1. i.e. Y|X,T=0 and Y|X,T=1
    print(f"Expected value: {cforest_explainer.expected_value}") #[0.15635275, 0.20793578]
    
    shap_values = cforest_explainer.shap_values(X)
    SHAP_interv = pd.DataFrame(np.array([abs(pd.DataFrame(shap_values[:, :, 1])).sum(axis=0),X.columns]).T, columns=['SHAP_intervention', 'feature'])
    SHAP_control = pd.DataFrame(np.array([abs(pd.DataFrame(shap_values[:, :, 0])).sum(axis=0),X.columns]).T, columns=['SHAP_control', 'feature'])

    # plot SHAP values for control 
    plt.figure()
    shap.summary_plot(shap_values[:, :, 0], X,show=False)    
    #ax.set_title("Causal Forest SHAP  values for training data considering control group")
    plt.savefig(path/f'shap_{name}_control_featselect_{feature_selection}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # plot SHAP values for intervention
    plt.figure()
    shap.summary_plot(shap_values[:, :, 1], X,show=False)    
    #plt.title("Causal Forest SHAP  values for training data considering intervention group")
    plt.savefig(path/f'shap_{name}_interv_featselect_{feature_selection}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # SHAP values for treatment effect
    shap_cate_values = shap_values[:, :, 1] - shap_values[:, :, 0]
    SHAP_diff = pd.DataFrame(np.array([abs(pd.DataFrame(shap_cate_values)).sum(axis=0),X.columns]).T, columns=['SHAP_diff', 'feature'])

    # Summary plot for treatment effect
    plt.figure()
    shap.summary_plot(shap_cate_values, X , plot_type="dot",show=False)
    plt.savefig(path/f'shap_{name}_diff_featselect_{feature_selection}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

        
    shap_control = pd.DataFrame(shap_values[:, :, 0], columns=X.columns)
    shap_control.to_csv(path/f'shap_values_control_{name}_featselect_{feature_selection}.csv', index = False)

    shap_intervention = pd.DataFrame(shap_values[:, :, 1], columns=X.columns)
    shap_intervention.to_csv(path/f'shap_values_intervention_{name}_featselect_{feature_selection}.csv', index = False)

    return SHAP_interv, SHAP_control, SHAP_diff, shap_values
    

def get_importance_ranking(dict_importance_list:dict, path:Path, name:str):
    """Get importance ranking according to SHAP value, impurity and permutation

    Args:
        dict_importance_list (dict): dictionary with SHAP value, impurity and permutation 
        path (Path): Path to save outputs
        name (str): name of the dataset

    """
    df_importance_list = dict_importance_list['Permutation']
    for key in dict_importance_list.keys():
        if key != 'Permutation':
            df_importance_list = pd.merge(df_importance_list, dict_importance_list[key],how = 'outer')

    df_importance_rank = df_importance_list.copy().reset_index(drop=True)
    for c in df_importance_rank.columns:
        if c != 'feature':
            if c == 'Permutation':
                df_importance_rank[c] = df_importance_list[c].abs().rank(ascending=False, method='min')
            else: 
                df_importance_rank[c] = df_importance_list[c].rank(ascending=False, method='min')
    df_importance_rank['total_rank'] = df_importance_rank[['Permutation','SHAP_intervention','Impurity']].sum(axis=1)
    
    df_importance_rank['total_above_20_rank'] = 0
    for c in ['Permutation','SHAP_intervention','Impurity']:
        df_importance_rank['total_above_20_rank'] += (df_importance_rank[c]<=20).astype(int)

    df_importance_rank = df_importance_rank.sort_values(by="total_rank")
    df_importance_rank.to_csv(path/f"rank_{name}_importance.csv")
    df_importance_list.to_csv(path/f"values_{name}_importance.csv")

    return df_importance_rank, df_importance_list


def get_outcome_per_level(logger:logging.Logger, feature_name_c:str, cate_prediction:np.ndarray, df_train:pd.DataFrame, path:Path):
    """Get cognitive change according to the feature level

    Args:
        logger (logging.Logger): logger
        feature_name_c (str): feature
        cate_prediction (np.ndarray): predicted CATE
        df_train (pd.DataFrame): intervention group and cognitive change
        path (Path): Path to save outputs
    """

    X = pd.DataFrame(np.array([cate_prediction, df_train[feature_name_c], df_train['group'], df_train['ztotal_composite_0y'], df_train['ztotal_composite_2y'], df_train['ztotal_composite_change']]).T, columns = ['CATE', feature_name_c, 'group','cognition_0y','cognition_2y','cognitive_change'])
    
    # Plot CATE vs. feature
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=X, x=feature_name_c, y="CATE", alpha=0.6, hue='cognition_0y',palette="coolwarm")
    plt.axhline(0, color="red", linestyle="--", label="Zero Effect")
    plt.title(f"Predicted CATE vs. Baseline {feature_name_c}")
    plt.xlabel(f"Baseline {feature_name_c}")
    plt.ylabel("Conditional Average Treatment Effect (CATE)")
    plt.legend()
    plt.grid(True)
    plt.savefig(path/f'cate_vs_{feature_name_c}.png', dpi=300, bbox_inches='tight')
    plt.show()


    # Compare outcomes by treatment and the feature  ---
    # Split data into low and high feature groups (e.g., median split)
    median_feature_a = X[feature_name_c].median()
    if feature_name_c == 'nulisa_BDNF_0y':
        low_thr = 10
        low_feature_a = X[feature_name_c] <= low_thr
        medium_feature_a = (X[feature_name_c] <= median_feature_a) & (X[feature_name_c] > low_thr)
    else:
        low_feature_a = X[feature_name_c] <= median_feature_a
    high_feature_a = X[feature_name_c] > median_feature_a


    # Calculate mean cognitive change for treatment and control groups
    if feature_name_c == 'nulisa_BDNF_0y':
        very_low_treated = X[(X['group']=='Intervention') & (X[feature_name_c] <= low_thr)]['cognitive_change']
        very_low_control = X[(X['group']=='Control') & (X[feature_name_c] <= low_thr)]['cognitive_change']
        low_treated = X[(X['group']=='Intervention') & (X[feature_name_c] > low_thr) & (X[feature_name_c] <= median_feature_a)]['cognitive_change']
        low_control = X[(X['group']=='Control') & (X[feature_name_c] > low_thr) & (X[feature_name_c] <= median_feature_a)]['cognitive_change']
    else:
        low_treated = X[(X['group']=='Intervention') & (X[feature_name_c] <= median_feature_a)]['cognitive_change']
        low_control = X[(X['group']=='Control') & (X[feature_name_c] <= median_feature_a)]['cognitive_change']
    high_treated = X[(X['group']=='Intervention') & (X[feature_name_c] > median_feature_a)]['cognitive_change']
    high_control = X[(X['group']=='Control') & (X[feature_name_c] > median_feature_a)]['cognitive_change']

    logger.info(f"\n--- Cognitive Change by {feature_name_c} and Treatment, median {median_feature_a}---")
    if feature_name_c == 'nulisa_BDNF_0y':
        logger.info(f"Very Low {feature_name_c}- Treated: {np.mean(very_low_treated):.3f}, Control: {np.mean(very_low_control):.3f}")
    logger.info(f"Low {feature_name_c}- Treated: {np.mean(low_treated):.3f}, Control: {np.mean(low_control):.3f}")
    logger.info(f"High {feature_name_c} - Treated: {np.mean(high_treated):.3f}, Control: {np.mean(high_control):.3f}")

    # Calculate treatment effect for each group
    if feature_name_c == 'nulisa_BDNF_0y':
        verylow_effect = np.mean(very_low_treated) - np.mean(very_low_control)

        logger.info(f"\nIntervention  Effect for Very Low {feature_name_c}: {verylow_effect:.3f}")
        # --- Step 5: Interpretation ---
        if verylow_effect < -0.01:
            logger.info(f"Intervention is HARMFUL or LESS BENEFICIAL for individuals with very low {feature_name_c} (< {round(low_thr,2)}).")
        elif verylow_effect > 0.01:
            logger.info(f"Intervention is BENEFICIAL for individuals with very low {feature_name_c}(< {round(low_thr,2)}).")
        else:
            logger.info(f"No clear Intervention  effect for individuals with very low {feature_name_c}(< {round(low_thr,2)}).")
    
    low_effect = np.mean(low_treated) - np.mean(low_control)
    high_effect = np.mean(high_treated) - np.mean(high_control)

    logger.info(f"\nIntervention  Effect for Low {feature_name_c}: {low_effect:.3f}")
    # --- Step 5: Interpretation ---
    if low_effect < -0.01:
        logger.info(f"Intervention is HARMFUL or LESS BENEFICIAL for individuals with low {feature_name_c} (< {round(median_feature_a,2)}).")
    elif low_effect > 0.01:
        logger.info(f"Intervention is BENEFICIAL for individuals with low {feature_name_c}(< {round(median_feature_a,2)}).")
    else:
        logger.info(f"No clear Intervention  effect for individuals with low {feature_name_c}(< {round(median_feature_a,2)}).")
    
    logger.info(f"\nIntervention  Effect for High {feature_name_c}: {high_effect:.3f}")
    if high_effect < -0.01:
        logger.info(f"Intervention is HARMFUL or LESS BENEFICIAL for individuals with high {feature_name_c}(> {round(median_feature_a,2)}).")
    elif high_effect > 0.01:
        logger.info(f"Intervention is BENEFICIAL for individuals with high {feature_name_c}(> {round(median_feature_a,2)}).")
    else:
        logger.info(f"No clear Intervention  effect for individuals with high {feature_name_c}(> {round(median_feature_a,2)}).")


def get_features_pre_selection(logger:logging.Logger, feature_selection:str, df_train_val:pd.DataFrame, df:pd.DataFrame, 
                               path_options_output:Path, col_nulisa_baseline:List, cofactors_to_test:List, cognition:str, col_cofacteurs_base:List ):
    """Pre-selection.
     - correlation_me: Spearman correlation to create clusters and Mixed-effect to select features within the cluster
     - correlation: Spearman correlation to create clusters and select the feature within the cluster by selecting the one with the smallest p-value for spearman test.

    Args:
        logger (logging.Logger): logger
        feature_selection (str): version of feature selection. Choices: 'correlation', 'correlation_me'
        df_train_val (pd.DataFrame): data, group and cognitive change for train +  validation set
        df (pd.DataFrame):  data, group and cognitive change for full dataset
        path_options_output (Path): Path to save output
        col_nulisa_baseline (List): List of baseline Nulisa columns
        cofactors_to_test (List): List of cofactors to add in the list of features
        cognition (str): Cognitive score name
        col_cofacteurs_base (List): List of cofactors for Mixed-effect model

    """
    
    # list of input features
    #if feature_selection == 'bbm_evol':
    #    df_feature_selection = pd.read_csv(path_options_input/"list_bbm_effect_evol_pval.csv")
    #    bbm_list = df_feature_selection['bbm']
    #    col_nulisa_baseline = [f"{b}_0y" for b in bbm_list]
    
    if feature_selection in ['correlation', 'correlation_me']:

        df_corr = df_train_val[col_nulisa_baseline + cofactors_to_test]

        # ! Get correlation matrix (Spearman) and associated annotations
        corr, pvals, annot = calculate_anot_spearman(df_corr)
        corr_mask = np.triu(np.ones_like(corr, dtype=bool)) # Matrix to keep only numbers below the diagonal
        corr = np.where(corr_mask ,np.nan,corr)
        corr = pd.DataFrame(corr, index=df_corr.columns, columns=df_corr.columns)
        corr_high = corr.mask(corr<0.8,np.nan).dropna(how = 'all')

        non_null = corr_high.stack().reset_index()
        non_null.columns = ['bbm_1', 'bbm_2', 'correlation']           
            
        corr_high.to_csv(path_options_output/f'preselection_correlation__featselect_{feature_selection}.csv', index=True)

        # Create graph
        G = nx.Graph()
        G.add_edges_from(non_null[['bbm_1','bbm_2']].values)

        # Find connected components
        clusters = [list(component) for component in nx.connected_components(G)]

        # Create dictionary
        cluster_dict = {f'cluster_{i+1}': sorted(members) for i, members in enumerate(clusters)}

        logger.info(cluster_dict)
        
        df_clusters_correlation = {}
        if feature_selection == 'correlation':
            df_corr_cognition = df[col_nulisa_baseline + cofactors_to_test + [f"{cognition}_change"]]

            for cluster in cluster_dict.values():
                results = []
                for bm in cluster:
                    r, p = spearmanr(df_corr_cognition[bm], df_corr_cognition[f"{cognition}_change"], nan_policy='omit')
                    results.append({'biomarker': bm, 'rho': r, 'pval': p})

                corr_df = pd.DataFrame(results)

                logger.info(cluster)
                logger.info(corr_df)

                kept = corr_df[corr_df['pval'] == corr_df['pval'].min()]['biomarker']
                removed = corr_df[corr_df['biomarker'] != kept.iloc[0]]['biomarker'].tolist()
                df_clusters_correlation[kept.iloc[0]] = list(set(removed))

        if feature_selection == 'correlation_me':
            all_items = list(set([v for values in cluster_dict.values() for v in values]))

            df_me = af.mixedeffect_multiple_predictors(df = df_train_val,
                                outcome=cognition,
                                predictors_list = [s.rsplit('_', 1)[0] for s in all_items],
                                second_predictor = 'group',
                                quadratic = False,
                                outcome_years = [0, 1, 2],
                                predictor_years = [0],
                                name_file = 'bbm',
                                horizontal = True,
                                path = path_options_output,
                                cofactors = col_cofacteurs_base, 
                                group_ref = 'Control')
            
            df_me_quadra = af.mixedeffect_multiple_predictors(df = df_train_val,
                                outcome=cognition,
                                predictors_list = [s.rsplit('_', 1)[0] for s in all_items],
                                second_predictor = 'group',
                                quadratic = True,
                                outcome_years = [0, 1, 2],
                                predictor_years = [0],
                                name_file = 'bbm',
                                horizontal = True,
                                path = path_options_output,
                                cofactors = col_cofacteurs_base, 
                                group_ref = 'Control')
            
            df_me.reset_index(drop=False, inplace=True)
            df_me.rename(columns={'Unnamed: 0':'bbm',
                                'index':'bbm'}, inplace=True)
            df_me['bbm'] = [f"{c}_0y" for c in df_me['bbm']]
            
            df_me_quadra.reset_index(drop=False, inplace=True)
            df_me_quadra.rename(columns={'Unnamed: 0':'bbm',
                                'index':'bbm'}, inplace=True)
            df_me_quadra['bbm'] = [f"{c}_0y" for c in df_me_quadra['bbm']]

            for cluster_name, cluster in cluster_dict.items():
                logger.info(cluster)
                df_me_temp = df_me[df_me['bbm'].isin(cluster)][['bbm','group:bm:year pvalue']]
                df_me_quadra_temp = df_me_quadra[df_me_quadra['bbm'].isin(cluster)][['bbm','group:bm:year2 pvalue']]

                dftemp = pd.merge(df_me_temp, df_me_quadra_temp, on='bbm', how='outer')
                logger.info(dftemp)
                dftemp.to_csv(path_options_output/f'mixed_effects_cluster_{cluster_name}_featselect_{feature_selection}.csv', index=False)

                df_me_temp = df_me_temp.rename(columns={'group:bm:year pvalue':'pvalue'})
                df_me_quadra_temp = df_me_quadra_temp.rename(columns={'group:bm:year2 pvalue':'pvalue'})

                df_me_merged = pd.concat([df_me_temp, df_me_quadra_temp])

                kept = df_me_merged[df_me_merged['pvalue'] == df_me_merged['pvalue'].min()]['bbm']
                removed = df_me_merged[df_me_merged['bbm'] != kept.iloc[0]]['bbm'].tolist()
                df_clusters_correlation[kept.iloc[0]] = list(set(removed))

    feature_names = col_nulisa_baseline + cofactors_to_test
    logger.info(f" Number of features before selection: {len(feature_names)}")

    if feature_selection in ['correlation', 'correlation_me']:
        for kept, removed in df_clusters_correlation.items():
            for r in removed:
                feature_names.remove(r)
    logger.info(f" Number of features after selection: {len(feature_names)}")

    return feature_names



#! ********************************
#! *************** Unused functions
#! ********************************


def basu_feature_selection(X:np.ndarray, ite_preds:np.ndarray, p_threshold:float=0.1):
    """Performs Basu-style feature selection by regressing ITEs on covariates.

    Args:
        X (np.ndarray): _description_
        ite_preds (np.ndarray): _description_
        p_threshold (float, optional): threshold for pvalue. Defaults to 0.1.

    """
    X_df = pd.DataFrame(X, columns=X.columns)

    X_const = sm.add_constant(X_df)
    ols_model = sm.OLS(ite_preds, X_const).fit()

    pvals = ols_model.pvalues.drop('const', errors='ignore')

    selected_features = pvals[pvals < p_threshold].index.tolist()

    return X_df[selected_features].values, selected_features



def train_model(X_train:pd.DataFrame, y_train:np.ndarray, treatment_train:np.ndarray, 
                best_params:pd.Series, path:Path, rerun_model:bool = False):
    """Train the model

    Args:
        X_train (pd.DataFrame): Data
        y_train (np.ndarray): cognitive score
        treatment_train (np.ndarray): treatment/intervention group
        best_params (pd.Series): Selected hyperparameters
        path (Path): Path to save the model
        rerun_model (bool, optional): If True, rerun the model. Defaults to False.

    """
    if rerun_model or not os.path.isfile(path/'causal_random_forest_model.joblib'):
        crforest = CausalRandomForestRegressor(criterion="causal_mse",
                                        min_samples_leaf=best_params['min_samples_leaf'],
                                        max_depth=best_params['max_depth'],
                                        n_estimators=best_params['n_estimators'])
        crforest.fit(X=X_train, y=y_train, treatment=treatment_train)
        dump(crforest, path/'causal_random_forest_model.joblib')

    else : 
        crforest = load(path/'causal_random_forest_model.joblib')
    
    return crforest
