##############################
# Purpose: Elastic Net

# Date creation: 2025-10-18
# Author: ludivinemv

#Input data:
#    dataset_original.csv


#Output data:
#    (see the output folders defined in ``__main__``)

##############################


# ! PUBLIC IMPORTS
import ast
import logging
import os
import pickle
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from scipy.stats import loguniform, uniform
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

# ! PRIVATE IMPORTS
from utils.log_functions import setup_logger
from utils.data_transform import windsorize_column, clean_open_file

# ! GLOBAL VARIABLES
from projectspecific.config import (
    DATA_PROCESSED_PATH,
    LOG_PATH,
    LONG_PATH,
)

VERSION = "2-0"
GROUP_CLASS = "intervention"

N_FEATURES_TO_PLOT = 25
SHRINK_THRESHOLD = 1e-8
MIN_NON_ZERO_FEATURES = 5


# ! FUNCTIONS


def _clean_feature_name(name):
    """Remove the Nulisa prefix and time-point suffixes of a feature name.

    Args:
        name (str): Raw feature name, e.g. ``nulisa_APOE4_0y`` or
            ``nulisa_APOE4_0y_x_group``.

    Returns:
        str: Shortened feature name, e.g. ``APOE4`` or ``APOE4_group``.
    """
    return (
        name.removeprefix("nulisa_")
        .removesuffix("_0y")
        .replace("_0y_x_group", "_group")
    )


def _get_custom_colors():
    """Build the list of 25 colors used for the shrinkage path plots.

    Returns:
        list: The first 25 colors of the concatenated ``tab20``,
            ``tab20b`` and ``tab20c`` colormaps.
    """
    all_colors = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = colormaps[cmap_name].resampled(20)
        all_colors += [cmap(i) for i in range(20)]
    return all_colors[:25]


def _get_fold_data(
    X, y, train_ids, test_ids, base_feature_names, windsor_columns,
    windsor_ratio, add_interactions=False,
):
    """Split the data and preprocess it using the training set only.

    The preprocessing is fitted on the training individuals and then
    applied to the test individuals:
        1. Winsorization of ``windsor_columns``. The limits are those of
           the winsorized training data (its min and max); the test values
           are clipped to these limits.
        2. Standardization (mean and standard deviation of the training
           set).
        3. Optional interaction terms ``<feature>_x_group``
           (scaled feature * ``group_binary``).

    Args:
        X (pd.DataFrame): Raw features, with an ``ID`` column (and a
            ``group_binary`` column if ``add_interactions``).
        y (pd.DataFrame): Target, with an ``ID`` column.
        train_ids (pd.Series): IDs of the training individuals.
        test_ids (pd.Series): IDs of the test individuals.
        base_feature_names (list[str]): Features of ``X`` used as inputs.
        windsor_columns (list[str]): Columns to winsorize.
        windsor_ratio (float): Percentage winsorized at each tail.
        add_interactions (bool, optional): Add the ``_x_group`` interaction
            terms. Defaults to False.

    Returns:
        tuple: ``(X_train, y_train, X_test, y_test)``. ``X_train`` and
            ``X_test`` are scaled DataFrames (no ``ID``); ``y_train`` and
            ``y_test`` are 1-D numpy arrays.
    """
    train_mask = X['ID'].isin(list(train_ids))
    test_mask = X['ID'].isin(test_ids)
    X_train = X[train_mask].copy()
    X_test = X[test_mask].copy()

    # Winsorize: limits learnt on the training set only
    ratio = windsor_ratio / 100
    for col in windsor_columns:
        X_train[col] = windsorize_column(X_train[col], ratio, ratio)
        X_test[col] = X_test[col].clip(
            X_train[col].min(), X_train[col].max()
        )

    # Scale: mean and std learnt on the training set only
    X_train_scaled, X_test_scaled = scaling(
        X_train[base_feature_names], X_test[base_feature_names],
        base_feature_names,
    )

    if add_interactions:
        group_train = X_train['group_binary'].to_numpy()
        group_test = X_test['group_binary'].to_numpy()
        for feature in base_feature_names:
            X_train_scaled[f"{feature}_x_group"] = (
                X_train_scaled[feature].to_numpy() * group_train
            )
            X_test_scaled[f"{feature}_x_group"] = (
                X_test_scaled[feature].to_numpy() * group_test
            )

    y_train = y[y['ID'].isin(train_ids)].drop('ID', axis=1)
    y_train = y_train.iloc[:, 0].to_numpy()
    y_test = y[y['ID'].isin(test_ids)].drop('ID', axis=1)
    y_test = y_test.iloc[:, 0].to_numpy()
    return X_train_scaled, y_train, X_test_scaled, y_test


def _save_figure(output_dir, stem, extensions=('png',)):
    """Save the current matplotlib figure in one or several formats.

    Args:
        output_dir (pathlib.Path): Folder where the figure is saved.
        stem (str): File name without extension.
        extensions (tuple[str], optional): File extensions to write.
            Defaults to ``('png',)``.
    """
    for extension in extensions:
        plt.savefig(output_dir / f'{stem}.{extension}', bbox_inches='tight')


def get_data():
    """Open the data and metadata, standardize them and select mITT.

    Steps:
        - Open data and metadata.
        - Standardize column names.
        - Find the patients from the mITT subgroup.

    Note:
        The data are not winsorized nor scaled here: both are done on the
        training set of each fold (see ``_get_fold_data``).

    Returns:
        tuple: ``(df, col_cofacteurs_base, cofactors_to_test, col_nulisa,
            col_nulisa_baseline)`` where ``df`` is the standardized data
            restricted to mITT and the other items are lists of column
            names.
    """
    # open meta and data files
    df = clean_open_file(
        path=DATA_PROCESSED_PATH / 'dataset_original.csv',
        output_type='dataframe',
    )
    meta = clean_open_file(
        path=DATA_PROCESSED_PATH / 'description' / 'columns_description.csv',
        output_type='dataframe',
    )

    # Get the string of a dictionary and convert it to a dictionary
    label_map = ast.literal_eval(
        meta[meta['column'] == 'sub_sample_id']['values'].values[0]
    )
    # get Intervention and Control instead of 2 and 3 in df['group']
    df['group'] = df['sub_sample_id'].replace(label_map)

    # standardize cognition columns names
    col_cognitive = [
        item for item in df.columns if item.lower().startswith('z')
    ]
    cogn_list = set([s.rsplit('_', 1)[0] for s in col_cognitive])

    for cogn in cogn_list:
        df[f'{cogn}_change'] = df[f'{cogn}_2y'] - df[f'{cogn}_0y']

    # get the patients in the mITT subgroup
    df['mITT'] = np.where(
        (df['ztotal_composite_1y'].notna())
        | (df['ztotal_composite_2y'].notna()),
        1,
        0,
    )

    meta = pd.concat(
        [
            meta,
            pd.DataFrame(
                [
                    ['mITT',
                     'Individuals from the modified intention-to-treat '
                     'group',
                     np.nan, 'int64', 0, 0, 0, 0],
                    ['age_1y', 'Age at 1 year', 'years', 'float64',
                     1, 0, 1, 0],
                    ['age_2y', 'Age at 2 years', 'years', 'float64',
                     1, 0, 0, 1],
                    ['group', 'Intervention group', np.nan, np.nan,
                     0, 0, 0, 0],
                ],
                columns=meta.columns,
            ),
        ],
        ignore_index=True,
    )

    nulisa_col = meta[
        meta['column'].str.startswith('nulisa_')
    ]['column'].tolist()

    for c in nulisa_col:
        # NB: chained assignment, kept as in the original code.
        meta.loc[meta['column'] == c, 'description'] = (
            f"Nulisa {c.rsplit('_', 2)[1]} "
            f"at year {c.rsplit('_', 2)[2].replace('y', '')}"
        )

    for c in ["nulisa_APOE4", "nulisa_CHIT1", "nulisa_GDF15"]:
        # Remove {c}_0y and {c}_2y
        df.drop(columns=[f"{c}_0y", f"{c}_2y"], inplace=True, errors='ignore')

        # Rename {c}_0_halfmin to {c}_0y and {c}_2_halfmin to {c}_2y
        df.rename(
            columns={
                f"{c}_0_halfmin": f"{c}_0y",
                f"{c}_2_halfmin": f"{c}_2y",
            },
            inplace=True,
            errors='ignore',
        )

    df['diabetes_0y'] = np.where(df['diabetes_ever_no_0y'] == 1, 0, 1)
    df = df[df['mITT'] == 1]

    # ! Lists ##########################################################
    # basic cofactors
    col_cofacteurs_base = ['age_0y', 'sex', 'educ_years_0y', 'campid']

    # cofactors to evaluate
    cofactors_to_test = [
        'sex',
        'age_0y',
        'educ_years_0y',
        'bmi_0y',
        'campid',
    ]

    # List of Nulisa biomarkers columns
    col_nulisa = [
        item for item in df.columns if item.lower().startswith('nulisa')
    ]

    col_nulisa_baseline = [s for s in col_nulisa if s.endswith('_0y')]

    return (
        df,
        col_cofacteurs_base,
        cofactors_to_test,
        col_nulisa,
        col_nulisa_baseline,
    )


def scaling(X_train, X_test, feature_names):
    """Standardize the features using the training set statistics.

    Args:
        X_train (pd.DataFrame | np.ndarray): Training features; the scaler
            is fitted on them.
        X_test (pd.DataFrame | np.ndarray): Test features; transformed with
            the training mean and standard deviation.
        feature_names (list[str]): Column names of the returned DataFrames.

    Returns:
        tuple: ``(X_train_scaled, X_test_scaled)``, two DataFrames with
            ``feature_names`` as columns.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names
    )
    return X_train_scaled, X_test_scaled


def _compute_shrink_paths(X_train, y_train, feature_names, l1_ratio):
    """Fit ElasticNet models over a grid of alphas to get shrinkage paths.

    Args:
        X_train (pd.DataFrame): Training features (without ``ID``).
        y_train (np.ndarray): Training target (1-D).
        feature_names (list[str]): Names of the features.
        l1_ratio (float): ``l1_ratio`` used for all the fitted models.

    Returns:
        tuple: ``(coef_df, shrink_alpha)`` where ``coef_df`` holds one row
            of coefficients per alpha (plus ``alpha`` and ``log10_alpha``
            columns) and ``shrink_alpha`` maps each feature to the largest
            ``log10(alpha)`` for which it is still non-zero.
    """
    alpha_values_to_plot = np.logspace(-4, 0, 75)
    all_coefs = []
    all_alpha = []

    for alpha_val in alpha_values_to_plot:
        temp_model = ElasticNet(
            alpha=alpha_val, l1_ratio=l1_ratio,
            max_iter=10000, random_state=42, selection='random',
        )
        temp_model.fit(X_train, y_train)
        all_coefs.append(temp_model.coef_)
        all_alpha.append(alpha_val)

    coef_df = pd.DataFrame(all_coefs, columns=feature_names)
    coef_df['alpha'] = all_alpha
    coef_df['log10_alpha'] = np.log10(coef_df['alpha'])

    shrink_alpha = {}
    for feature in feature_names:
        max_abs_coef = np.abs(coef_df[feature]).max()
        if max_abs_coef > 0:
            threshold = SHRINK_THRESHOLD * max_abs_coef
            shrink_point = coef_df[
                np.abs(coef_df[feature]) >= threshold
            ]['log10_alpha'].max()
            shrink_alpha[feature] = shrink_point
        else:
            shrink_alpha[feature] = coef_df['log10_alpha'].min()

    return coef_df, shrink_alpha


def _plot_cv_mse_vs_alpha(results, best_alpha, best_score, fold):
    """Plot train and validation MSE against alpha (from the inner CV).

    The figure is left open: the caller saves and closes it.

    Args:
        results (pd.DataFrame): ``cv_results_`` of the random search.
        best_alpha (float): Alpha of the selected model.
        best_score (float): ``best_score_`` of the random search.
        fold (int): Fold index (used in the title).
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(
        np.log10(results['param_alpha']),
        -results['mean_train_score'],
        alpha=0.5, color='blue', label='Train MSE (CV)',
    )
    plt.scatter(
        np.log10(results['param_alpha']),
        -results['mean_test_score'],
        alpha=0.5, color='red', label='Validation MSE (CV)',
    )
    plt.scatter(
        np.log10(best_alpha), -best_score,
        color='green', s=100, label=f'Best Model (alpha={best_alpha:.2e})',
    )
    plt.xlabel('log10(alpha)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(f'Fold {fold}: Train vs Validation Average MSE vs. Alpha (CV)')
    plt.grid(True)
    plt.legend()


def _plot_shrinkage_paths(plot_df, features, colors, best_alpha_log10, title):
    """Plot coefficient shrinkage paths against ``log10(alpha)``.

    The figure is left open: the caller saves and closes it.

    Args:
        plot_df (pd.DataFrame): Coefficients per alpha, with a
            ``log10_alpha`` column and one column per feature.
        features (list[str]): Columns of ``plot_df`` to draw, in order.
        colors (list): Colors, cycled if there are more features.
        best_alpha_log10 (float): ``log10`` of the best alpha, drawn as a
            vertical dashed line.
        title (str): Title of the figure.
    """
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(features):
        plt.plot(
            plot_df['log10_alpha'], plot_df[feature],
            label=feature, linewidth=2, color=colors[i % len(colors)],
        )
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.axvline(
        best_alpha_log10, color='green', linestyle='--', label='Best model'
    )
    plt.xlabel('log10(alpha)')
    plt.ylabel('Coefficient Value')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Features')
    plt.grid(True)
    plt.tight_layout()


def _summarize_ranges(feature_ranges):
    """Get the minimum and maximum of the values of each feature.

    Args:
        feature_ranges (dict): Maps each feature to a list of values.

    Returns:
        tuple: ``(sorted_features, mins, maxs)`` where ``sorted_features``
            is the sorted list of keys and ``mins`` / ``maxs`` map each
            feature to its minimum / maximum.
    """
    features = sorted(feature_ranges.keys())
    mins = {f: min(feature_ranges[f]) for f in features}
    maxs = {f: max(feature_ranges[f]) for f in features}
    return features, mins, maxs


def _plot_range_bars(features, mins, maxs, color, label, xlabel, title):
    """Draw horizontal bars spanning the min-max range of each feature.

    The figure is left open: the caller saves and closes it.

    Args:
        features (list[str]): Features to draw (one bar each).
        mins (dict): Minimum value per feature.
        maxs (dict): Maximum value per feature.
        color (str): Bar color.
        label (str): Legend label of the bars.
        xlabel (str): Label of the x axis.
        title (str): Title of the figure.
    """
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(features))
    plt.barh(
        y_pos,
        [maxs[f] - mins[f] for f in features],
        left=[mins[f] for f in features],
        color=color, alpha=0.6, label=label,
    )
    plt.xlabel(xlabel)
    plt.ylabel('Feature')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yticks(y_pos, features)
    plt.legend(loc='upper right')
    plt.tight_layout()


def _plot_coef_ranges_with_ranks(
    features, coef_min, coef_max, coef_ranges, rank_ranges,
    appearance_count, add_note=False,
):
    """Plot coefficient ranges with per-fold crosses and rank labels.

    The figure is left open: the caller saves and closes it.

    Args:
        features (list[str]): Features to draw, from bottom to top.
        coef_min (dict): Minimum coefficient per feature.
        coef_max (dict): Maximum coefficient per feature.
        coef_ranges (dict): Maps each feature to its per-fold coefficients.
        rank_ranges (dict): Maps each feature to its per-fold shrink ranks.
        appearance_count (dict): Number of folds in which each feature has
            a non-zero coefficient.
        add_note (bool, optional): Add an explanatory note in the top-left
            corner. Defaults to False.
    """
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(features))
    ytick_labels = [f"{f} (n={appearance_count[f]})" for f in features]

    # Plot coefficient ranges (bars)
    width = [
        coef_max[f] - coef_min[f] if (coef_max[f] - coef_min[f]) != 0
        else 0.001
        for f in features
    ]
    bars = plt.barh(
        y_pos,
        width,
        left=[coef_min[f] for f in features],
        color='skyblue', alpha=0.6, label='Coefficient Range',
    )

    # Add crosses and rank text for each coefficient in each fold
    for i, feature in enumerate(features):
        for coef, rank in zip(coef_ranges[feature], rank_ranges[feature]):
            plt.scatter(
                coef, i,
                marker='x', color='red', s=50, alpha=0.8, linewidths=1,
            )
            # Add rank as text next to the cross
            plt.text(
                coef, i,
                str(int(rank)),
                ha='center', va='center',
                fontsize=8,
                color='black',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none',
                          pad=1),
            )

    if add_note:
        plt.text(
            0.02, 0.98,
            'Red crosses: Fold-specific coefficients. Associated number: '
            'Importance rank according to Shrinking alpha',
            transform=plt.gca().transAxes,
            ha='left', va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
        )

    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature (n = Appearance Count)')
    plt.title(
        'Coefficient ranges across Folds '
        '(Ordered by Mean coefficient magnitude)'
    )
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yticks(y_pos, ytick_labels)
    plt.legend(handles=[bars], loc='upper right')
    plt.tight_layout()


def get_graphs(
    fold_data_fn, feature_names, path_options_output_model, group,
    train_dict=None, test_dict=None, n_folds=4,
):
    """Draw all the figures from the saved ElasticNet outputs.

    Reads the pickled fold outputs and random searches of ``group``, then
    saves per-fold figures (CV MSE vs alpha,
    shrinkage paths) and summary figures (coefficient and rank ranges).

    Args:
        fold_data_fn (callable): ``fold_data_fn(train_ids, test_ids)``
            returning ``(X_train, y_train, X_test, y_test)`` preprocessed
            with the training set only (see ``_get_fold_data``).
        feature_names (list[str]): Names of the model features (columns
            of ``X_train``).
        path_options_output_model (pathlib.Path): Model output folder; the
            files are read from and saved in its ``group`` subfolder.
        group (str): Cohort name (e.g. ``"full"``, ``"intervention"``).
        train_dict (dict, optional): Fold index -> training IDs.
            Defaults to None.
        test_dict (dict, optional): Fold index -> test IDs.
            Defaults to None.
        n_folds (int, optional): Number of outer folds. Defaults to 4.

    Returns:
        None
    """
    custom_colors = _get_custom_colors()
    output_dir = path_options_output_model / group

    # --- Open outputs ---
    with open(
        output_dir / f'elasticnet_all_folds_outputs_{group}.pkl', 'rb'
    ) as f:
        fold_outputs = pickle.load(f)

    # --- Initialize outputs ---
    feature_coef_ranges = {}
    feature_rank_ranges = {}

    # --- Loop over folds ---
    for fold in range(n_folds):
        print(f"\n--- Processing Fold {fold} ---")

        with open(output_dir / f'random_search_fold{fold}_{group}.pkl',
                  'rb') as f:
            random_search = pickle.load(f)
        results = pd.DataFrame(random_search.cv_results_)

        best_model = fold_outputs[fold]['best_model']
        coef_table_all = fold_outputs[fold]['coef_table_all']
        shrink_alpha_df = fold_outputs[fold]['shrink_alpha']

        best_alpha = best_model.alpha

        # --- Plot 1: Train and Validation MSE vs. Alpha (from CV) ---
        _plot_cv_mse_vs_alpha(
            results, best_alpha, random_search.best_score_, fold
        )
        _save_figure(
            output_dir, f'fold_{fold}_cv_mse_vs_alpha_{group}',
            ('png', 'svg'),
        )
        plt.close()

        # Load data for this fold
        X_train, y_train, X_test, y_test = fold_data_fn(
            train_dict[fold], test_dict[fold]
        )

        # --- Plot 2: Coefficient Shrinkage Paths ---
        coef_df, shrink_alpha = _compute_shrink_paths(
            X_train, y_train, feature_names, best_model.l1_ratio
        )

        all_features_ordered = sorted(
            feature_names, key=lambda x: shrink_alpha[x], reverse=True
        )
        top_n_features = all_features_ordered[:N_FEATURES_TO_PLOT]
        plot_df = coef_df[top_n_features + ['log10_alpha']].copy()

        plot_df.columns = [_clean_feature_name(x) for x in plot_df.columns]
        top_n_features = [_clean_feature_name(x) for x in top_n_features]

        best_alpha_log10 = np.log10(best_alpha)
        _plot_shrinkage_paths(
            plot_df, top_n_features, custom_colors, best_alpha_log10,
            f'Coefficient shrinkage paths for fold {fold}.',
        )
        _save_figure(
            output_dir, f'fold_{fold}_alpha_vs_coefficients_25top_{group}',
            ('png', 'svg'),
        )
        plt.close()

        # --- Plot 2b: Coefficient Shrinkage Paths (Only Best Model
        # Features, Ordered by Shrink Alpha) ---
        # Get features with non-zero coefficients in the best model
        best_model_features = [
            f for f, coef in zip(feature_names, best_model.coef_)
            if coef != 0
        ]

        # Order by shrink_alpha (highest first = shrinks slowest)
        best_model_features_ordered = sorted(
            best_model_features, key=lambda x: shrink_alpha[x], reverse=True
        )

        best_model_plot_df = coef_df[
            best_model_features_ordered + ['log10_alpha']
        ].copy()
        best_model_plot_df.columns = [
            _clean_feature_name(x) for x in best_model_plot_df.columns
        ]

        best_model_features_ordered = [
            _clean_feature_name(x) for x in best_model_features_ordered
        ]

        _plot_shrinkage_paths(
            best_model_plot_df, best_model_features_ordered, custom_colors,
            best_alpha_log10,
            'Coefficient shrinkage paths for the features selected in the '
            f'best model for fold {fold}',
        )
        _save_figure(
            output_dir,
            f'fold_{fold}_alpha_vs_coefficients_best_model_{group}',
            ('png', 'svg'),
        )
        plt.close()

        # --- Table: Features + Coefficients for the best model ---
        coef_table = coef_table_all[
            coef_table_all[f'Coefficient_fold_{fold}'] != 0
        ].sort_values(
            by=f'Coefficient_fold_{fold}', ascending=False, key=abs
        )
        coef_table.reset_index(inplace=True, drop=True)

        # --- Update feature_coef_ranges and feature_rank_ranges ---
        coef_table['Feature'] = [
            _clean_feature_name(x) for x in coef_table['Feature']
        ]

        shrink_alpha_df['Feature'] = [
            _clean_feature_name(x) for x in shrink_alpha_df['Feature']
        ]

        for _, row in coef_table.iterrows():
            feature = row['Feature']
            if feature not in feature_coef_ranges:
                feature_coef_ranges[feature] = []
                feature_rank_ranges[feature] = []
            feature_coef_ranges[feature].append(
                row[f'Coefficient_fold_{fold}']
            )
            # Rank is 1-based index in the sorted table
            feature_rank_ranges[feature].append(
                shrink_alpha_df[
                    shrink_alpha_df['Feature'] == feature
                ][f'shrink_rank_fold_{fold}'].iloc[0]
            )

    # --- Generate summary graph: coefficient ranges and rank ranges ---
    features_with_non_zero_coef = sorted(feature_coef_ranges.keys())
    _, coef_min, coef_max = _summarize_ranges(feature_coef_ranges)
    _, rank_min, rank_max = _summarize_ranges(feature_rank_ranges)

    # --- Figure 1: Coefficient Ranges ---
    _plot_range_bars(
        features_with_non_zero_coef, coef_min, coef_max,
        color='skyblue', label='Coefficient Range',
        xlabel='Coefficient Value', title='Coefficient Ranges Across Folds',
    )
    _save_figure(
        output_dir, f'summary_coef_range_bars_{group}', ('png', 'svg')
    )
    plt.close()

    # --- Figure 2: Rank Ranges ---
    _plot_range_bars(
        features_with_non_zero_coef, rank_min, rank_max,
        color='salmon', label='Rank Range',
        xlabel='Rank', title='Rank Ranges Across Folds',
    )
    _save_figure(output_dir, f'summary_rank_ranges_{group}', ('png', 'svg'))
    plt.close()

    # --- Calculate mean absolute coefficients and appearance counts ---
    feature_coef_means = {}
    feature_appearance_count = {}

    for fold in fold_outputs:
        fold_outputs[fold]['coef_table']['Feature'] = [
            _clean_feature_name(x)
            for x in fold_outputs[fold]['coef_table']['Feature']
        ]
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
        reverse=True,
    )

    # --- Figure 1: Coefficient Ranges with Crosses and Rank Text ---
    _plot_coef_ranges_with_ranks(
        features_with_non_zero_coef, coef_min, coef_max,
        feature_coef_ranges, feature_rank_ranges, feature_appearance_count,
    )
    _save_figure(output_dir, f'summary_coef_ranges_{group}', ('png', 'svg'))
    plt.close()

    # --- Figure 1: Coefficient Ranges with Crosses and Rank Text ---
    # ! at least in 2 models
    features_with_non_zero_coef_above_2 = [
        feat for feat in features_with_non_zero_coef
        if feature_appearance_count[feat] >= 2
    ]

    _plot_coef_ranges_with_ranks(
        features_with_non_zero_coef_above_2, coef_min, coef_max,
        feature_coef_ranges, feature_rank_ranges, feature_appearance_count,
    )
    _save_figure(
        output_dir, f'summary_coef_ranges_{group}_atleast2models',
        ('png', 'svg'),
    )
    plt.close()


def apply_elasticnet(
    fold_data_fn, feature_names, path_options_output_model, group,
    train_dict=None, test_dict=None, n_folds=4, rerun_model=True,
    rerun_opti=True,
):
    """Run ElasticNet with cross-validation and save the outputs.

    For each fold, a randomized search is run (or reloaded), the best
    model with more than ``MIN_NON_ZERO_FEATURES`` non-zero coefficients
    is selected, and figures, coefficient tables and metrics are saved.
    Summary tables across folds are then built.

    Args:
        fold_data_fn (callable): ``fold_data_fn(train_ids, test_ids)``
            returning ``(X_train, y_train, X_test, y_test)`` preprocessed
            with the training set only (see ``_get_fold_data``).
        feature_names (list[str]): Names of the model features (columns
            of ``X_train``).
        path_options_output_model (pathlib.Path): Model output folder;
            files are saved in its ``group`` subfolder.
        group (str): Cohort name (e.g. ``"full"``, ``"intervention"``).
        train_dict (dict, optional): Fold index -> training IDs.
            Defaults to None.
        test_dict (dict, optional): Fold index -> test IDs.
            Defaults to None.
        n_folds (int, optional): Number of outer folds. Defaults to 4.
        rerun_model (bool, optional): Recompute and save the fold outputs
            (otherwise they are reloaded). Defaults to True.
        rerun_opti (bool, optional): Rerun the hyperparameter search
            (otherwise it is reloaded; only used if ``rerun_model``).
            Defaults to True.

    Returns:
        tuple: ``(fold_outputs, metrics_df)`` where ``fold_outputs`` maps
            each fold to its model, coefficient tables, shrinkage ranks
            and metrics, and ``metrics_df`` is the per-fold metrics table.
    """
    custom_colors = _get_custom_colors()
    output_dir = path_options_output_model / group

    # --- Initialize outputs ---
    fold_outputs = {}
    metrics_all_folds = []
    feature_coef_ranges = {}
    feature_rank_ranges = {}

    # --- Loop over folds ---
    for fold in range(n_folds):
        print(f"\n--- Processing Fold {fold} ---")

        # Load data for this fold
        X_train, y_train, X_test, y_test = fold_data_fn(
            train_dict[fold], test_dict[fold]
        )

        # --- ElasticNet with RandomizedSearchCV ---
        if rerun_model and rerun_opti:
            model = ElasticNet(
                max_iter=10000, random_state=42, tol=1e-4,
                selection='random',
            )
            param_distributions = {
                'alpha': loguniform(1e-4, 1e2),
                'l1_ratio': uniform(0.1, 0.9),
            }
            kf_inner = KFold(n_splits=3, shuffle=True, random_state=42)
            random_search = RandomizedSearchCV(
                model, param_distributions, n_iter=30, cv=kf_inner,
                scoring='neg_mean_squared_error', n_jobs=-1,
                random_state=42, verbose=1,
                return_train_score=True,  # Enable to get train scores
            )
            random_search.fit(X_train, y_train)

            with open(output_dir / f'random_search_fold{fold}_{group}.pkl',
                      'wb') as f:
                pickle.dump(random_search, f)
        else:
            with open(output_dir / f'random_search_fold{fold}_{group}.pkl',
                      'rb') as f:
                random_search = pickle.load(f)

        # --- Filter candidates to keep only those with > 5 non-zero
        # features ---
        results = pd.DataFrame(random_search.cv_results_)
        # Sort by best score first
        sorted_indices = results['mean_test_score'].argsort()[::-1]
        best_model = None
        best_alpha = None
        best_l1_ratio = None

        for idx in sorted_indices:
            params = results['params'].iloc[idx]
            temp_model = ElasticNet(
                **params, max_iter=10000, random_state=42, tol=1e-4,
                selection='random',
            )
            temp_model.fit(X_train, y_train)
            non_zero_count = np.sum(temp_model.coef_ != 0)
            if non_zero_count <= MIN_NON_ZERO_FEATURES:
                print(
                    f"Selected model with {non_zero_count} non-zero "
                    "features. Search for another one"
                )

            if non_zero_count > MIN_NON_ZERO_FEATURES:
                best_model = temp_model
                best_alpha = temp_model.alpha
                best_l1_ratio = temp_model.l1_ratio
                print(
                    f"Selected model with {non_zero_count} non-zero "
                    f"features (alpha={best_alpha:.2e}, "
                    f"l1_ratio={best_l1_ratio:.2f})"
                )
                break

        if best_model is None:
            print(
                "No model with > 5 non-zero features found. "
                "Using best model from RandomizedSearchCV."
            )
            best_model = random_search.best_estimator_
            best_alpha = best_model.alpha
            best_l1_ratio = best_model.l1_ratio

        # --- Evaluate on train and validation sets ---
        train_pred = best_model.predict(X_train)
        val_pred = best_model.predict(X_test)
        train_mse = np.mean((y_train - train_pred) ** 2)
        test_mse = np.mean((y_test - val_pred) ** 2)

        # --- Save metrics for this fold ---
        metrics_all_folds.append({
            'Fold': fold,
            'Train_MSE': train_mse,
            'Test_MSE': test_mse,
            'Best_Alpha': best_alpha,
            'Best_L1_Ratio': best_l1_ratio,
            'Non_Zero_Features': np.sum(best_model.coef_ != 0),
        })

        # --- Plot 1: Train and Validation MSE vs. Alpha (from CV) ---
        _plot_cv_mse_vs_alpha(
            results, best_alpha, random_search.best_score_, fold
        )
        _save_figure(output_dir, f'fold_{fold}_cv_mse_vs_alpha_{group}')
        plt.close()

        # --- Plot 2: Coefficient Shrinkage Paths ---
        coef_df, shrink_alpha = _compute_shrink_paths(
            X_train, y_train, feature_names, best_model.l1_ratio
        )

        shrink_alpha_df = pd.DataFrame.from_dict(
            shrink_alpha, orient="index"
        )
        shrink_alpha_df.columns = [f'shrink_alpha_fold_{fold}']
        shrink_alpha_df = shrink_alpha_df.sort_values(
            by=f'shrink_alpha_fold_{fold}', ascending=False
        )
        shrink_alpha_df['feature'] = shrink_alpha_df.index
        shrink_alpha_df.reset_index(inplace=True, drop=True)

        rank_shrink = shrink_alpha_df[
            f'shrink_alpha_fold_{fold}'
        ].drop_duplicates()
        rank_shrink = pd.DataFrame(rank_shrink)
        rank_shrink['shrink_rank'] = rank_shrink.index + 1

        shrink_alpha_df = pd.merge(
            shrink_alpha_df, rank_shrink, on=[f'shrink_alpha_fold_{fold}']
        )

        all_features_ordered = sorted(
            feature_names, key=lambda x: shrink_alpha[x], reverse=True
        )
        top_n_features = all_features_ordered[:N_FEATURES_TO_PLOT]
        plot_df = coef_df[top_n_features + ['log10_alpha']].copy()

        best_alpha_log10 = np.log10(best_alpha)
        _plot_shrinkage_paths(
            plot_df, top_n_features, custom_colors, best_alpha_log10,
            f'Fold {fold}: Coefficient shrinkage paths',
        )
        _save_figure(
            output_dir, f'fold_{fold}_alpha_vs_coefficients_25top_{group}'
        )
        plt.close()

        # --- Plot 2b: Coefficient Shrinkage Paths (Only Best Model
        # Features, Ordered by Shrink Alpha) ---
        # Get features with non-zero coefficients in the best model
        best_model_features = [
            f for f, coef in zip(feature_names, best_model.coef_)
            if coef != 0
        ]

        # Order by shrink_alpha (highest first = shrinks slowest)
        best_model_features_ordered = sorted(
            best_model_features, key=lambda x: shrink_alpha[x], reverse=True
        )

        if len(best_model_features_ordered) > 0:
            best_model_plot_df = coef_df[
                best_model_features_ordered + ['log10_alpha']
            ].copy()
            _plot_shrinkage_paths(
                best_model_plot_df, best_model_features_ordered,
                custom_colors, best_alpha_log10,
                'Coefficient shrinkage paths for the features selected in '
                f'the best model for fold {fold}',
            )
            _save_figure(
                output_dir,
                f'fold_{fold}_alpha_vs_coefficients_best_model_{group}',
            )
            plt.close()

        # --- Table: Features + Coefficients for the best model ---
        coefs = best_model.coef_
        coef_table_all = pd.DataFrame(
            {'Feature': feature_names, 'Coefficient': coefs}
        )
        coef_table_all = coef_table_all.sort_values(
            by='Coefficient', ascending=False, key=abs
        )
        coef_table_all.reset_index(inplace=True, drop=True)

        coef_table = coef_table_all[
            coef_table_all['Coefficient'] != 0
        ].sort_values(by='Coefficient', ascending=False, key=abs)
        coef_table.reset_index(inplace=True, drop=True)
        coef_table.to_csv(
            output_dir / f'fold_{fold}_best_model_coefficients_{group}.csv',
            index=False,
        )

        # --- Store outputs for this fold ---
        fold_outputs[fold] = {
            'best_model': best_model,
            'shrink_alpha': shrink_alpha_df.copy(),
            'coef_table': coef_table,
            'coef_table_all': coef_table_all,
            'metrics': {'Train_MSE': train_mse, 'Test_MSE': test_mse},
        }

        fold_outputs[fold]['shrink_alpha'].columns = [
            f'shrink_alpha_fold_{fold}', 'Feature',
            f'shrink_rank_fold_{fold}',
        ]
        fold_outputs[fold]['coef_table_all'].columns = [
            'Feature', f'Coefficient_fold_{fold}'
        ]

        # --- Update feature_coef_ranges and feature_rank_ranges ---
        for _, row in coef_table.iterrows():
            feature = row['Feature']
            if feature not in feature_coef_ranges:
                feature_coef_ranges[feature] = []
                feature_rank_ranges[feature] = []
            feature_coef_ranges[feature].append(row['Coefficient'])
            # Rank is 1-based index in the sorted table
            feature_rank_ranges[feature].append(
                shrink_alpha_df[
                    shrink_alpha_df['feature'] == feature
                ]['shrink_rank'].iloc[0]
            )

    # --- Save summary table of metrics for all folds ---
    metrics_df = pd.DataFrame(metrics_all_folds)
    metrics_df.to_csv(
        output_dir / f'summary_metrics_all_folds_{group}.csv', index=False
    )

    # --- Generate summary graph: coefficient ranges and rank ranges ---
    features_with_non_zero_coef = sorted(feature_coef_ranges.keys())
    _, coef_min, coef_max = _summarize_ranges(feature_coef_ranges)
    _, rank_min, rank_max = _summarize_ranges(feature_rank_ranges)

    # --- Figure 1: Coefficient Ranges ---
    _plot_range_bars(
        features_with_non_zero_coef, coef_min, coef_max,
        color='skyblue', label='Coefficient Range',
        xlabel='Coefficient Value', title='Coefficient Ranges Across Folds',
    )
    _save_figure(output_dir, f'summary_coef_range_bars_{group}')
    plt.close()

    # --- Figure 2: Rank Ranges ---
    _plot_range_bars(
        features_with_non_zero_coef, rank_min, rank_max,
        color='salmon', label='Rank Range',
        xlabel='Rank', title='Rank Ranges Across Folds',
    )
    _save_figure(output_dir, f'summary_rank_ranges_{group}')
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

    for fold in fold_outputs:
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
        reverse=True,
    )

    # --- Figure 1: Coefficient Ranges with Crosses and Rank Text ---
    _plot_coef_ranges_with_ranks(
        features_with_non_zero_coef, coef_min, coef_max,
        feature_coef_ranges, feature_rank_ranges, feature_appearance_count,
        add_note=True,
    )
    _save_figure(output_dir, f'summary_coef_ranges_{group}')
    plt.close()

    all_shrink = fold_outputs[0]['shrink_alpha'].copy()
    all_shrink = all_shrink[
        ['Feature', 'shrink_alpha_fold_0', 'shrink_rank_fold_0']
    ]
    for f in range(1, n_folds):
        all_shrink = pd.merge(
            all_shrink, fold_outputs[f]['shrink_alpha'],
            on=['Feature'], how='right',
        )
    all_shrink.columns = ['Feature'] + [
        col for f in range(0, n_folds)
        for col in (f'shrink_alpha_fold_{f}', f'shrink_rank_fold_{f}')
    ]
    all_shrink = round(all_shrink, 4)
    shrink_range = all_shrink[
        [f"shrink_rank_fold_{f}" for f in range(n_folds)]
    ]

    all_shrink['min shrink rank'] = np.min(shrink_range, axis=1)
    all_shrink['median shrink rank'] = np.median(shrink_range, axis=1)
    all_shrink['max shrink rank'] = np.max(shrink_range, axis=1)
    all_shrink["min/median/max shrink rank"] = [
        f"{mn} / {md} / {mx}"
        for mn, md, mx in zip(
            all_shrink['min shrink rank'],
            all_shrink['median shrink rank'],
            all_shrink['max shrink rank'],
        )
    ]

    # --- Save all outputs ---
    if rerun_model:
        with open(output_dir / f'elasticnet_all_folds_outputs_{group}.pkl',
                  'wb') as f:
            pickle.dump(fold_outputs, f)
    else:
        with open(output_dir / f'elasticnet_all_folds_outputs_{group}.pkl',
                  'rb') as f:
            fold_outputs = pickle.load(f)

    all_coeff_fin = fold_outputs[0]['coef_table_all'].copy()
    for f in range(1, n_folds):
        all_coeff_fin = pd.merge(
            all_coeff_fin, fold_outputs[f]['coef_table_all'],
            on=['Feature'], how='right',
        )
    all_coeff_fin = round(all_coeff_fin, 4)

    all_coeff_fin_range = all_coeff_fin[
        [f"Coefficient_fold_{f}" for f in range(n_folds)]
    ]

    # Ignore coefficients equal to 0
    coeff_nonzero = all_coeff_fin_range.replace(0, np.nan)

    all_coeff_fin['min Coeff'] = coeff_nonzero.min(axis=1, skipna=True)
    all_coeff_fin['median Coeff'] = coeff_nonzero.median(axis=1, skipna=True)
    all_coeff_fin['max Coeff'] = coeff_nonzero.max(axis=1, skipna=True)

    all_coeff_fin["min/median/max Coeff"] = [
        f"{mn} / {md} / {mx}"
        for mn, md, mx in zip(
            all_coeff_fin['min Coeff'],
            all_coeff_fin['median Coeff'],
            all_coeff_fin['max Coeff'],
        )
    ]

    table_final = all_coeff_fin[
        ["Feature", 'min Coeff', 'median Coeff', 'max Coeff',
         "min/median/max Coeff"]
    ]
    all_shrink = all_shrink[
        ["Feature", 'min shrink rank', 'median shrink rank',
         'max shrink rank', "min/median/max shrink rank"]
    ]

    table_final = pd.merge(table_final, all_shrink, on='Feature', how='right')

    appar = pd.DataFrame.from_dict(
        fold_outputs[0]['appearance_count'], orient="index"
    )
    appar['Feature'] = appar.index
    appar.reset_index(inplace=True, drop=True)
    appar.columns = ['Appearance count', 'Feature']
    table_final = pd.merge(table_final, appar, on='Feature', how='right')

    table_final.sort_values(
        inplace=True, by=['Appearance count'], ascending=False
    )

    table_final.to_csv(output_dir / f'table_final_{group}.csv', index=False)

    print("All folds processed successfully. Summary tables and graphs saved.")
    return fold_outputs, metrics_df


def run_cohort(
    X, y, base_feature_names, windsor_columns, windsor_ratio,
    path_options_output_model, group, train_dict, test_dict, n_folds,
    rerun_model, rerun_opti, add_interactions=False,
):
    """Run (optionally) ElasticNet on one cohort, then draw its graphs.

    Winsorization and scaling are fitted on the training set of each fold.

    Args:
        X (pd.DataFrame): Raw features, with an ``ID`` column (and a
            ``group_binary`` column if ``add_interactions``).
        y (pd.DataFrame): Target, with an ``ID`` column.
        base_feature_names (list[str]): Features of ``X`` used as inputs.
        windsor_columns (list[str]): Columns to winsorize.
        windsor_ratio (float): Percentage winsorized at each tail.
        path_options_output_model (pathlib.Path): Model output folder.
        group (str): Cohort name (e.g. ``"full"``, ``"control"``).
        train_dict (dict): Fold index -> training IDs.
        test_dict (dict): Fold index -> test IDs.
        n_folds (int): Number of outer folds.
        rerun_model (bool): Run ``apply_elasticnet`` before plotting.
        rerun_opti (bool): Rerun the hyperparameter search (otherwise
            it is reloaded).
        add_interactions (bool, optional): Add ``<feature>_x_group``
            interaction terms. Defaults to False.

    Returns:
        None
    """
    feature_names = list(base_feature_names)
    if add_interactions:
        feature_names += [f"{f}_x_group" for f in base_feature_names]

    fold_data_fn = partial(
        _get_fold_data, X, y,
        base_feature_names=base_feature_names,
        windsor_columns=windsor_columns,
        windsor_ratio=windsor_ratio,
        add_interactions=add_interactions,
    )

    if rerun_model:
        apply_elasticnet(
            fold_data_fn=fold_data_fn,
            feature_names=feature_names,
            path_options_output_model=path_options_output_model,
            group=group,
            train_dict=train_dict,
            test_dict=test_dict,
            n_folds=n_folds,
            rerun_model=rerun_model,
            rerun_opti=rerun_opti,
        )

    get_graphs(
        fold_data_fn=fold_data_fn,
        feature_names=feature_names,
        path_options_output_model=path_options_output_model,
        group=group,
        train_dict=train_dict,
        test_dict=test_dict,
        n_folds=n_folds,
    )


if __name__ == "__main__":

    #########################
    # Arguments
    #########################
    centered = False
    windsor_ratio = 1

    rerun_split = False
    rerun_model = False
    rerun_opti = False

    cross_val = True
    n_folds = 4

    version_model = 'v1'
    cognition = 'ztotal_composite'

    # ! ################################################################
    # ! Load and clean the data ########################################
    # ! ################################################################

    (df, col_cofacteurs_base, cofactors_to_test,
     col_nulisa, col_nulisa_baseline) = get_data()
    df.reset_index(drop=True, inplace=True)

    name_folder = 'mitt_windsor_halmin'
    if centered:
        name_folder += '_centered'

    logger = setup_logger(
        name='ElasticNet',
        log_dir=LOG_PATH,
        subfolder_name='machine learning R version',
        log_file=f'elasticnet_{version_model}.log',
        level=logging.INFO,
        mode='w',
    )

    logger.info(
        f" Options: centered: {centered}, subset: mITT, windsor: True, "
        f"windsor_ratio: {windsor_ratio}, cofactors_choice: age + sex + "
        f"education + BMI + site, rerun_split: {rerun_split}, "
        f"rerun_model: {rerun_model}, cross_val: {cross_val}, "
        f"rerun_opti: {rerun_opti}, cognition: {cognition} "
    )

    path_options_output_data = LONG_PATH / 'ElasticNet' / name_folder / 'data'
    path_options_output_model = (
        LONG_PATH / 'ElasticNet' / name_folder / version_model
    )

    os.makedirs(path_options_output_data, exist_ok=True)
    os.makedirs(path_options_output_model, exist_ok=True)
    os.makedirs(path_options_output_model / 'data_backup', exist_ok=True)

    # ! ################################################################
    # ! Prepare the target variable (cognitive change)
    # ! ################################################################
    df[f"{cognition}_change"] = df[f"{cognition}_2y"] - df[f"{cognition}_0y"]

    columns_to_keep = list(
        set(
            ['ID', 'group', 'ztotal_composite_change',
             f"{cognition}_2y", f"{cognition}_1y", f"{cognition}_0y"]
            + col_nulisa_baseline + cofactors_to_test + col_cofacteurs_base
        )
    )
    df = df[columns_to_keep]
    df.dropna(inplace=True)

    # ! ################################################################
    # ! Prepare for cross-validation
    # ! ################################################################

    # Make sure at least one dataset version folder exists
    if not os.listdir(path_options_output_data):
        os.makedirs(path_options_output_data / "dataset_v0", exist_ok=True)

    train_dict = {}
    test_dict = {}

    if rerun_split or not os.path.isfile(
        path_options_output_data / f'train_list_fold_0_over_{n_folds - 1}.csv'
    ):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            train_ids = df.iloc[train_idx]['ID']
            test_ids = df.iloc[val_idx]['ID']
            train_ids.to_csv(
                path_options_output_data
                / f'train_list_fold_{fold}_over_{n_folds - 1}.csv',
                index=False,
            )
            test_ids.to_csv(
                path_options_output_data
                / f'test_list_fold_{fold}_over_{n_folds - 1}.csv',
                index=False,
            )
            train_dict[fold] = train_ids
            test_dict[fold] = test_ids
    else:
        for fold in range(n_folds):
            train_dict[fold] = pd.read_csv(
                path_options_output_data
                / f'train_list_fold_{fold}_over_{n_folds - 1}.csv'
            )['ID']
            test_dict[fold] = pd.read_csv(
                path_options_output_data
                / f'test_list_fold_{fold}_over_{n_folds - 1}.csv'
            )['ID']

    # ! ################################################################
    # ! Select features
    # ! ################################################################

    df.reset_index(inplace=True, drop=True)
    feature_names = col_nulisa_baseline + cofactors_to_test

    # Raw features: winsorization and scaling are done inside each fold,
    # on the training set only.
    X = df[feature_names + ['ID']]
    y = df[['ztotal_composite_change'] + ['ID']]

    cohort_kwargs = dict(
        windsor_columns=col_nulisa_baseline,
        windsor_ratio=windsor_ratio,
        path_options_output_model=path_options_output_model,
        train_dict=train_dict,
        test_dict=test_dict,
        n_folds=n_folds,
        rerun_model=rerun_model,
        rerun_opti=rerun_opti,
    )

    print("Full cohort *************")

    run_cohort(X, y, feature_names, group="full", **cohort_kwargs)

    print("Intervention cohort *************")

    df_intervention = df[df['group'] == "Intervention"]
    df_intervention = df_intervention.reset_index(drop=True)
    run_cohort(
        df_intervention[feature_names + ['ID']],
        df_intervention[['ztotal_composite_change'] + ['ID']],
        feature_names, group="intervention", **cohort_kwargs,
    )

    print("Control cohort *************")

    df_control = df[df['group'] == "Control"]
    df_control = df_control.reset_index(drop=True)
    run_cohort(
        df_control[feature_names + ['ID']],
        df_control[['ztotal_composite_change'] + ['ID']],
        feature_names, group="control", **cohort_kwargs,
    )

    print("Full cohort with interaction terms *************")

    X_interaction = df[feature_names + ['ID']].copy()
    X_interaction['group_binary'] = np.where(
        df['group'] == 'Intervention', 1, 0
    )
    run_cohort(
        X_interaction, y, feature_names, group="full_interaction",
        add_interactions=True, **cohort_kwargs,
    )