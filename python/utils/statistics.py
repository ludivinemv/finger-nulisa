from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import shapiro, spearmanr, skew, kurtosis, ks_2samp

def compute_summary_stats(series:pd.Series)-> dict:
    """Get dictionary with statistics for the serie

    Args:
        series (pd.Series): Data

    """
    return {
        'mean': series.mean(),
        'std': series.std(),
        'median': series.median(),
        'skewness': skew(series, nan_policy='omit'),
        'kurtosis': kurtosis(series, nan_policy='omit')
    }

    
def calculate_anot_spearman(df_corr:pd.DataFrame):
    """Calculate Spearman correlation matrix with annotation (p-value as *)

    Args:
        df_corr (pd.DataFrame): Dataframe with values to use for correlation

    Returns:
        corr (pd.DataFrame): correlation matrix
        pvals (pd.DataFrame): associated pvals matrix
        annot (pd.DataFrame): associated annot matrix (correlation coefficient + pvalues as *)
    """
    corr, pvals = spearman_corr_with_p(df_corr)

    annot = corr.copy().astype(str)

    for i in corr.index:
        for j in corr.columns:
            p = pvals.loc[i, j]
            r = corr.loc[i, j]
            if not np.isnan(p):
                stars = significance_stars(p)
                annot.loc[i, j] = f"{r:.2f}\n{stars}"
            else:
                annot.loc[i, j] = ""

    return corr, pvals, annot


def spearman_corr_with_p(df:pd.DataFrame):
    """Get Spearman correlation coefficient and p-value

    Args:
        df (pd.DataFrame): data
    """
    rho = df.corr(method='spearman')
    pvals = pd.DataFrame(np.ones((len(df.columns), len(df.columns))), columns=df.columns, index=df.columns)

    for row in df.columns:
        for col in df.columns:
            if row != col:
                r, p = spearmanr(df[row], df[col], nan_policy='omit')
                rho.loc[row, col] = r
                pvals.loc[row, col] = p
            else:
                pvals.loc[row, col] = np.nan  # NaN on diagonal

    return rho, pvals
    
    
def significance_stars(p:float):
    """Transform p-values to stars

    Args:
        p (float): _description_

    Returns:
        _type_: _description_
    """
    if p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.1:
        return '*'
    else:
        return ''
    

def get_shapiro_test(df:pd.DataFrame, list_variables:list, filename:str, path:Path):
    """Get Shapiro test for each column in list_variables

    Args:
        df (pd.DataFrame): data
        list_variables (list): list of variables
        filename (str): filename to save
        path (Path): path to save output

    """
    temp = []
    for col in list_variables:
        _, p = shapiro(df[col].dropna())
        if p > 0.05:
            hyp = "Normal"
        else:
            hyp = "Not normal"
        temp.append([col, p, hyp])

    temp = pd.DataFrame(temp, columns=['Column', 'p-value', 'Hypothesis'])

    temp.to_csv(path/f'{filename}_distribution.csv', index=False)

    return temp