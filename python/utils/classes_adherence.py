from scipy import stats
import pandas as pd

class StatsDataFrame(pd.DataFrame):
    """Produce a dataframe with Statistics
    """
    def __init__(self, *args, **kwargs):
        # Initialize an empty DataFrame with the specified columns
        columns = ["characteristics", "values", "metrics", "Total", "Control", "Low adh.", "High adh.", "p-value control-high","p-value test control-high", "p-value control-low","p-value test control-low", "p-value low-high","p-value test low-high"]
        super().__init__(columns=columns, *args, **kwargs)

    @property
    def _constructor(self):
        return StatsDataFrame
    
    def add_numerical_stats(self, df:pd.DataFrame, characteristic:str, column_name:str, id_column="ID"):
        """       
        Add numerical statistics for a characteristic.

        Args:
            df (pd.DataFrame): Categorical data for the total group
            characteristic (str): Name of the characteristic
            column_name (str): name of the characteristic column
            id_column (str, optional): ID column name. Defaults to "ID".

        Returns:
            _type_: _description_
        """

        # Calculate metrics for total, control, and intervention
        def calculate_stats(data, id_column, column_name):
            if len(data) == 0:
                return {
                    "n(%)": f"0 (0%)",
                    "mean (std)": "N/A",
                    "min": "N/A",
                    "25%": "N/A",
                    "median": "N/A",
                    "75%": "N/A",
                    "max": "N/A"
                }
            n = data[id_column].nunique()
            mean = data[column_name].mean()
            std = data[column_name].std()
            min_val = data[column_name].min()
            q25 = data[column_name].quantile(0.25)
            median = data[column_name].median()
            q75 = data[column_name].quantile(0.75)
            max_val = data[column_name].max()
            return {
                "n(%)": f"{n} (100%)",
                "mean (std)": f"{mean:.2f} ({std:.2f})",
                "min": f"{min_val:.2f}",
                "25%": f"{q25:.2f}",
                "median": f"{median:.2f}",
                "75%": f"{q75:.2f}",
                "max": f"{max_val:.2f}"
            }

        total_data = df
        control_data = df[df['adherence_class'] == 'control']
        low_data = df[df['adherence_class'] == 'low']
        high_data = df[df['adherence_class'] == 'high']

        total_stats = calculate_stats(total_data, id_column, column_name)
        control_stats = calculate_stats(control_data, id_column, column_name)
        low_stats = calculate_stats(low_data, id_column, column_name)
        high_stats = calculate_stats(high_data, id_column, column_name)

        _, p = stats.shapiro(total_data[column_name].dropna())

        # Calculate p-value control-high
        try:
            if p > 0.1: # normal
                _, p_value = stats.ttest_ind(control_data[column_name], high_data[column_name], nan_policy='omit')
                p_value_str = f"{p_value:.4f}"
                p_value_test = f"t-test (shapiro p-value: {p:.4f})"

            else: # "Not normal"
                _, p_value = stats.mannwhitneyu(control_data[column_name], high_data[column_name], alternative='two-sided', nan_policy='omit')
                p_value_str = f"{p_value:.4f}"
                p_value_test = f"Mann-Whitney U test (shapiro p-value: {p:.4f})"
        except Exception:
            p_value_str = "N/A"
        
        
        # Calculate p-value control-low
        try:
            if p > 0.1: # normal
                _, p_value_low = stats.ttest_ind(control_data[column_name], low_data[column_name], nan_policy='omit')
                p_value_str_low = f"{p_value_low:.4f}"
                p_value_test_low = f"t-test (shapiro p-value: {p:.4f})"

            else: # "Not normal"
                _, p_value_low = stats.mannwhitneyu(control_data[column_name], low_data[column_name], alternative='two-sided', nan_policy='omit')
                p_value_str_low = f"{p_value_low:.4f}"
                p_value_test_low = f"Mann-Whitney U test (shapiro p-value: {p:.4f})"
        except Exception:
            p_value_str_low = "N/A"
        
        # Calculate p-value low-high
        try:
            if p > 0.1: # normal
                _, p_value_high = stats.ttest_ind(low_data[column_name], high_data[column_name], nan_policy='omit')
                p_value_str_high = f"{p_value_high:.4f}"
                p_value_test_high = f"t-test (shapiro p-value: {p:.4f})"

            else: # "Not normal"
                _, p_value_high = stats.mannwhitneyu(low_data[column_name], high_data[column_name], alternative='two-sided', nan_policy='omit')
                p_value_str_high = f"{p_value_high:.4f}"
                p_value_test_high = f"Mann-Whitney U test (shapiro p-value: {p:.4f})"
        except Exception:
            p_value_str_high = "N/A"

        # Create rows for each metric
        metrics = ["n(%)", "mean (std)", "min", "25%", "median", "75%", "max"]
        for i, metric in enumerate(metrics):
            new_row = {
                "characteristics": characteristic if i == 0 else "",
                "values": "",
                "metrics": metric,
                "Total": total_stats[metric],
                "Control": control_stats[metric],
                "Low adh.": low_stats[metric],
                "High adh.": high_stats[metric],
                "p-value control-high": p_value_str if i == 0 else "",
                "p-value test control-high": p_value_test if i == 0 else "",
                "p-value control-low": p_value_str_low if i == 0 else "",
                "p-value test control-low": p_value_test_low if i == 0 else "",
                "p-value low-high": p_value_str_high if i == 0 else "",
                "p-value test low-high": p_value_test_high  if i == 0 else ""
            }
            self.loc[len(self)] = new_row

    def add_categorical_stats(self, df:pd.DataFrame, characteristic:str, column_name:str, id_column="ID"):
        """
        Add categorical statistics for a characteristic.

        
        Args:
            df (pd.DataFrame): Categorical data for the total group
            characteristic (str): Name of the characteristic
            column_name (str): name of the characteristic column
            id_column (str, optional): ID column name. Defaults to "ID".

        """
        # Calculate value counts for total, control, and intervention
        def calculate_counts(data, column_name):
            if data.empty:
                return pd.Series(dtype=str)
            grouped = data[[id_column, column_name]].drop_duplicates()
            counts = grouped[column_name].value_counts()
            total = data[id_column].nunique()
            return counts.apply(lambda x: f"{x} ({(x / total * 100):.1f}%)")

        total_data = df
        control_data = df[df['adherence_class'] == 'control']
        low_data = df[df['adherence_class'] == 'low']
        high_data = df[df['adherence_class'] == 'high']

    
        total_counts = calculate_counts(total_data, column_name)
        control_counts = calculate_counts(control_data, column_name)
        low_counts = calculate_counts(low_data, column_name)
        high_counts = calculate_counts(high_data, column_name)


        # Create contingency table for chi-square testdf_tmp = (
        df_tmp = (df.loc[df["adherence_class"].isin(["control", "high"]),
                [id_column, column_name, "adherence_class"]].drop_duplicates())
        contingency = pd.crosstab(df_tmp[column_name], df_tmp["adherence_class"])
        try:
            _, p_value, _, _ = stats.chi2_contingency(contingency)
            p_value_str = f"{p_value:.4f}"
            p_value_test = f"Chi 2"

        except Exception:
            p_value_str = "N/A"
            


        df_tmp = (df.loc[df["adherence_class"].isin(["control", "low"]),
                [id_column, column_name, "adherence_class"]].drop_duplicates())
        contingency = pd.crosstab(df_tmp[column_name], df_tmp["adherence_class"])
        try:
            _, p_value_low, _, _ = stats.chi2_contingency(contingency)
            p_value_str_low = f"{p_value_low:.4f}"
            p_value_test_low = f"Chi 2"

        except Exception:
            p_value_str_low = "N/A"


        df_tmp = (df.loc[df["adherence_class"].isin(["low", "high"]),
                [id_column, column_name, "adherence_class"]].drop_duplicates())
        contingency = pd.crosstab(df_tmp[column_name], df_tmp["adherence_class"])
        try:
            _, p_value_high, _, _ = stats.chi2_contingency(contingency)
            p_value_str_high = f"{p_value_high:.4f}"
            p_value_test_high = f"Chi 2"

        except Exception:
            p_value_str_high = "N/A"


        # Add total row
        total_n = total_data[id_column].nunique()
        control_n = control_data[id_column].nunique()
        low_n = low_data[id_column].nunique()
        high_n = high_data[id_column].nunique()

        self.loc[len(self)] = {
            "characteristics": characteristic,
            "values": "Total",
            "metrics": "n (%)",
            "Total": f"{total_n} (100%)",
            "Control": f"{control_n} (100%)",
            "Low adh.": f"{low_n} (100%)",
            "High adh.": f"{high_n} (100%)",
            "p-value control-high": p_value_str ,
            "p-value test control-high": p_value_test,
            "p-value control-low": p_value_str_low,
            "p-value test control-low": p_value_test_low,
            "p-value low-high": p_value_str_high ,
            "p-value test low-high": p_value_test_high
        }

        # Add rows for each unique value
        all_values = sorted(set(total_counts.index) | set(control_counts.index) | set(low_counts.index)| set(high_counts.index))
        for value in all_values:
            self.loc[len(self)] = {
                "characteristics": "",
                "values": value,
                "metrics": "n (%)",
                "Total": total_counts.get(value, "0 (0.0%)"),
                "Control": control_counts.get(value, "0 (0.0%)"),
                "Low adh.": low_counts.get(value, "0 (0.0%)"),
                "High adh.": high_counts.get(value, "0 (0.0%)"),
                "p-value control-high": "" ,
                "p-value test control-high": "",
                "p-value control-low": "",
                "p-value test control-low": "",
                "p-value low-high": "" ,
                "p-value test low-high": ""
            }


                

    def add_numerical_stats_per_year(self, 
                                     df: pd.DataFrame, 
                                     characteristic: str, 
                                     column_name: str,
                                     visit_column: str = "visit", 
                                     id_column: str = "ID", 
                                     change_0_2:bool = False,
                                     change_0_1:bool = False, 
                                     change_1_2:bool = False,
                                     change_0_11:bool = False, 
                                     change_2_11:bool = False):
        """
        Add numerical stats per visit value: Total, 0, 1, 2, 11.
        

        Args:
            df (pd.DataFrame): Data for the full population
            characteristic (str): Name of the characteristic
            column_name (str): Column of the characteristic
            visit_column (str, optional): Name of the visit ir year column. Defaults to "visit".
            id_column (str, optional): Name of the ID column. Defaults to "ID".
            change_0_2 (bool, optional): If True, add change between 0 and 2 years. Defaults to False.
            change_0_1 (bool, optional): If True, add change between 0 and 1 years. Defaults to False.
            change_1_2 (bool, optional): If True, add change between 1 and 2 years. Defaults to False.
            change_0_11 (bool, optional): If True, add change between 0 and 11 years. Defaults to False.
            change_2_11 (bool, optional): If True, add change between 2 and 11 years. Defaults to False.
        """

        visit_values = [("Total", None), ("0", 0), ("1", 1), ("2", 2), ("11", 11)]

        def calculate_stats(data):
            if data.empty:
                return {
                    "n(%)": "0 (0%)",
                    "mean (std)": "N/A",
                    "min": "N/A",
                    "25%": "N/A",
                    "median": "N/A",
                    "75%": "N/A",
                    "max": "N/A"
                }

            n = data[id_column].nunique()
            mean = data[column_name].mean()
            std = data[column_name].std()
            min_val = data[column_name].min()
            q25 = data[column_name].quantile(0.25)
            median = data[column_name].median()
            q75 = data[column_name].quantile(0.75)
            max_val = data[column_name].max()
            return {
                "n(%)": f"{n} (100%)",
                "mean (std)": f"{mean:.2f} ({std:.2f})",
                "min": f"{min_val:.2f}",
                "25%": f"{q25:.2f}",
                "median": f"{median:.2f}",
                "75%": f"{q75:.2f}",
                "max": f"{max_val:.2f}"
            }

        def get_change(self, df, column_name: str,metrics:list, year1:int = 2, year2:int = 11,visit_column: str = "visit", id_column: str = "ID"):
            df2 = df[df[visit_column] == year1][[id_column, column_name, "adherence_class"]].rename(columns={column_name: f"v{year1}"})
            df11 = df[df[visit_column] == year2][[id_column, column_name]].rename(columns={column_name: f"v{year2}"})
            delta_df = pd.merge(df2, df11, on=id_column)
            delta_df["delta"] = delta_df[f"v{year2}"] - delta_df[f"v{year1}"]

            # Use "delta" column for stats
            total_stats = calculate_stats(delta_df.rename(columns={"delta": column_name}))
            control_stats = calculate_stats(delta_df[delta_df["adherence_class"] == "control"].rename(columns={"delta": column_name}))
            low_stats = calculate_stats(delta_df[delta_df["adherence_class"] == "low"].rename(columns={"delta": column_name}))
            high_stats = calculate_stats(delta_df[delta_df["adherence_class"] == "high"].rename(columns={"delta": column_name}))

            _, p = stats.shapiro(delta_df['delta'].dropna())

            # Compute p-value
            try:
                if p > 0.1: # normal
                    _, p_value = stats.ttest_ind(delta_df[delta_df["adherence_class"] == "control"]['delta'], delta_df[delta_df["adherence_class"] == "high"]['delta'], nan_policy='omit')
                    p_value_str = f"{p_value:.4f}"
                    p_value_test = f"t-test (shapiro p-value: {p:.4f})"

                else: # "Not normal"
                    _, p_value = stats.mannwhitneyu(delta_df[delta_df["adherence_class"] == "control"]['delta'], delta_df[delta_df["adherence_class"] == "high"]['delta'], alternative='two-sided', nan_policy='omit')
                    p_value_str = f"{p_value:.4f}"
                    p_value_test = f"Mann-Whitney U test (shapiro p-value: {p:.4f})"
                
            except Exception:
                p_value_str = "N/A"

            try:
                if p > 0.1: # normal
                    _, p_value_low = stats.ttest_ind(delta_df[delta_df["adherence_class"] == "control"]['delta'], delta_df[delta_df["adherence_class"] == "low"]['delta'], nan_policy='omit')
                    p_value_str_low = f"{p_value_low:.4f}"
                    p_value_test_low = f"t-test (shapiro p-value: {p:.4f})"

                else: # "Not normal"
                    _, p_value_low = stats.mannwhitneyu(delta_df[delta_df["adherence_class"] == "control"]['delta'], delta_df[delta_df["adherence_class"] == "low"]['delta'], alternative='two-sided', nan_policy='omit')
                    p_value_str_low = f"{p_value_low:.4f}"
                    p_value_test_low = f"Mann-Whitney U test (shapiro p-value: {p:.4f})"
                
            except Exception:
                p_value_str_low = "N/A"


            try:
                if p > 0.1: # normal
                    _, p_value_high = stats.ttest_ind(delta_df[delta_df["adherence_class"] == "low"]['delta'], delta_df[delta_df["adherence_class"] == "high"]['delta'], nan_policy='omit')
                    p_value_str_high = f"{p_value_high:.4f}"
                    p_value_test_high = f"t-test (shapiro p-value: {p:.4f})"

                else: # "Not normal"
                    _, p_value_high = stats.mannwhitneyu(delta_df[delta_df["adherence_class"] == "low"]['delta'], delta_df[delta_df["adherence_class"] == "high"]['delta'], alternative='two-sided', nan_policy='omit')
                    p_value_str_high = f"{p_value_high:.4f}"
                    p_value_test_high = f"Mann-Whitney U test (shapiro p-value: {p:.4f})"
                
            except Exception:
                p_value_str_high = "N/A"


            for i, metric in enumerate(metrics):
                row = {
                    "characteristics": "",
                    "values": f"change between {year1} and {year2}" if metric == "n(%)" else "",
                    "metrics": metric,
                    "Total": total_stats[metric],
                    "Control": control_stats[metric],
                    "Low adh.": low_stats[metric],
                    "High adh.": high_stats[metric],
                    "p-value control-high": p_value_str if metric == "n(%)" else "",
                    "p-value test control-high": p_value_test if metric == "n(%)" else "",
                    "p-value control-low": p_value_str_low if metric == "n(%)" else "",
                    "p-value test control-low": p_value_test_low if metric == "n(%)" else "",
                    "p-value low-high": p_value_str_high if metric == "n(%)" else "",
                    "p-value test low-high": p_value_test_high if metric == "n(%)" else ""
                }
                self.loc[len(self)] =  row
                
        metrics = ["n(%)", "mean (std)", "min", "25%", "median", "75%", "max"]

        for visit_label, visit_value in visit_values:
            if visit_value is None:
                visit_df = df.copy()
            else:
                visit_df = df[df[visit_column] == visit_value].copy()

            # Filter out missing values
            visit_df = visit_df.dropna(subset=[column_name])

            total_stats = calculate_stats(visit_df)
            control_stats = calculate_stats(visit_df[visit_df["adherence_class"] == "control"])
            low_stats = calculate_stats(visit_df[visit_df["adherence_class"] == "low"])
            high_stats = calculate_stats(visit_df[visit_df["adherence_class"] == "high"])

            _, p = stats.shapiro(visit_df[column_name].dropna())

            # Compute p-value
            try:
                if p > 0.1: # normal
                    _, p_value = stats.ttest_ind(visit_df[visit_df["adherence_class"] == "control"][column_name], visit_df[visit_df["adherence_class"] == "high"][column_name], nan_policy='omit')
                    p_value_str = f"{p_value:.4f}"
                    p_value_test = f"t-test (shapiro p-value: {p:.4f})"

                else: # "Not normal"
                    _, p_value = stats.mannwhitneyu(visit_df[visit_df["adherence_class"] == "control"][column_name], visit_df[visit_df["adherence_class"] == "high"][column_name], alternative='two-sided', nan_policy='omit')
                    p_value_str = f"{p_value:.4f}"
                    p_value_test = f"Mann-Whitney U test (shapiro p-value: {p:.4f})"
                
            except Exception:
                p_value_str = "N/A"



                
            try:
                if p > 0.1: # normal
                    _, p_value_low = stats.ttest_ind(visit_df[visit_df["adherence_class"] == "control"]['delta'], visit_df[visit_df["adherence_class"] == "low"]['delta'], nan_policy='omit')
                    p_value_str_low = f"{p_value_low:.4f}"
                    p_value_test_low = f"t-test (shapiro p-value: {p:.4f})"

                else: # "Not normal"
                    _, p_value_low = stats.mannwhitneyu(visit_df[visit_df["adherence_class"] == "control"]['delta'], visit_df[visit_df["adherence_class"] == "low"]['delta'], alternative='two-sided', nan_policy='omit')
                    p_value_str_low = f"{p_value_low:.4f}"
                    p_value_test_low = f"Mann-Whitney U test (shapiro p-value: {p:.4f})"
                
            except Exception:
                p_value_str_low = "N/A"


            try:
                if p > 0.1: # normal
                    _, p_value_high = stats.ttest_ind(visit_df[visit_df["adherence_class"] == "low"]['delta'], visit_df[visit_df["adherence_class"] == "high"]['delta'], nan_policy='omit')
                    p_value_str_high = f"{p_value_high:.4f}"
                    p_value_test_high = f"t-test (shapiro p-value: {p:.4f})"

                else: # "Not normal"
                    _, p_value_high = stats.mannwhitneyu(visit_df[visit_df["adherence_class"] == "low"]['delta'], visit_df[visit_df["adherence_class"] == "high"]['delta'], alternative='two-sided', nan_policy='omit')
                    p_value_str_high = f"{p_value_high:.4f}"
                    p_value_test_high = f"Mann-Whitney U test (shapiro p-value: {p:.4f})"
                
            except Exception:
                p_value_str_high = "N/A"


            for i, metric in enumerate(metrics):
                row = {
                    "characteristics": characteristic if metric == "n(%)" and visit_label == "Total" else "",
                    "values": visit_label if metric == "n(%)" else "",
                    "metrics": metric,
                    "Total": total_stats[metric],
                    "Control": control_stats[metric],
                    "Low adh.": low_stats[metric],
                    "High adh.": high_stats[metric],
                    "p-value control-high": p_value_str if metric == "n(%)" else "",
                    "p-value test control-high": p_value_test if metric == "n(%)" else "",
                    "p-value control-low": p_value_str_low if metric == "n(%)" else "",
                    "p-value test control-low": p_value_test_low if metric == "n(%)" else "",
                    "p-value low-high": p_value_str_high if metric == "n(%)" else "",
                    "p-value test low-high": p_value_test_high if metric == "n(%)" else ""
               
                }
                self.loc[len(self)] = row

        if change_0_1:
            get_change(self, df, column_name, metrics, year1 = 0, year2 = 1,visit_column = visit_column, id_column = id_column)
        if change_0_2:
            get_change(self, df, column_name, metrics, year1 = 0, year2 = 2,visit_column = visit_column, id_column = id_column)
        if change_0_11:
            get_change(self, df, column_name, metrics, year1 = 0, year2 = 11,visit_column = visit_column, id_column = id_column)
        if change_1_2:
            get_change(self, df, column_name, metrics, year1 = 1, year2 = 2,visit_column = visit_column, id_column = id_column)
        if change_2_11:
            get_change(self, df, column_name, metrics, year1 = 2, year2 = 11,visit_column = visit_column, id_column = id_column)
       



