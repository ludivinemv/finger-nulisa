from scipy import stats
import pandas as pd

class StatsDataFrame(pd.DataFrame):
    """Produce a dataframe with Statistics
    """
    def __init__(self, *args, **kwargs):
        # Initialize an empty DataFrame with the specified columns
        columns = ["characteristics", "values", "metrics", "Total", "Control", "Intervention", "p-value","p-value test"]
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
        control_data = df[df['group'] == 'Control']
        intervention_data = df[df['group'] == 'Intervention']

        total_stats = calculate_stats(total_data, id_column, column_name)
        control_stats = calculate_stats(control_data, id_column, column_name)
        intervention_stats = calculate_stats(intervention_data, id_column, column_name)

        _, p = stats.shapiro(total_data[column_name].dropna())

        # Calculate p-value
        try:
            if p > 0.1: # normal
                _, p_value = stats.ttest_ind(control_data[column_name], intervention_data[column_name], nan_policy='omit')
                p_value_str = f"{p_value:.4f}"
                p_value_test = f"t-test (shapiro p-value: {p:.4f})"

            else: # "Not normal"
                _, p_value = stats.mannwhitneyu(control_data[column_name], intervention_data[column_name], alternative='two-sided', nan_policy='omit')
                p_value_str = f"{p_value:.4f}"
                p_value_test = f"Mann-Whitney U test (shapiro p-value: {p:.4f})"
        except Exception:
            p_value_str = "N/A"

        # Create rows for each metric
        metrics = ["n(%)", "mean (std)", "min", "25%", "median", "75%", "max"]
        for i, metric in enumerate(metrics):
            new_row = {
                "characteristics": characteristic if i == 0 else "",
                "values": "",
                "metrics": metric,
                "Total": total_stats[metric],
                "Control": control_stats[metric],
                "Intervention": intervention_stats[metric],
                "p-value": p_value_str if i == 0 else "",
                "p-value test": p_value_test
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
        control_data = df[df['group'] == 'Control']
        intervention_data = df[df['group'] == 'Intervention']
    
        total_counts = calculate_counts(total_data, column_name)
        control_counts = calculate_counts(control_data, column_name)
        intervention_counts = calculate_counts(intervention_data, column_name)


        # Create contingency table for chi-square test
        contingency = pd.crosstab(
            df[[id_column, column_name]].drop_duplicates()[column_name],
            df[df[column_name].notna()][['group', id_column]].drop_duplicates()['group']
        )
        try:
            _, p_value, _, _ = stats.chi2_contingency(contingency)
            p_value_str = f"{p_value:.4f}"
            p_value_test = f"Chi 2"

        except Exception:
            p_value_str = "N/A"

        # Add total row
        total_n = total_data[id_column].nunique()
        control_n = control_data[id_column].nunique()
        intervention_n = intervention_data[id_column].nunique()

        self.loc[len(self)] = {
            "characteristics": characteristic,
            "values": "Total",
            "metrics": "n (%)",
            "Total": f"{total_n} (100%)",
            "Control": f"{control_n} (100%)",
            "Intervention": f"{intervention_n} (100%)",
            "p-value": p_value_str,
            "p-value test": p_value_test
        }

        # Add rows for each unique value
        all_values = sorted(set(total_counts.index) | set(control_counts.index) | set(intervention_counts.index))
        for value in all_values:
            self.loc[len(self)] = {
                "characteristics": "",
                "values": value,
                "metrics": "n (%)",
                "Total": total_counts.get(value, "0 (0.0%)"),
                "Control": control_counts.get(value, "0 (0.0%)"),
                "Intervention": intervention_counts.get(value, "0 (0.0%)"),
                "p-value": "",
                "p-value test": "",
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
            df2 = df[df[visit_column] == year1][[id_column, column_name, "group"]].rename(columns={column_name: f"v{year1}"})
            df11 = df[df[visit_column] == year2][[id_column, column_name]].rename(columns={column_name: f"v{year2}"})
            delta_df = pd.merge(df2, df11, on=id_column)
            delta_df["delta"] = delta_df[f"v{year2}"] - delta_df[f"v{year1}"]

            # Use "delta" column for stats
            total_stats = calculate_stats(delta_df.rename(columns={"delta": column_name}))
            control_stats = calculate_stats(delta_df[delta_df["group"] == "Control"].rename(columns={"delta": column_name}))
            intervention_stats = calculate_stats(delta_df[delta_df["group"] == "Intervention"].rename(columns={"delta": column_name}))

            _, p = stats.shapiro(delta_df['delta'].dropna())

            # Compute p-value
            try:
                if p > 0.1: # normal
                    _, p_value = stats.ttest_ind(delta_df[delta_df["group"] == "Control"]['delta'], delta_df[delta_df["group"] == "Intervention"]['delta'], nan_policy='omit')
                    p_value_str = f"{p_value:.4f}"
                    p_value_test = f"t-test (shapiro p-value: {p:.4f})"

                else: # "Not normal"
                    _, p_value = stats.mannwhitneyu(delta_df[delta_df["group"] == "Control"]['delta'], delta_df[delta_df["group"] == "Intervention"]['delta'], alternative='two-sided', nan_policy='omit')
                    p_value_str = f"{p_value:.4f}"
                    p_value_test = f"Mann-Whitney U test (shapiro p-value: {p:.4f})"
                
            except Exception:
                p_value_str = "N/A"

            for i, metric in enumerate(metrics):
                row = {
                    "characteristics": "",
                    "values": f"change between {year1} and {year2}" if metric == "n(%)" else "",
                    "metrics": metric,
                    "Total": total_stats[metric],
                    "Control": control_stats[metric],
                    "Intervention": intervention_stats[metric],
                    "p-value": p_value_str if metric == "n(%)" else "",
                    "p-value test": p_value_test if metric == "n(%)" else "",
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
            control_stats = calculate_stats(visit_df[visit_df["group"] == "Control"])
            intervention_stats = calculate_stats(visit_df[visit_df["group"] == "Intervention"])

            _, p = stats.shapiro(visit_df[column_name].dropna())

            # Compute p-value
            try:
                if p > 0.1: # normal
                    _, p_value = stats.ttest_ind(visit_df[visit_df["group"] == "Control"][column_name], visit_df[visit_df["group"] == "Intervention"][column_name], nan_policy='omit')
                    p_value_str = f"{p_value:.4f}"
                    p_value_test = f"t-test (shapiro p-value: {p:.4f})"

                else: # "Not normal"
                    _, p_value = stats.mannwhitneyu(visit_df[visit_df["group"] == "Control"][column_name], visit_df[visit_df["group"] == "Intervention"][column_name], alternative='two-sided', nan_policy='omit')
                    p_value_str = f"{p_value:.4f}"
                    p_value_test = f"Mann-Whitney U test (shapiro p-value: {p:.4f})"
                
            except Exception:
                p_value_str = "N/A"

            for i, metric in enumerate(metrics):
                row = {
                    "characteristics": characteristic if metric == "n(%)" and visit_label == "Total" else "",
                    "values": visit_label if metric == "n(%)" else "",
                    "metrics": metric,
                    "Total": total_stats[metric],
                    "Control": control_stats[metric],
                    "Intervention": intervention_stats[metric],
                    "p-value": p_value_str if metric == "n(%)" else "",
                    "p-value test": p_value_test if metric == "n(%)" else "",

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
       


