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
       





class ResumeDataFrame(pd.DataFrame):
    """Produce a dataframe with resume table
    """
    def __init__(self, list_outcomes, list_bbm, nulisa=True):
        if list_outcomes is None:
            raise ValueError("list_outcomes cannot be None")
        if list_bbm is None:
            raise ValueError("list_bbm cannot be None")

        columns_resume = [
            'diff. baseline BBM between groups',
            'diff. BBM evolution between groups'
        ]

        for outcome in list_outcomes:
            columns_resume += [
                f'baseline bbm infl. baseline {outcome} in full population group',
                f'baseline bbm infl. evol {outcome} in full population group',
                f'baseline bbm infl. baseline {outcome} in control group',
                f'baseline bbm infl. evol {outcome} in control group',
                f'baseline bbm infl. baseline {outcome} in intervention group',
                f'baseline bbm infl. evol {outcome} in intervention group',
            ]
            if outcome == 'ztotal_composite':
                columns_resume += [f'baseline bbm infl. {outcome} curve in full population group',
                                   f'baseline bbm infl. {outcome} curve in control group',
                                   f'baseline bbm infl. {outcome} curve in intervention group',
                                   ]

            columns_resume += [
                f'baseline bbm linked with diff. baseline {outcome} between groups',
                f'baseline bbm linked with diff. evol {outcome} between groups'
            ]
            if outcome == 'ztotal_composite':
                columns_resume += [f'baseline bbm linked with diff. {outcome} curve between groups']

            columns_resume += [
                f'bbm evol infl. {outcome} evol',
                f'bbm evol linked with diff. evol {outcome} between groups'
            ]

        columns_interest = ['Expected effect', 'Class', 'Description'] + columns_resume

        # Add 'Class Alamar' only if nulisa is True
        if nulisa:
            columns_interest.insert(1, 'Class Alamar')

        # Initialize empty DataFrame with desired structure
        data = pd.DataFrame(index=list_bbm, columns=columns_interest)

        # Pass initialized data to the parent constructor
        super().__init__(data)

        # Store additional attributes if needed
        self.list_outcomes = list_outcomes
        self.list_bbm = list_bbm
        self.nulisa = nulisa
        self._columns_interest = columns_resume  # Store the dynamic columns


    def get_columns_interest(self):
        return self._columns_interest
    
    def add_description(self, dict_desc:dict, column_name:str):
        for key, values in dict_desc.items():
            self.loc[key, column_name] = values

    def add_class(self, dict_desc:dict, column_name:str):
        for key, values in dict_desc.items():
            for value in values:
                self.loc[value, column_name] = key

    def add_bbm_by_intervention(self, df:pd.DataFrame, pval:float, pval_type:str = 'pval'):
        """
        Add numerical statistics for a characteristic.

        Parameters:
        - df: Dataframe with coefficients and pvalues from mixed-effect model with bbm as outcome and group as predictor
        - pval: name of the characteristic column
        """
        
        # 'diff. baseline BBM between groups'
        bbm_group_base = df[df[f'group {pval_type}']<pval]
        for index in bbm_group_base.index:
            if bbm_group_base.loc[index, 'group coeff' ] < 0:
                self.loc[index, 'diff. baseline BBM between groups'] = 'Lower in intervention'
            else:                     
                self.loc[index, 'diff. baseline BBM between groups'] = 'Higher in intervention'

        # diff. BBM evolution between groups
        bbm_group_evol = df[df[f'group:year {pval_type}']<pval]
        for index in bbm_group_evol.index:
            if bbm_group_evol.loc[index, 'year coeff' ] < 0:
                if bbm_group_evol.loc[index, 'year coeff' ] + bbm_group_evol.loc[index, 'group:year coeff' ] > 0:
                    self.loc[index, 'diff. BBM evolution between groups'] = 'Decrease in C / increase in I'
                else:
                    self.loc[index, 'diff. BBM evolution between groups'] = 'Decrease in C / decrease more in I'
            else:
                if bbm_group_evol.loc[index, 'year coeff' ] + bbm_group_evol.loc[index, 'group:year coeff' ] > 0:
                    self.loc[index, 'diff. BBM evolution between groups'] = 'Increase in C / increase more in I'
                else:
                    self.loc[index, 'diff. BBM evolution between groups'] = 'Increase in C / decrease in I'


    def add_cogntion_by_bbm_baseline(self, df:pd.DataFrame, outcome:str, pval:float, df_quadra:pd.DataFrame|None = None, pval_type:str = 'pval', subgroup:str = 'full population'):
        """
        Add numerical statistics for a characteristic.

        Parameters:
        - df: Dataframe with coefficients and pvalues from mixed-effect model with bbm as outcome and group as predictor
        - pval: name of the characteristic column
        """
        
        # baseline bbm infl. baseline outcome
        all_pat_bm = df[df[f'bm {pval_type}']<pval]
        for index in all_pat_bm.index:
            if all_pat_bm.loc[index, 'bm coeff' ] < 0:
                self.loc[index, f'baseline bbm infl. baseline {outcome} in {subgroup} group'] = f"Higher baseline BBM linked with lower baseline {outcome}"
            else:                     
                self.loc[index, f'baseline bbm infl. baseline {outcome} in {subgroup} group'] = f"Higher baseline BBM linked with higher baseline {outcome}"

        # baseline bbm infl. evol outcome
        all_pat_bmyear = df[df[f'bm:year {pval_type}']<pval]
        for index in all_pat_bmyear.index:
            if all_pat_bmyear.loc[index, 'bm:year coeff' ] < 0:                    
                self.loc[index, f'baseline bbm infl. evol {outcome} in {subgroup} group'] = f'Higher baseline BBM linked with lower increase of {outcome}'
            else:                     
                self.loc[index, f'baseline bbm infl. evol {outcome} in {subgroup} group'] = f'Higher baseline BBM linked with higher increase of {outcome}'

        # baseline bbm infl. ztotal_composite curve
        if isinstance(df_quadra, pd.DataFrame):
            df_quadra = df_quadra[(df_quadra[f'bm:year2 {pval_type}']<pval)|(df_quadra['bm:year pvalue']<pval)]
            all_pat_qadra_year2 = df_quadra[(df_quadra[f'bm:year2 {pval_type}']<pval)]

            for index in all_pat_qadra_year2.index:
                if all_pat_qadra_year2.loc[index, 'bm:year2 coeff' ] < 0:                    
                    self.loc[index, f'baseline bbm infl. {outcome} curve in {subgroup} group'] = "Higher baseline values of these BBM lower the deceleration over time of the increase of cognition"
                else:                     
                    self.loc[index, f'baseline bbm infl. {outcome} curve'] = "Higher baseline values of these BBM increase the deceleration over time of the increase of cognition"



    def add_cogntion_by_bbm_change(self, df:pd.DataFrame, outcome:str, pval:float, pval_type:str = 'pval'):
        """
        Add numerical statistics for a characteristic.

        Parameters:
        - df: Dataframe with coefficients and pvalues from mixed-effect model with bbm as outcome and group as predictor
        - pval: name of the characteristic column
        """

        # f'bbm evol infl. {outcome} evol'
        all_evol_bmyear = df[df[f'bm:year {pval_type}']<pval]
        for index in all_evol_bmyear.index:
            if all_evol_bmyear.loc[index, 'bm:year coeff' ] < 0:                    
                self.loc[index, f'bbm evol infl. {outcome} evol'] = f'Higher increase of BBM linked with lower increase of {outcome}'
            else:                     
                self.loc[index, f'bbm evol infl. {outcome} evol'] = f'Higher increase of BBM linked with higher increase of {outcome}'

    def add_cogntion_by_bbm_baseline_intervention(self, df:pd.DataFrame, outcome:str, pval:float, df_quadra:pd.DataFrame|None = None, pval_type:str = 'pval'):
        """
        Add numerical statistics for a characteristic.

        Parameters:
        - df: Dataframe with coefficients and pvalues from mixed-effect model with bbm as outcome and group as predictor
        - pval: name of the characteristic column
        """
    
        # baseline bbm linked with diff. baseline outcome between groups

        group_pat_control = df[df[f'bm {pval_type}']<pval]
        group_pat_inter = df[df[f'group:bm {pval_type}']<pval]
        for index in group_pat_inter.index:
            if (group_pat_inter.loc[index, 'group:bm coeff' ] > 0)  and (group_pat_inter.loc[index, f'bm {pval_type}' ] < 0.1) and (group_pat_inter.loc[index, 'bm coeff' ] < 0):
                self.loc[index, f'baseline bbm linked with diff. baseline {outcome} between groups'] = f"Higher baseline BBM linked with lower cognition at baseline in the control group but the effect is neutralised or inverted in the intervention group"
            elif (group_pat_inter.loc[index, 'group:bm coeff' ] < 0)  and (group_pat_inter.loc[index, f'bm {pval_type}' ] < 0.1) and (group_pat_inter.loc[index, 'bm coeff' ] > 0):
                self.loc[index, f'baseline bbm linked with diff. baseline {outcome} between groups'] = f"Higher baseline BBM linked with higher cognition at baseline in the control group but the effect is neutralised or inverted in the intervention group"
            elif (group_pat_inter.loc[index, 'group:bm coeff' ] > 0)  and (group_pat_inter.loc[index, f'bm {pval_type}' ] < 0.1) and (group_pat_inter.loc[index, 'bm coeff' ] > 0):
                self.loc[index, f'baseline bbm linked with diff. baseline {outcome} between groups'] = f"Higher baseline BBM linked with higher cognition at baseline in the control group but the effect is increased in the intervention group"
            elif (group_pat_inter.loc[index, 'group:bm coeff' ] < 0)  and (group_pat_inter.loc[index, f'bm {pval_type}' ] < 0.1) and (group_pat_inter.loc[index, 'bm coeff' ] < 0):
                self.loc[index, f'baseline bbm linked with diff. baseline {outcome} between groups'] = f"Higher baseline BBM linked with lower cognition at baseline in the control group but the effect is increased in the intervention group"
            elif (group_pat_inter.loc[index, 'group:bm coeff' ] > 0)  and (group_pat_inter.loc[index, f'bm {pval_type}' ] > 0.1) :
                self.loc[index, f'baseline bbm linked with diff. baseline {outcome} between groups'] = f"Baseline BBM are not linked with a difference of cognition at baseline in the control group but a higher baseline BBM value is linked with a higher cognition at baseline the intervention group"
            elif (group_pat_inter.loc[index, 'group:bm coeff' ] < 0)  and (group_pat_inter.loc[index, f'bm {pval_type}' ] > 0.1) :
                self.loc[index, f'baseline bbm linked with diff. baseline {outcome} between groups'] = f"Baseline BBM are not linked with a difference of cognition at baseline in the control group but a higher baseline BBM value is linked with a lower cognition at baseline the intervention group"
                            
                                
        # baseline bbm linked with diff. evol ztotal_composite between groups

        group_evol_inter = df[df[f'group:bm:year {pval_type}']<pval]
        for index in group_evol_inter.index:
            if (group_evol_inter.loc[index, 'group:bm:year coeff' ] > 0)  and (group_evol_inter.loc[index, f'bm:year {pval_type}' ] < 0.1) and (group_evol_inter.loc[index, 'bm:year coeff' ] < 0):
                self.loc[index, f'baseline bbm linked with diff. evol {outcome} between groups'] = f"Higher baseline BBM linked with lower increase (or even decrease) of the cognitive score over time in the control group while the intervention reduces or cancels the negative effect of the BBM"
            elif (group_evol_inter.loc[index, 'group:bm:year coeff' ] < 0)  and (group_evol_inter.loc[index, f'bm:year {pval_type}' ] < 0.1) and (group_evol_inter.loc[index, 'bm:year coeff' ] > 0):
                self.loc[index, f'baseline bbm linked with diff. evol {outcome} between groups'] = f"Higher baseline BBM linked with higher increase of the cognitive score over time in the control group while the intervention reduces or cancels the negative effect of the BBM"
            elif (group_evol_inter.loc[index, 'group:bm:year coeff' ] > 0)  and (group_evol_inter.loc[index, f'bm:year {pval_type}' ] < 0.1) and (group_evol_inter.loc[index, 'bm:year coeff' ] > 0):
                self.loc[index, f'baseline bbm linked with diff. evol {outcome} between groups'] = f"Higher baseline BBM linked with higher increase of the cognitive score over time in the control group while the intervention increase the positive effect of the BBM"
            elif (group_evol_inter.loc[index, 'group:bm:year coeff' ] < 0)  and (group_evol_inter.loc[index, f'bm:year {pval_type}' ] < 0.1) and (group_evol_inter.loc[index, 'bm:year coeff' ] < 0):
                self.loc[index, f'baseline bbm linked with diff. evol {outcome} between groups'] = f"Higher baseline BBM linked with lower increase (or even decrease) of the cognitive score over time in the control group while the intervention increase the negative effect of the BBM"
            elif (group_evol_inter.loc[index, 'group:bm:year coeff' ] > 0)  and (group_evol_inter.loc[index, f'bm:year {pval_type}' ] > 0.1) :
                self.loc[index, f'baseline bbm linked with diff. evol {outcome} between groups'] = f"Baseline BBM does not influence the evolution of the cognitive score in the control group but higher baseline BBM value in the intervention group is linked with faster cognitive improvement"
            elif (group_evol_inter.loc[index, 'group:bm:year coeff' ] < 0)  and (group_evol_inter.loc[index, f'bm:year {pval_type}' ] > 0.1) :
                self.loc[index, f'baseline bbm linked with diff. evol {outcome} between groups'] = f"Baseline BBM does not influence the evolution of the cognitive score in the control group but higher baseline BBM value in the intervention group is linked with slower cognitive improvement"
                    
        # baseline bbm linked with diff. ztotal_composite curve
        if isinstance(df_quadra, pd.DataFrame):
            df_quadra = df_quadra[(df_quadra[f'group:bm:year2 {pval_type}']<pval)|(df_quadra[f'group:bm:year {pval_type}']<pval)]
            group_pat_quadra_year2 = df_quadra[df_quadra[f'group:bm:year2 {pval_type}']<pval]

            for index in group_pat_quadra_year2.index:
                if (group_pat_quadra_year2.loc[index, 'group:bm:year2 coeff' ] > 0)  and (group_pat_quadra_year2.loc[index, f'bm:year2 {pval_type}' ] < 0.1) and (group_pat_quadra_year2.loc[index, 'bm:year2 coeff' ] < 0):
                    self.loc[index, f'baseline bbm linked with diff. {outcome} curve between groups'] = f"Higher baseline BBM increase the deceleration over time of cognition in the control group, but the intervention neutralise or invert the effect of the BBM"
                elif (group_pat_quadra_year2.loc[index, 'group:bm:year2 coeff' ] < 0)  and (group_pat_quadra_year2.loc[index, f'bm:year2 {pval_type}' ] < 0.1) and (group_pat_quadra_year2.loc[index, 'bm:year2 coeff' ] > 0):
                    self.loc[index, f'baseline bbm linked with diff. {outcome} curve between groups'] = f"Higher baseline BBM lower the deceleration over time of cognition in the control group, but the intervention neutralise or invert the effect of the BBM"
                elif (group_pat_quadra_year2.loc[index, 'group:bm:year2 coeff' ] > 0)  and (group_pat_quadra_year2.loc[index, f'bm:year2 {pval_type}' ] < 0.1) and (group_pat_quadra_year2.loc[index, 'bm:year2 coeff' ] > 0):
                    self.loc[index, f'baseline bbm linked with diff. {outcome} curve between groups'] = f"Higher baseline BBM lower the deceleration over time of cognition in the control group, and the intervention accentuate the effect of the BBM"
                elif (group_pat_quadra_year2.loc[index, 'group:bm:year2 coeff' ] < 0)  and (group_pat_quadra_year2.loc[index, f'bm:year2 {pval_type}' ] < 0.1) and (group_pat_quadra_year2.loc[index, 'bm:year2 coeff' ] < 0):
                    self.loc[index, f'baseline bbm linked with diff. {outcome} curve between groups'] = f"Higher baseline BBM increase the deceleration over time of cognition in the control group, and the intervention accentuate the effect of the BBM"
                elif (group_pat_quadra_year2.loc[index, 'group:bm:year2 coeff' ] > 0)  and (group_pat_quadra_year2.loc[index, f'bm:year2 {pval_type}' ] > 0.1) :
                    self.loc[index, f'baseline bbm linked with diff. {outcome} curve between groups'] = f"BBM baseline don't have an influence on the evolution of the speed over time in the control group but have one in the intervention group, lowering the deceleration"
                elif (group_pat_quadra_year2.loc[index, 'group:bm:year2 coeff' ] < 0)  and (group_pat_quadra_year2.loc[index, f'bm:year2 {pval_type}' ] > 0.1) :
                    self.loc[index, f'baseline bbm linked with diff. {outcome} curve between groups'] = f"BBM baseline don't have an influence on the evolution of the speed over time in the control group but have one in the intervention group, increasing the deceleration"
            

    def add_cogntion_by_bbm_change_intervention(self, df:pd.DataFrame, outcome:str, pval:float, pval_type:str = 'pval'):
        """
        Add numerical statistics for a characteristic.

        Parameters:
        - df: Dataframe with coefficients and pvalues from mixed-effect model with bbm as outcome and group as predictor
        - pval: name of the characteristic column
        """

        # f'bbm evol linked with diff. evol {outcome} between groups'
        group_evol_evol_inter = df[df[f'group:bm:year {pval_type}']<pval]
        for index in group_evol_evol_inter.index:
            if (group_evol_evol_inter.loc[index, 'group:bm:year coeff' ] > 0)  and (group_evol_evol_inter.loc[index, f'bm:year {pval_type}' ] < 0.1) and (group_evol_evol_inter.loc[index, 'bm:year coeff' ] < 0):
                self.loc[index, f'bbm evol linked with diff. evol {outcome} between groups'] = f"Higher increase of BBM linked with lower increase (or even decrease) of the cognitive score over time in the control group while the intervention reduces or cancels the negative effect of the BBM"
            elif (group_evol_evol_inter.loc[index, 'group:bm:year coeff' ] < 0)  and (group_evol_evol_inter.loc[index, f'bm:year {pval_type}' ] < 0.1) and (group_evol_evol_inter.loc[index, 'bm:year coeff' ] > 0):
                self.loc[index, f'bbm evol linked with diff. evol {outcome} between groups'] = f"Higher increase of BBM linked with higher increase of the cognitive score over time in the control group while the intervention reduces or cancels the negative effect of the BBM"
            elif (group_evol_evol_inter.loc[index, 'group:bm:year coeff' ] > 0)  and (group_evol_evol_inter.loc[index, f'bm:year {pval_type}' ] < 0.1) and (group_evol_evol_inter.loc[index, 'bm:year coeff' ] > 0):
                self.loc[index, f'bbm evol linked with diff. evol {outcome} between groups'] = f"Higher increase of BBM linked with higher increase of the cognitive score over time in the control group while the intervention increase the positive effect of the BBM"
            elif (group_evol_evol_inter.loc[index, 'group:bm:year coeff' ] < 0)  and (group_evol_evol_inter.loc[index, f'bm:year {pval_type}' ] < 0.1) and (group_evol_evol_inter.loc[index, 'bm:year coeff' ] < 0):
                self.loc[index, f'bbm evol linked with diff. evol {outcome} between groups'] = f"Higher increase of BBM linked with lower increase (or even decrease) of the cognitive score over time in the control group while the intervention increase the negative effect of the BBM"
            elif (group_evol_evol_inter.loc[index, 'group:bm:year coeff' ] > 0)  and (group_evol_evol_inter.loc[index, f'bm:year {pval_type}' ] > 0.1) :
                self.loc[index, f'bbm evol linked with diff. evol {outcome} between groups'] = f"BBM evol does not influence the evolution of the cognitive score in the control group but higher increase of BBM value in the intervention group is linked with faster cognitive improvement"
            elif (group_evol_evol_inter.loc[index, 'group:bm:year coeff' ] < 0)  and (group_evol_evol_inter.loc[index, f'bm:year {pval_type}' ] > 0.1) :
                self.loc[index, f'bbm evol linked with diff. evol {outcome} between groups'] = f"BBM evol does not influence the evolution of the cognitive score in the control group but higher increase of BBM value in the intervention group is linked with slower cognitive improvement"
                    
