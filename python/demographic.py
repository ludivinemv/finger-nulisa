##############################
# Purpose: Analyse data demographics

# Date creation: 2025-07-17
# Author: Ludivine Morvan

# Input data: 
#       luovutettavat_85_1_merge.csv

#Output data:
#       group_stats.csv
#       main_statistics.csv
#       apoe4_histo.png
#       multilifestyle_over_years_statistics.csv

##############################


#! PUBLIC IMPORTS
import os, sys
import pandas as pd
import numpy as np  
import ast

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import shapiro

#! PRIVATE IMPORTS
sys.path.append('P:/H1_FINGER AD BBM/')
sys.path.append('P:/H1_FINGER AD BBM/Nulisa Biomarkers/Scripts/')

from projectspecific.project_functions import get_data, get_histogram

from utils.classes import StatsDataFrame

#! GLOBAL VARIABLES
from projectspecific.config import DEMO_PATH

#! FUNCTIONS
def get_group_stats(df: pd.DataFrame, subset:str) -> pd.DataFrame:
  """Get group statistics (number and rate) for intervention and control groups

  Args:
      df (pd.DataFrame): _input dataframe
      subset (str): _subset name

  Returns:
      pd.DataFrame: _group statistics dataframe
  """
  df_stats_inter = pd.DataFrame(columns=['Group', 'Number', 'Rate (%)'], index = np.arange(3))
  df_stats_inter.loc[0,'Group'] = 'Intervention'    
  df_stats_inter.loc[1,'Group'] = 'Control'
  df_stats_inter.loc[2,'Group'] = 'Total'
  
  df_stats_inter.loc[0,'Number'] = df[df['sub_sample_id']==2]['ID'].nunique()
  df_stats_inter.loc[1,'Number'] = df[df['sub_sample_id']==3]['ID'].nunique()
  df_stats_inter.loc[2,'Number'] = df['ID'].nunique()
  
  df_stats_inter.loc[0,'Rate (%)'] = round(df[df['sub_sample_id']==2]['ID'].nunique()/df['ID'].nunique()*100,2)
  df_stats_inter.loc[1,'Rate (%)'] = round(df[df['sub_sample_id']==3]['ID'].nunique()/df['ID'].nunique()*100,2)
  df_stats_inter.loc[2,'Rate (%)'] = round(df['ID'].nunique()/df['ID'].nunique()*100,2)

  df_stats_inter.to_csv(DEMO_PATH/subset/'group_stats.csv', index=False)
  
  return df_stats_inter


if __name__ == "__main__":   
  
  color_carrier ="#0072B2"
  color_noncarrier= "#E69F00"
  color_control="#4D4D4D"
  color_interv = "#CC79A7"
  subset = 'mITT'

  #! Load the data
  df_all, meta = get_data()

  
  for subset in [
                 #'all_patients', 
                 #'mitt'
                 'mitt_nulisa'
                 ]: # for each subsets
      # select the subset
      col_nulisa = df_all.columns[df_all.columns.str.startswith('nulisa')].tolist()
      col_baseline = df_all.columns[
          df_all.columns.str.startswith('nulisa') & df_all.columns.str.endswith('_0y')
      ].tolist()
      col_2y = df_all.columns[
          df_all.columns.str.startswith('nulisa') & df_all.columns.str.endswith('_2y')
      ].tolist()
      df = df_all[df_all['mITT']==1] if subset in ['mitt','mitt_nulisa'] else df_all         
      #df_02 = df.dropna(subset=col_nulisa, how='all')
      df = df.dropna(subset=col_baseline, how='all')

      # if the folder does not exist, create it
      os.makedirs(DEMO_PATH/subset, exist_ok=True)




      #! #################################################################
      #! Tree
      #! #################################################################
      
      # Split into groups
      df_interv = df[df['group'] == 'Intervention'].copy()
      df_control = df[df['group'] == 'Control'].copy()

      # Variables to check for missing data
      cols = ['age_0y', 'sex', 'educ_years_0y', 'bmi_0y', 'campid']
      cols_all = ['apoe4_status', 'ztotal_composite_0y','lumipulse_ptau217_0y']
        
      print(df[cols_all].isna().sum())
      print(df[cols].isna().sum())

      # Process each group
      for name, data in [('Intervention', df_interv), ('Control', df_control)]:
          print(f'\n--- {name} ---')
          print(f'Total individuals: {len(data)}')

          # Missing data per variable
          print('\nMissing values per variable:')
          print(data[cols].isna().sum())

          # df_1: no missing values in any of the specified variables
          df_1 = data.dropna(subset=cols)
          print(f'\n No missing age, sex, education, bmi, or site: {len(df_1)}')

          # df_2: no missing ztotal_composite_0y
          df_2 = df_1.dropna(subset=['ztotal_composite_0y'])
          print(f'\nNo missing ztotal_composite_0y: {len(df_2)}')

          # df_2: no missing ztotal_composite_0y
          df_3 = df_2.dropna(subset=['ztotal_composite_2y'])
          print(f'\n cognitive score at 2 years: {len(df_3)}')

          # df_2: no missing ztotal_composite_0y
          df_3 = df_1.dropna(subset=col_2y, how='all')
          print(f'\nat least one Nulisa at 2 years: {len(df_3)}')


      #! #################################################################
      #! Nulisa mean 
      #! #################################################################
      

      cols = df.columns[
          df.columns.str.startswith('nulisa') &
          df.columns.str.endswith('_0y')
      ]

      # Calculate mean (SD) for each group
      groups = {
          'Total': df,
          'Control': df[df['group'] == 'Control'],
          'Intervention': df[df['group'] == 'Intervention']
      }

      table = pd.DataFrame(index=cols)

      for name, data in groups.items():
          table[name] = data[cols].apply(
              lambda x: f"{x.mean():.2f} ({x.std():.2f})"
          )

      print(table)
      table.to_excel(DEMO_PATH/'nulisa_mean_sd_0y_by_group.xlsx')
      #! #################################################################
      #! Group distribution table (ratio from intervention and control groups) 
      #! #################################################################
      
      df_group_stats = get_group_stats(df, subset=subset)
        

      #! #################################################################
      #! Get main distribution table #####################################
      #! #################################################################
      
      # Create a table that will contain statistics
      df_stats = StatsDataFrame() # StatsDataFrame is a custom class defined in utils/classes.py

      #! Sex
      dict_sex = meta[meta['column']=='sex']['values'].values[0]
      dict_sex = ast.literal_eval(dict_sex)

      df_sex = df.copy()
      df_sex['sex'] = df_sex['sex'].replace(dict_sex)

      df_stats.add_categorical_stats(df_sex, characteristic = "Sex", column_name = "sex")
      
      #! Age
      df_stats.add_numerical_stats(df, characteristic = "Age at baseline", column_name = "age_0y")

      #! Age classes      
      get_histogram(df, 'age_0y', ylabel = 'Age group', path=DEMO_PATH/subset)
  
      #! Campid distribution
      dict_campid = meta[meta['column']=='campid']['values'].values[0]
      dict_campid = ast.literal_eval(dict_campid)

      df_camp = df[['ID','campid', 'group']].drop_duplicates()
      df_camp['campid'] = df_camp['campid'].replace(dict_campid) #replace number of the site with the city
      
      df_stats.add_categorical_stats(df_camp, characteristic = "Study place", column_name = "campid")
      
      #! Apoe4       
      df_apoe = df.copy()
      df_apoe['apoe4_status'] = df_apoe['apoe4_status'].replace({1:'Apoe4 carrier', 0: 'Apoe4 non-carrier'})

      df_stats.add_categorical_stats(df_apoe, characteristic = "APOE4", column_name = "apoe4_status")

      #! Other descriptive data
      df_stats.add_numerical_stats(df, characteristic = "Education", column_name = "educ_years_0y")
      df_stats.add_numerical_stats(df, characteristic = "Lumipulse pTau217 at baseline", column_name = "lumipulse_ptau217_0y")
      df_stats.add_numerical_stats(df, characteristic = "BMI at baseline", column_name = "bmi_0y")
      df_stats.add_numerical_stats(df, characteristic = "Cholesterol at baseline", column_name = "fs_kol_0y")
      df_stats.add_numerical_stats(df, characteristic = "Creatinine at baseline", column_name = "fs_krea_0y")

      df_stats.to_csv(DEMO_PATH/subset/'main_statistics.csv', index=False) # save the table with statistics


      #! Create a table that will contain statistics on multilifestyle index over the years        
      df_lifestyle = df[['ID','group','multilifestyle_0y','multilifestyle_1y','multilifestyle_2y']]
      df_lifestyle['change_0_2'] = df_lifestyle['multilifestyle_2y'] - df_lifestyle['multilifestyle_0y']

      df_stats_multi = StatsDataFrame()
      df_stats_multi.add_numerical_stats(df_lifestyle, characteristic = "Lifestyle index at baseline", column_name = "multilifestyle_0y")
      df_stats_multi.add_numerical_stats(df_lifestyle, characteristic = "Lifestyle index at 1 year",   column_name = "multilifestyle_1y")
      df_stats_multi.add_numerical_stats(df_lifestyle, characteristic = "Lifestyle index at 2 years",  column_name = "multilifestyle_2y")
      df_stats_multi.add_numerical_stats(df_lifestyle, characteristic = "Lifestyle index change between 0 and 2 years",  column_name = "change_0_2")

      df_stats_multi.to_csv(DEMO_PATH/subset/'multilifestyle_over_years_statistics.csv', index=False)

      #! Create a table that will contain statistics on cognition over the years        
      df_lifestyle = df[['ID','group','ztotal_composite_0y','ztotal_composite_1y','ztotal_composite_2y']]
      df_lifestyle['change_0_2'] = df_lifestyle['ztotal_composite_2y'] - df_lifestyle['ztotal_composite_0y']

      df_stats_multi = StatsDataFrame()
      df_stats_multi.add_numerical_stats(df_lifestyle, characteristic = "Ztotal composite at baseline", column_name = "ztotal_composite_0y")
      df_stats_multi.add_numerical_stats(df_lifestyle, characteristic = "Ztotal composite at 1 year",   column_name = "ztotal_composite_1y")
      df_stats_multi.add_numerical_stats(df_lifestyle, characteristic = "Ztotal composite at 2 years",  column_name = "ztotal_composite_2y")
      df_stats_multi.add_numerical_stats(df_lifestyle, characteristic = "Ztotal composite change between 0 and 2 years",  column_name = "change_0_2")

      df_stats_multi.to_csv(DEMO_PATH/subset/'ztotal_composite_over_years_statistics.csv', index=False)

      #! #################################################################
      #! APOE4 Nulisa according to Apoe4 status ##########################
      #! #################################################################
      
      df_apoe2 = df[['ID','nulisa_APOE4_0y','apoe4_status']].dropna()

      sns.histplot(df_apoe2[df_apoe2['apoe4_status']==1]['nulisa_APOE4_0y'], kde=True, label = 'Apoe4 carrier', color= color_carrier)
      sns.histplot(df_apoe2[df_apoe2['apoe4_status']==0]['nulisa_APOE4_0y'], kde=True, label = 'Apoe4 non-carrier', color= color_noncarrier)
      plt.legend()
      plt.savefig(DEMO_PATH/subset/'apoe4_histo.png')
      plt.show()
      plt.close()



      #! #################################################################
      #! Get Multilifestyle distribution #################################
      #! #################################################################

      #! Create a table that will contain statistics on multilifestyle index over the years        
      df_lifestyle = df[['ID','group','multilifestyle_0y','multilifestyle_1y','multilifestyle_2y']]
      df_lifestyle['change_0_2'] = df_lifestyle['multilifestyle_2y'] - df_lifestyle['multilifestyle_0y']

      df_stats_multi = StatsDataFrame()
      df_stats_multi.add_numerical_stats(df_lifestyle, characteristic = "Lifestyle index at baseline", column_name = "multilifestyle_0y")
      df_stats_multi.add_numerical_stats(df_lifestyle, characteristic = "Lifestyle index at 1 year",   column_name = "multilifestyle_1y")
      df_stats_multi.add_numerical_stats(df_lifestyle, characteristic = "Lifestyle index at 2 years",  column_name = "multilifestyle_2y")
      df_stats_multi.add_numerical_stats(df_lifestyle, characteristic = "Lifestyle index change between 0 and 2 years",  column_name = "change_0_2")

      df_stats_multi.to_csv(DEMO_PATH/subset/'multilifestyle_over_years_statistics.csv', index=False)

      #! Histograms of multilifestyle index over the years
      get_histogram(df_lifestyle, 'multilifestyle_0y', path = DEMO_PATH/subset)
      get_histogram(df_lifestyle, 'multilifestyle_1y', path = DEMO_PATH/subset)
      get_histogram(df_lifestyle, 'multilifestyle_2y', path = DEMO_PATH/subset)


      #! Normality test of multilifestyle at baseline
      stat, p = shapiro(df_lifestyle['multilifestyle_0y'].dropna())
      normal_status = p > 0.05
      
      print(f"Normal status of multilifestyle_0y: {normal_status}")
     

      #! #################################################################
      #! Get Adherence distribution ######################################
      #! #################################################################

      #! Create a table that will contain statistics on multilifestyle index over the years        
      df_adherence = df[['ID','group','c_diet_parti','c_cog_parti','c_exe_parti','c_cvd_parti','multiparti']]

      df_stats_adh = StatsDataFrame()
      df_stats_adh.add_numerical_stats(df_adherence, characteristic = "Total adherence", column_name = "multiparti")
      df_stats_adh.add_numerical_stats(df_adherence, characteristic = "Adherence dietetics",   column_name = "c_diet_parti")
      df_stats_adh.add_numerical_stats(df_adherence, characteristic = "Adherence cognition",   column_name = "c_cog_parti")
      df_stats_adh.add_numerical_stats(df_adherence, characteristic = "Adherence exercise",   column_name = "c_exe_parti")
      df_stats_adh.add_numerical_stats(df_adherence, characteristic = "Adherence cvd-int",   column_name = "c_cvd_parti")

      df_stats_adh.to_csv(DEMO_PATH/subset/'adherence_statistics.csv', index=False)

      #! Histograms of multilifestyle index over the years
      get_histogram(df_adherence, 'multiparti', path = DEMO_PATH/subset)
      get_histogram(df_adherence, 'c_diet_parti', path = DEMO_PATH/subset)
      get_histogram(df_adherence, 'c_cog_parti', path = DEMO_PATH/subset)
      get_histogram(df_adherence, 'c_exe_parti', path = DEMO_PATH/subset)
      get_histogram(df_adherence, 'c_cvd_parti', path = DEMO_PATH/subset)


      #! Normality test of multilifestyle at baseline
      stat, p = shapiro(df_adherence['multiparti'].dropna())
      normal_status = p > 0.05
      
      print(f"Normal status of multiparti: {normal_status}")
     