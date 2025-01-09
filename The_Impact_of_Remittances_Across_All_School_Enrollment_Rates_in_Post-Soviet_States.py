"""
The Impact of Remittances on Every Level of School Enrollment in Post-Soviet Countries
"""

import pandas as pd
import numpy as np
import re
import statsmodels.api as sm
from sys import displayhook
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.linear_model import OLSResults
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

# File paths containg the raw data for the variables used in the research.
ter_enr_file = './data/School_enrollment,_tertiary_(%_gross).csv'
sec_enr_file = './data/School_enrollment,_secondary_(%_gross).csv'
prim_enr_file = './data/School_enrollment,_primary_(%_gross).csv'
remit_file = './data/Personal_remittances,_received_(current_US$).csv'
ex_rate_file = './data/Official_exchange_rate_(LCU_per_US$,_period_average).csv'
exp_file = './data/Households_and_NPISHs_Final_consumption_expenditure,_PPP_(current_international_$).csv'
pop_file = './data/Population,_total.csv'
ppp_file = './data/GDP_per_capita,_PPP_(current_international_$).csv'

''' Create a list with the targeted Post-Soviet countries to filter the following variable dataframes with those countries only '''
# for Turkmenistan, there was no remittance-related data provided
# for Kazakhstan, remittances comprised on average 0.2 % of the country's GDP
ps_ctry = ps_ctry = ['Armenia', 'Azerbaijan', 'Belarus', 'Estonia', 'Georgia', 'Kyrgyz Republic', \
                     'Latvia', 'Lithuania', 'Moldova', 'Tajikistan', 'Ukraine', 'Uzbekistan']

len(ps_ctry) # 12 countries excluding Russia, Kazakhstan, Turkmenistan
processed_from_date = '2000/1/1'

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

print('''********************************* START: READ & TIDY UP THE DATA *********************************''')

# Create a function to preprocess all the variables in a desirable format of panel data
def prep_var(file_path):
    var = pd.read_csv(file_path, header=4)
    var = var.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis = 1)
    var.columns = list(var.columns[:1]) + [pd.to_datetime(int(year), format='%Y') for year in var.columns[1:]]
    var = var.rename(columns={'Country Name' : 'country'}).set_index('country').T
    var = var[ps_ctry].reset_index().melt(id_vars=['index'])
    var.columns = ['Date', 'country', re.sub(r'./data/|.csv', '', file_path)]
    var = var.query( " Date >= '" + processed_from_date + "'" )
    return(var)

''' Var1. Annual Tertiary school enrollment's data (% gross) (Last Updated: 2024-03-28) '''
ter_enr = prep_var(ter_enr_file)
ter_enr.columns = ['date', 'country', 'ter_enr_rate']
ter_enr['ter_enr_rate'] = ter_enr['ter_enr_rate'].apply(lambda x : float(x) if x!=".." else np.nan) / 100 # to convert percentage to number (e.g. 3.15 (%) to 0.0315)
ter_enr.groupby('country')['ter_enr_rate'].apply(lambda x: x.isnull().sum())
ter_enr = ter_enr.reset_index(drop=True)
print(ter_enr.isnull().sum())
# Azerbaijan - 7, Tajikistan - 6 null values

# ter_enr.rename(columns={'ter_enr_rate':'Tertiary enrollment rate', 'date':'Date'}, inplace=True)
# sns.set(style='whitegrid', rc={"grid.linewidth": 0.1}, font_scale=1.1)
# sns.lineplot(data=ter_enr, x='Date', y='Tertiary enrollment rate', hue='country')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.show()

''' Var1.2 Annual Secondary school enrollment's data (% gross) (Last Updated: 2024-03-28) '''
sec_enr = prep_var(sec_enr_file)
sec_enr.columns = ['date', 'country', 'sec_enr_rate']
sec_enr['sec_enr_rate'] = sec_enr['sec_enr_rate'].apply(lambda x : float(x) if x!=".." else np.nan) / 100
sec_enr.groupby('country')['sec_enr_rate'].apply(lambda x: x.isnull().sum())
sec_enr = sec_enr.reset_index(drop=True)
print(sec_enr.isnull().sum())
# Azerbaijan - 10, Tajikistan - 9 null values

# sec_enr.rename(columns={'sec_enr_rate':'Secondary enrollment rate', 'date':'Date'}, inplace=True)
# sns.set(style='whitegrid', rc={"grid.linewidth": 0.1}, font_scale=1.1)
# sns.lineplot(data=sec_enr, x='Date', y='Secondary enrollment rate', hue='country')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.show()

''' Var1.3 Annual Primary school enrollment's data (% gross) (Last Updated: 2024-03-28) '''
prim_enr = prep_var(prim_enr_file)
prim_enr.columns = ['date', 'country', 'prim_enr_rate']
prim_enr['prim_enr_rate'] = prim_enr['prim_enr_rate'].apply(lambda x : float(x) if x!=".." else np.nan) / 100
prim_enr.groupby('country')['prim_enr_rate'].apply(lambda x: x.isna().sum())
# prim_enr.loc[prim_enr['country']=='Georgia']
print(prim_enr.isnull().sum())
prim_enr = prim_enr.reset_index(drop=True)
# Tajikistan - 5 null values

# prim_enr.rename(columns={'prim_enr_rate':'Primary enrollment rate', 'date':'Date'}, inplace=True)
# sns.set(style='whitegrid', rc={"grid.linewidth": 0.1}, font_scale=1.1)
# sns.lineplot(data=prim_enr, x='Date', y='Primary enrollment rate', hue='country')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.show()

''' Var2. Household expenditure (current international $) (Last Updated: 2024/03/28)'''
exp = prep_var(exp_file)
exp.columns = ['date', 'country', 'hhld_exp $']
exp['hhld_exp $'] = exp['hhld_exp $'].astype(float)
exp.groupby('country')['hhld_exp $'].apply(lambda x: x.isnull().sum())
exp = exp.reset_index(drop=True)
print(exp.isnull().sum())
# Uzbekistan - 11 null values

''' Var4. Personal remittances, received (current US$) (Last Updated: 2023/12/18) '''
remit = prep_var(remit_file)
remit.columns = ['date', 'country', 'rem $']
remit.groupby('country')['rem $'].apply(lambda x: x.isnull().sum())
remit = remit.reset_index(drop=True)
print(remit.isnull().sum())
# Uzbekistan - 5 null values

# ''' Var4.2 Official exchange rate (LCU per US$, period average) '''
# ex_rate = prep_var(ex_rate_file)
# ex_rate.columns = ['date', 'country', 'exchange_rate']
# ex_rate.groupby('country')['exchange_rate'].apply(lambda x: x.isnull().sum())
# ex_rate = ex_rate.reset_index(drop=True)
# print(ex_rate.isnull().sum())

# Uzbekistan - 13, Tajikistan - 2 null values

''' Var5. Population, total (Last Updated: 2023/12/18) to calculate household expenditure per capita '''
pop = prep_var(pop_file)
pop.columns = ['date', 'country', 'pop']
pop.groupby('country')['pop'].apply(lambda x: x.isnull().sum())
pop = pop.reset_index(drop=True)

''' Var6. GDP per capita, PPP (current international $) (Last Updated: 2023/12/18) '''
ppp = prep_var(ppp_file)
ppp.columns = ['date', 'country', 'PPP_pc $']
ppp.groupby('country')['PPP_pc $'].apply(lambda x: x.isnull().sum())
ppp = ppp.reset_index(drop=True)
print(ppp.isnull().sum())

''' Merge & sort the data 
        every ds are of len=240, from 2000-23
        Null values for each targeted variable are:
            tertiary enr. rate = 18,
            secondary enr. rate = 47,
            primary enr. rate = 12,
            remittance column = 7,
            household expenditure = 14,
            exchange rate column = 12, 
            ppp column = 0 '''
                
for i, df in enumerate([ter_enr, sec_enr, prim_enr, exp, remit, pop, ppp]):
    if i==0: panel_df = df.copy()
    else: panel_df = pd.merge(left=panel_df, right=df, how='inner', left_on=['date','country'], \
                              right_on=['date', 'country'])

panel_df = panel_df.sort_values(by=['country', 'date'])
panel_df.dropna().groupby("country")["date"].apply(lambda x : print("THE LIST OF DATES _________________ :", list(x)[0]) )
panel_df.describe()

print('''*********************************** END: READ & TIDY UP THE DATA *********************************''')

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

print('''********************************* START: DATA PROCESSING *********************************''')

rem_lag = 1
ppp_lag = 1

''' Enrollment rate (dependent variables 1, 2, 3) '''
panel_df['ter_enr_rate_diff'] = panel_df['ter_enr_rate'].diff()
panel_df['sec_enr_rate_diff'] = panel_df['sec_enr_rate'].diff()
panel_df['prim_enr_rate_diff'] = panel_df['prim_enr_rate'].diff()

# 1) Get the time-series (country-level within the targeted total period) mean values of the tertiary, secondary and primary enrollment rates' columns and rename the columns appropriately
mean_ter_enroll = panel_df.groupby('country')[['ter_enr_rate_diff']].mean().reset_index()
mean_ter_enroll.columns = ['country','ter_enr_rate_diff_mean']
mean_sec_enroll = panel_df.groupby('country')[['sec_enr_rate_diff']].mean().reset_index()
mean_sec_enroll.columns = ['country','sec_enr_rate_diff_mean']
mean_prim_enroll = panel_df.groupby('country')[['prim_enr_rate_diff']].mean().reset_index()
mean_prim_enroll.columns = ['country','prim_enr_rate_diff_mean']

# 2) Get the time-series standard deviations of the enrollment rates' columns and rename the columns appropriately
std_ter_enroll = panel_df.groupby('country')[['ter_enr_rate_diff']].std().reset_index()
std_ter_enroll.columns = ['country','ter_enr_rate_diff_std']
std_sec_enroll = panel_df.groupby('country')[['sec_enr_rate_diff']].std().reset_index()
std_sec_enroll.columns = ['country','sec_enr_rate_diff_std']
std_prim_enroll = panel_df.groupby('country')[['prim_enr_rate_diff']].std().reset_index()
std_prim_enroll.columns = ['country','prim_enr_rate_diff_std']

# 3) Add the means and the standard deviations of each country for all the 3 enrollment rates to the panel_df
panel_df = pd.merge(pd.merge(panel_df, mean_ter_enroll, how='left', left_on='country', right_on='country'), std_ter_enroll, how='left', left_on='country', right_on='country')
panel_df = pd.merge(pd.merge(panel_df, mean_sec_enroll, how='left', left_on='country', right_on='country'), std_sec_enroll, how='left', left_on='country', right_on='country')
panel_df = pd.merge(pd.merge(panel_df, mean_prim_enroll, how='left', left_on='country', right_on='country'), std_prim_enroll, how='left', left_on='country', right_on='country')

# 4) For each country, subtract its mean value from every observation and divide by that country's standard deviation.
panel_df['ter_enr_rate_diff_standard'] = (panel_df['ter_enr_rate_diff'] - panel_df['ter_enr_rate_diff_mean']) / panel_df['ter_enr_rate_diff_std']
panel_df['sec_enr_rate_diff_standard'] = (panel_df['sec_enr_rate_diff'] - panel_df['sec_enr_rate_diff_mean']) / panel_df['sec_enr_rate_diff_std']
panel_df['prim_enr_rate_diff_standard'] = (panel_df['prim_enr_rate_diff'] - panel_df['prim_enr_rate_diff_mean']) / panel_df['prim_enr_rate_diff_std']

# panel_df.rename(columns={'prim_enr_rate_diff_standard':'Primary enrollment rate', 'date':'Date'}, inplace=True)
# sns.set(style='whitegrid', rc={"grid.linewidth": 0.1}, font_scale=1.1)
# sns.lineplot(data=panel_df, x='Date', y='Primary enrollment rate', hue='country')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.show()

# panel_df.rename(columns={'sec_enr_rate_diff_standard':'Secondary enrollment rate', 'date':'Date'}, inplace=True)
# sns.set(style='whitegrid', rc={"grid.linewidth": 0.1}, font_scale=1.1)
# sns.lineplot(data=panel_df, x='Date', y='Secondary enrollment rate', hue='country')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.show()

# panel_df.rename(columns={'ter_enr_rate_diff_standard':'Tertiary enrollment rate', 'date':'Date'}, inplace=True)
# sns.set(style='whitegrid', rc={"grid.linewidth": 0.1}, font_scale=1.1)
# sns.lineplot(data=panel_df, x='Date', y='Tertiary enrollment rate', hue='country')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.show()

''' Remittance (independent variable) '''

# Take the household expenditure per capita, total household expenditure / total population
panel_df['hhld_exp_pc $'] = panel_df['hhld_exp $'] / panel_df['pop']

# Take the purchasing power relative to people's expenditure showing how much people can afford after taking out the expenses
panel_df['PPP/hhld_exp_pc'] = panel_df['PPP_pc $']/panel_df['hhld_exp_pc $'] 

# Adjust the remittances in current USD with {PPP in international $ devided by the household expenditure} as purchasing power of dollar is different in every country
panel_df['PPP_rem'] = panel_df['rem $'] * panel_df['PPP/hhld_exp_pc']

# Take the PPP-adjusted remittance per capita
panel_df['PPP_rem_pc'] = (panel_df['PPP_rem'] / panel_df['pop'])

# Take the percentage change
panel_df['PPP_rem_pc_pct'] = panel_df.groupby('country')['PPP_rem_pc'].pct_change()

# Processed remittance per capita lagged by 1 year
panel_df['PPP_rem_pc_pct_{t-1}'] = panel_df['PPP_rem_pc_pct'].shift(rem_lag)

# panel_df.rename(columns={'PPP_rem_pc_pct_{t-1}':'Remittance', 'date':'Date'}, inplace=True)
# sns.set(style='whitegrid', rc={"grid.linewidth": 0.1}, font_scale=1.1)
# sns.lineplot(data=panel_df, x='Date', y='Remittance', hue='country')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.show()

''' Purchasing power parity (control variable)'''

panel_df = panel_df.sort_values(by = ['country', 'date'])
panel_df = panel_df.dropna()

# Share of the remittance per capita from PPP per capita = total remittance * (PPP pc / household expenditure pc) 
panel_df['PPP_rem_pc $'] = (panel_df['rem $']/ panel_df['pop']) * panel_df['PPP/hhld_exp_pc'] 

# PPP per capita - (Share of the remittance from PPP per capita)
panel_df['(PPP - PPP_rem)_pc $'] = panel_df['PPP_pc $'] - panel_df['PPP_rem_pc $']

# Take the percentage change
panel_df['(PPP - PPP_rem)_pc_pct $'] = panel_df.groupby('country')['(PPP - PPP_rem)_pc $'].pct_change()

# Lag by 1 year
panel_df['(PPP - PPP_rem)_pc_pct_{t-1} $'] = panel_df['(PPP - PPP_rem)_pc_pct $'].shift(ppp_lag)

# panel_df.rename(columns={'(PPP - PPP_rem)_pc_pct_{t-1} $':'Purchasing power parity', 'date':'Date'}, inplace=True)
# sns.set(style='whitegrid', rc={"grid.linewidth": 0.1}, font_scale=1.1)
# sns.lineplot(data=panel_df, x='Date', y='Purchasing power parity', hue='country')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.show()

panel_df.rename(columns={'(PPP - PPP_rem)_pc_pct_{t-1} $':'Purchasing power parity', 'PPP_rem_pc_pct_{t-1}':'Remittance', \
                'ter_enr_rate_diff_standard':'Enrollment in tertiary', 'sec_enr_rate_diff_standard':'Enrollment in secondary', \
                'prim_enr_rate_diff_standard':'Enrollment in primary'}, inplace=True)

ter_enr_reg_name  = 'Enrollment in tertiary'
sec_enr_reg_name  = 'Enrollment in secondary'
prim_enr_reg_name  = 'Enrollment in primary'
rem_reg_name  = 'Remittance'
ppp_reg_name  = 'Purchasing power parity'
reg_var_list  = [ter_enr_reg_name, sec_enr_reg_name, prim_enr_reg_name, rem_reg_name, ppp_reg_name]

''' winsorize ''' # Remove the outliers
# Note 1: The outliers are defined the data points that are at most 3 standard deviations \
# away (above and below) from the mean since for the normal distribution, the values that are 3 std away from the mean account for 99.73%.
# Note 2: The observations that represent relatively extreme values at certain country-levels, \
# are left in the dataframe.

sig = 3
for rn in reg_var_list:
        print(panel_df[rn])
        panel_df[rn] = panel_df[rn].clip( lower = panel_df[rn].mean() - (sig * panel_df[rn].std()), \
                upper = (sig * panel_df[rn].std()) + panel_df[rn].mean() )

panel_df = panel_df.sort_values(by = 'date')
# Generate a dummy variable for each year. 
for year in panel_df['date'].unique():
        panel_df[ year ] = ( panel_df['date'] == year ).apply( lambda x : int(x) )

panel_df = panel_df[ ['date', 'country'] + reg_var_list + list(panel_df.columns[-21:])]
# panel_df = panel_df.query( " date >= '" + processed_from_date + "'" )
len(panel_df['country'].unique())
panel_df.describe()

"""_____START PLOTS_____"""
# Check linear relationship between the dependent and independent variables, scatterplot
sns.set(style='whitegrid', rc={"grid.linewidth": 0.1}, font_scale=2)
sns.set_context("paper", font_scale=1.2) 
splot = sns.pairplot(panel_df[reg_var_list+["country"]], hue="country", grid_kws={"despine": False})
plt.show()

# Check correlation between the variables with the heatmap.
sns.set(font_scale=1.4)
htmp = sns.heatmap(panel_df[panel_df.columns[2:7]].corr(), vmin=-1, vmax=1, annot=True, fmt=".2f", linewidth=.5, cmap="vlag")
htmp.set_xticklabels(htmp.get_xmajorticklabels(), fontsize=16)
htmp.set_yticklabels(htmp.get_ymajorticklabels(), fontsize=16)
plt.show()

"""_____END PLOTS_____"""

print('''*********************************** END: DATA PROCESSING *********************************''')

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

print('''********************************* START: FIXED EFFECT MODEL, RIDGE REGRESSION, PARTIAL LEAST SQUARES REGRESSION *********************************''')
panel_df = panel_df.reset_index(drop=True)
panel_df = panel_df.sort_values(by = 'date')
panel_df = panel_df.dropna()
panel_df.describe()
prim_endg = panel_df[[reg_var_list[2]]]
sec_endg = pd.DataFrame(panel_df[reg_var_list[1]])
ter_endg = pd.DataFrame(panel_df[reg_var_list[0]])

''' 

Covariance type (c_type variable)
    
    'HC0': White's (1980) heteroskedasticity robust standard errors.
    'HC1', 'HC2', 'HC3': MacKinnon and White's (1985) heteroskedasticity robust standard errors.
    'robust': White”s robust covariance
    'HAC' : heteroskedasticity-autocorrelation robust covariance
    'cluster' : two-way clustered robust standard error (for both cross-sections and years)
    
'''
panel_df.loc[panel_df['date'] == '2005-01-01']
panel_df.loc[panel_df['date'] == '2006-01-01']
panel_df.loc[panel_df['date'] == '2007-01-01']
panel_df.loc[panel_df['date'] == '2008-01-01']
panel_df.loc[panel_df['date'] == '2009-01-01']
panel_df.loc[panel_df['date'] == '2012-01-01']
panel_df.loc[panel_df['date'] == '2012-01-01']
panel_df.loc[panel_df['date'] == '2013-01-01']

print(''' __ START: OLS Regression for Primary Enrollment Rate ____''')
panel_df = panel_df.sort_values(by = 'date')
panel_df = panel_df.dropna()
exog = panel_df[reg_var_list[-2:] + list(panel_df.columns[-21:])]
# fixed_effect_mdl_prim = sm.OLS(prim_endg, sm.add_constant(exog))
fixed_effect_mdl_prim = sm.OLS(prim_endg, sm.add_constant(exog))
# panel_df['date'] = pd.factorize(panel_df['date'], sort = True) [0] + 1
# panel_df['country'] = pd.factorize(panel_df['country'], sort = True) [0] + 1
fitted_fixed_effect_mdl_prim = fixed_effect_mdl_prim.fit(cov_type = 'cluster', cov_kwds={'groups' : np.array(panel_df['country'])})
displayhook(fitted_fixed_effect_mdl_prim.summary())
# 2005-8, 2012, 2018-20

print(''' __ START: OLS Regression for Secondary Enrollment Rate ____''')

exog = panel_df[reg_var_list[-2:] + list(panel_df.columns[-21:])]
fixed_effect_mdl_sec = sm.OLS(sec_endg, sm.add_constant(exog))
fitted_fixed_effect_mdl_sec = fixed_effect_mdl_sec.fit(cov_type = 'cluster', cov_kwds={'groups' : np.array(panel_df['country'])}) # maximum lag considered for the control of autocorrelation.
displayhook(fitted_fixed_effect_mdl_sec.summary())
# 2007

print(''' __ START: OLS Regression for Tertiary Enrollment Rate ____''')

exog = panel_df[reg_var_list[-2:] + list(panel_df.columns[-21:])]
fixed_effect_mdl_ter = sm.OLS(ter_endg, sm.add_constant(exog))
fitted_fixed_effect_mdl_ter = fixed_effect_mdl_ter.fit(cov_type = 'cluster', cov_kwds={'groups' : np.array(panel_df['country'])})
displayhook(fitted_fixed_effect_mdl_ter.summary())
# 2019