import pandas as pd
import glob
import shutil
import sys
import numpy as np
from IPython.display import display, HTML
import random

loan_data = pd.DataFrame()

def try_convert_date(value):
    try:
        new_val = pd.datetime.strptime(value, '%b-%y')
    except:
        new_val = pd.datetime.strptime(value, '%b-%Y')
    return new_val

def clean_loan_data(df):
    df = df.loc[:, (df != 0).any(axis=0)]
    dont_care_list = [
    	'annual_inc_joint',
    	'debt_settlement_flag_date',
    	'deferral_term',
    	'desc',
    	'dti_joint',
    	'hardship_amount',
    	'hardship_dpd',
    	'hardship_end_date',
    	'hardship_last_payment_amount',
    	'hardship_length',
    	'hardship_loan_status',
    	'hardship_payoff_balance_amount',
    	'hardship_reason',
    	'hardship_start_date',
    	'hardship_status',
    	'hardship_type',
        'hardship_flag',
    	'id',
    	'member_id',
    	'orig_projected_additional_accrued_interest',
    	'payment_plan_start_date',
    	'revol_bal_joint',
    	'sec_app_chargeoff_within_12_mths',
    	'sec_app_collections_12_mths_ex_med',
    	'sec_app_earliest_cr_line',
    	'sec_app_inq_last_6mths',
    	'sec_app_mort_acc',
    	'sec_app_mths_since_last_major_derog',
    	'sec_app_num_rev_accts',
    	'sec_app_open_acc',
    	'sec_app_open_act_il',
    	'sec_app_revol_util',
    	'settlement_amount',
    	'settlement_date',
    	'settlement_percentage',
    	'settlement_status',
    	'settlement_term',
    	'url',
    	'verification_status_joint',
        'application_type',
        'collection_recovery_fee',
        'debt_settlement_flag',
        'debt_settlement_flag_date',
        'deferral_term',
        'delinq_amnt',
        'disbursement_method',
        'earliest_cr_line',
        'emp_title',
        'hardship_amount',
        'hardship_dpd',
        'hardship_end_date',
        'hardship_flag',
        'hardship_last_payment_amount',
        'hardship_length',
        'hardship_loan_status',
        'hardship_payoff_balance_amount',
        'hardship_reason',
        'hardship_start_date',
        'hardship_status',
        'issue_d',
        'last_credit_pull_d',
        'last_pymnt_amnt',
        'last_pymnt_d',
        'next_pymnt_d',
        'orig_projected_additional_accrued_interest',
        'out_prncp',
        'out_prncp_inv',
        'payment_plan_start_date',
        'policy_code',
        'pymnt_plan',
        'recoveries',
        'sec_app_earliest_cr_line',
        'settlement_amount',
        'settlement_date',
        'settlement_percentage',
        'settlement_status',
        'settlement_term',
        'title',
        'total_pymnt',
        'total_pymnt_inv',
        'total_rec_int',
        'total_rec_late_fee',
        'total_rec_prncp'
    	]
    df['funded_amnt_inv'] = pd.to_numeric(df['funded_amnt_inv'])
    df['term'] = df['term'].str.replace(" months","")
    df['term'] = pd.to_numeric(df['term'])
    df['int_rate'] = df['int_rate'].str.replace("%","")
    df['int_rate'] = pd.to_numeric(df['int_rate'])
    df.drop(df[df.loan_amnt == 0].index, inplace=True)
    df['emp_length'] = df['emp_length'].str.replace("years","")
    df['emp_length'] = df['emp_length'].str.replace("year","")
    df['emp_length'] = df['emp_length'].str.replace(" ","")
    df.loc[(df['emp_length'] != '<1') & 
                  (df['emp_length'] != '1') & 
                  (df['emp_length'] != '2') & 
                  (df['emp_length'] != '3') & 
                  (df['emp_length'] != '4') & 
                  (df['emp_length'] != '5') & 
                  (df['emp_length'] != '6') & 
                  (df['emp_length'] != '7') & 
                  (df['emp_length'] != '8') & 
                  (df['emp_length'] != '9') & 
                  (df['emp_length'] != '10+'), 'emp_length'] = 0
    df['annual_inc'] = pd.to_numeric(df['annual_inc'])
    df['issue_d_month'] = pd.DatetimeIndex(df['issue_d'].apply(try_convert_date)).month
    df['issue_d_year'] = pd.DatetimeIndex(df['issue_d'].apply(try_convert_date)).year
    df.loc[(df['zip_code'] == 0), 'zip_code'] =  random.choice(df['zip_code'].unique())
    df.drop(df[df.earliest_cr_line == '0'].index, inplace=True)
    df['earliest_cr_line_month'] = pd.DatetimeIndex(df['earliest_cr_line'].apply(try_convert_date)).month
    df['earliest_cr_line_year'] = pd.DatetimeIndex(df['earliest_cr_line'].apply(try_convert_date)).year
    df[['mths_since_last_delinq']] = df[['mths_since_last_delinq']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['mths_since_last_delinq']] = df[['mths_since_last_delinq']].astype(int)
    df[['mths_since_last_record']] = df[['mths_since_last_record']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['mths_since_last_record']] = df[['mths_since_last_record']].astype(int)
    df['revol_util'] = df['revol_util'].str.replace("%","")
    df['revol_util'] = pd.to_numeric(df['revol_util'])
    df.drop(df[df.last_pymnt_d == '0'].index, inplace=True)
    df.dropna(subset=['last_pymnt_d'], inplace=True)
    df['last_pymnt_d_month'] = pd.DatetimeIndex(df['last_pymnt_d'].apply(try_convert_date)).month
    df['last_pymnt_d_year'] = pd.DatetimeIndex(df['last_pymnt_d'].apply(try_convert_date)).year
    df.dropna(subset=['last_credit_pull_d'], inplace=True)
    df.drop(df[df.last_credit_pull_d == '0'].index, inplace=True)
    df['last_credit_pull_d_month'] = pd.DatetimeIndex(df['last_credit_pull_d'].apply(try_convert_date)).month
    df['last_credit_pull_d_year'] = pd.DatetimeIndex(df['last_credit_pull_d'].apply(try_convert_date)).year
    df.loc[df['mths_since_last_major_derog'].isnull(), 'mths_since_last_major_derog'] = 1200
    df[['mths_since_last_major_derog']] = df[['mths_since_last_major_derog']].astype(int)
    df[['tot_coll_amt']] = df[['tot_coll_amt']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['tot_coll_amt']] = df[['tot_coll_amt']].astype(int)
    df[['tot_cur_bal']] = df[['tot_cur_bal']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['tot_cur_bal']] = df[['tot_cur_bal']].astype(int)
    df[['open_acc_6m']] = df[['open_acc_6m']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['open_act_il']] = df[['open_act_il']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['open_il_12m']] = df[['open_il_12m']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['open_il_24m']] = df[['open_il_24m']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['open_acc_6m']] = df[['open_acc_6m']].astype(int)
    df[['open_act_il']] = df[['open_act_il']].astype(int)
    df[['open_il_12m']] = df[['open_il_12m']].astype(int)
    df[['open_il_24m']] = df[['open_il_24m']].astype(int)
    df[['max_bal_bc']] = df[['max_bal_bc']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['max_bal_bc']] = df[['max_bal_bc']].astype(int)
    df[['all_util']] = df[['all_util']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['total_rev_hi_lim']] = df[['total_rev_hi_lim']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['inq_fi']] = df[['inq_fi']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['total_cu_tl']] = df[['total_cu_tl']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['inq_last_12m']] = df[['inq_last_12m']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['all_util']] = df[['all_util']].astype(int)
    df[['total_rev_hi_lim']] = df[['total_rev_hi_lim']].astype(int)
    df[['inq_fi']] = df[['inq_fi']].astype(int)
    df[['total_cu_tl']] = df[['total_cu_tl']].astype(int)
    df[['inq_last_12m']] = df[['inq_last_12m']].astype(int)
    df[['inq_last_12m']] = df[['inq_last_12m']].astype(int)
    df[['acc_open_past_24mths']] = df[['acc_open_past_24mths']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['avg_cur_bal']] = df[['avg_cur_bal']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['bc_open_to_buy']] = df[['bc_open_to_buy']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['acc_open_past_24mths']] = df[['acc_open_past_24mths']].astype(int)
    df[['avg_cur_bal']] = df[['avg_cur_bal']].astype(int)
    df[['bc_open_to_buy']] = df[['bc_open_to_buy']].astype(int)
    df[['inq_last_12m']] = df[['inq_last_12m']].astype(int)
    df[['mo_sin_rcnt_tl']] = df[['mo_sin_rcnt_tl']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['mo_sin_rcnt_tl']] = df[['mo_sin_rcnt_tl']].astype(int)
    df[['mort_acc']] = df[['mort_acc']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['mths_since_recent_inq']] = df[['mths_since_recent_inq']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['mths_since_recent_revol_delinq']] = df[['mths_since_recent_revol_delinq']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['mths_since_recent_bc']] = df[['mths_since_recent_bc']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['mths_since_recent_bc_dlq']] = df[['mths_since_recent_bc_dlq']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['mort_acc']] = df[['mort_acc']].astype(int)
    df[['mths_since_recent_inq']] = df[['mths_since_recent_inq']].astype(int)
    df[['mths_since_recent_revol_delinq']] = df[['mths_since_recent_revol_delinq']].astype(int)
    df[['mths_since_recent_bc']] = df[['mths_since_recent_bc']].astype(int)
    df[['mths_since_recent_bc_dlq']] = df[['mths_since_recent_bc_dlq']].astype(int)
    df[['num_accts_ever_120_pd']] = df[['num_accts_ever_120_pd']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['num_actv_bc_tl']] = df[['num_actv_bc_tl']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['num_actv_rev_tl']] = df[['num_actv_rev_tl']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['num_bc_sats']] = df[['num_bc_sats']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['num_bc_tl']] = df[['num_bc_tl']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['num_accts_ever_120_pd']] = df[['num_accts_ever_120_pd']].astype(int)
    df[['num_actv_bc_tl']] = df[['num_actv_bc_tl']].astype(int)
    df[['num_actv_rev_tl']] = df[['num_actv_rev_tl']].astype(int)
    df[['num_bc_sats']] = df[['num_bc_sats']].astype(int)
    df[['num_bc_tl']] = df[['num_bc_tl']].astype(int)
    df[['num_il_tl']] = df[['num_il_tl']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['num_op_rev_tl']] = df[['num_op_rev_tl']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['num_rev_accts']] = df[['num_rev_accts']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['num_rev_tl_bal_gt_0']] = df[['num_rev_tl_bal_gt_0']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['num_sats']] = df[['num_sats']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['num_il_tl']] = df[['num_il_tl']].astype(int)
    df[['num_op_rev_tl']] = df[['num_op_rev_tl']].astype(int)
    df[['num_rev_accts']] = df[['num_rev_accts']].astype(int)
    df[['num_rev_tl_bal_gt_0']] = df[['num_rev_tl_bal_gt_0']].astype(int)
    df[['num_sats']] = df[['num_sats']].astype(int)
    df[['num_tl_120dpd_2m']] = df[['num_tl_120dpd_2m']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['num_tl_30dpd']] = df[['num_tl_30dpd']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['num_tl_90g_dpd_24m']] = df[['num_tl_90g_dpd_24m']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['num_tl_op_past_12m']] = df[['num_tl_op_past_12m']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['pct_tl_nvr_dlq']] = df[['pct_tl_nvr_dlq']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['num_tl_120dpd_2m']] = df[['num_tl_120dpd_2m']].astype(int)
    df[['num_tl_30dpd']] = df[['num_tl_30dpd']].astype(int)
    df[['num_tl_90g_dpd_24m']] = df[['num_tl_90g_dpd_24m']].astype(int)
    df[['num_tl_op_past_12m']] = df[['num_tl_op_past_12m']].astype(int)
    df[['pub_rec_bankruptcies']] = df[['pub_rec_bankruptcies']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['percent_bc_gt_75']] = df[['percent_bc_gt_75']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['pub_rec_bankruptcies']] = df[['pub_rec_bankruptcies']].astype(int)
    df[['tot_hi_cred_lim']] = df[['tot_hi_cred_lim']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['total_bal_ex_mort']] = df[['total_bal_ex_mort']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['total_bc_limit']] = df[['total_bc_limit']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['tax_liens']] = df[['tax_liens']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['tot_hi_cred_lim']] = df[['tot_hi_cred_lim']].astype(int)
    df[['total_bal_ex_mort']] = df[['total_bal_ex_mort']].astype(int)
    df[['total_bc_limit']] = df[['total_bc_limit']].astype(int)
    df[['total_il_high_credit_limit']] = df[['total_il_high_credit_limit']].applymap(lambda x: 0 if pd.isnull(x) else x)
    df[['total_il_high_credit_limit']] = df[['total_il_high_credit_limit']].astype(int)
    df = df.loc[df['loan_status'].isin(['Fully Paid', 'Charged Off', 'Default'])]
    df.loc[df['loan_status'] == 'Default', 'loan_status'] = 'Charged Off'
    df['zip_code'] = df['zip_code'].str.replace("xx","")
    df['zip_code'] = pd.to_numeric(df['zip_code'])
    df.drop(df[df.home_ownership.isin(['OTHER', 'NONE', 'ANY'])].index, inplace=True)
    df.drop(dont_care_list, axis=1, inplace=True)
    for f in df.columns:
        df[[f]] = df[[f]].applymap(lambda x: 0 if pd.isnull(x) else x)

for file in glob.glob("Loan*.csv"):
    df = pd.read_csv(file, dtype={'id':float,  
                                  'member_id':float,  
                                  'loan_amnt':int,  
                                  'funded_amnt':int,  
                                  'funded_amnt_inv':str,  
                                  'term':str,  
                                  'int_rate':str,  
                                  'installment':float,  
                                  'grade':str,  
                                  'sub_grade':str,  
                                  'emp_title':str,  
                                  'emp_length':str,  
                                  'home_ownership':str,  
                                  'annual_inc':str,  
                                  'verification_status':str,  
                                  'issue_d':str,  
                                  'loan_status':str,  
                                  'pymnt_plan':str,  
                                  'url':str,  
                                  'desc':str,  
                                  'purpose':str,  
                                  'title':str,  
                                  'zip_code':str,  
                                  'addr_state':str,  
                                  'dti':float,  
                                  'delinq_2yrs':int,  
                                  'earliest_cr_line':str,  
                                  'inq_last_6mths':int,  
                                  'mths_since_last_delinq':float,  
                                  'mths_since_last_record':float,  
                                  'open_acc':int,  
                                  'pub_rec':int,  
                                  'revol_bal':int,  
                                  'revol_util':str,  
                                  'total_acc':int,  
                                  'initial_list_status':str,  
                                  'out_prncp':float,  
                                  'out_prncp_inv':float,  
                                  'total_pymnt':float,  
                                  'total_pymnt_inv':float,  
                                  'total_rec_prncp':float,  
                                  'total_rec_int':float,  
                                  'total_rec_late_fee':float,  
                                  'recoveries':float,  
                                  'collection_recovery_fee':float,  
                                  'last_pymnt_d':str,  
                                  'last_pymnt_amnt':float,  
                                  'next_pymnt_d':str,  
                                  'last_credit_pull_d':str,  
                                  'collections_12_mths_ex_med':int,  
                                  'mths_since_last_major_derog':float,  
                                  'policy_code':int,  
                                  'application_type':str,  
                                  'annual_inc_joint':float,  
                                  'dti_joint':float,  
                                  'verification_status_joint':str,  
                                  'acc_now_delinq':int,  
                                  'tot_coll_amt':float,  
                                  'tot_cur_bal':float,  
                                  'open_acc_6m':float,  
                                  'open_act_il':float,  
                                  'open_il_12m':float,  
                                  'open_il_24m':float,  
                                  'mths_since_rcnt_il':float,  
                                  'total_bal_il':float,  
                                  'il_util':float,  
                                  'open_rv_12m':float,  
                                  'open_rv_24m':float,  
                                  'max_bal_bc':float,  
                                  'all_util':float,  
                                  'total_rev_hi_lim':float,  
                                  'inq_fi':float,  
                                  'total_cu_tl':float,  
                                  'inq_last_12m':float,  
                                  'acc_open_past_24mths':float,  
                                  'avg_cur_bal':float,  
                                  'bc_open_to_buy':float,  
                                  'bc_util':float,  
                                  'chargeoff_within_12_mths':int,  
                                  'delinq_amnt':int,  
                                  'mo_sin_old_il_acct':float,  
                                  'mo_sin_old_rev_tl_op':float,  
                                  'mo_sin_rcnt_rev_tl_op':float,  
                                  'mo_sin_rcnt_tl':float,  
                                  'mort_acc':float,  
                                  'mths_since_recent_bc':float,  
                                  'mths_since_recent_bc_dlq':float,  
                                  'mths_since_recent_inq':float,  
                                  'mths_since_recent_revol_delinq':float,  
                                  'num_accts_ever_120_pd':float,  
                                  'num_actv_bc_tl':float,  
                                  'num_actv_rev_tl':float,  
                                  'num_bc_sats':float,  
                                  'num_bc_tl':float,  
                                  'num_il_tl':float,  
                                  'num_op_rev_tl':float,  
                                  'num_rev_accts':float,  
                                  'num_rev_tl_bal_gt_0':float,  
                                  'num_sats':float,  
                                  'num_tl_120dpd_2m':float,  
                                  'num_tl_30dpd':float,  
                                  'num_tl_90g_dpd_24m':float,  
                                  'num_tl_op_past_12m':float,  
                                  'pct_tl_nvr_dlq':float,  
                                  'percent_bc_gt_75':float,  
                                  'pub_rec_bankruptcies':float,  
                                  'tax_liens':int,  
                                  'tot_hi_cred_lim':float,  
                                  'total_bal_ex_mort':float,  
                                  'total_bc_limit':float,  
                                  'total_il_high_credit_limit':float,  
                                  'revol_bal_joint':float,  
                                  'sec_app_earliest_cr_line':str,  
                                  'sec_app_inq_last_6mths':float,  
                                  'sec_app_mort_acc':float,  
                                  'sec_app_open_acc':float,  
                                  'sec_app_revol_util':float,  
                                  'sec_app_open_act_il':float,  
                                  'sec_app_num_rev_accts':float,  
                                  'sec_app_chargeoff_within_12_mths':float,  
                                  'sec_app_collections_12_mths_ex_med':float,  
                                  'sec_app_mths_since_last_major_derog':float,  
                                  'hardship_flag':str,  
                                  'hardship_type':str,  
                                  'hardship_reason':str,  
                                  'hardship_status':str,  
                                  'deferral_term':float,  
                                  'hardship_amount':float,  
                                  'hardship_start_date':str,  
                                  'hardship_end_date':str,  
                                  'payment_plan_start_date':str,  
                                  'hardship_length':float,  
                                  'hardship_dpd':float,  
                                  'hardship_loan_status':str,  
                                  'orig_projected_additional_accrued_interest':float,  
                                  'hardship_payoff_balance_amount':float,  
                                  'hardship_last_payment_amount':float,  
                                  'disbursement_method':str,  
                                  'debt_settlement_flag':str,  
                                  'debt_settlement_flag_date':str,  
                                  'settlement_status':str,  
                                  'settlement_date':str,  
                                  'settlement_amount':str,  
                                  'settlement_percentage':str,  
                                  'settlement_term':str})
    loan_data = loan_data.append(df)

clean_loan_data(loan_data)
loan_data.to_pickle('data_cleaned.pkl')
