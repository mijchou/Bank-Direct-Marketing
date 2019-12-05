import os
os.chdir(r'D:\Users\NT80199\Desktop\Projects\project3\datasets')

# Setups

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
bureau = pd.read_csv('bureau.csv')

## previous applications

from tqdm import tqdm_notebook as tqdm

application = pd.read_csv('train.csv')
previous_application = pd.read_csv('previous_application.csv')

PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['AMT_ANNUITY',
                   'AMT_APPLICATION',
                   'AMT_CREDIT',
                   'AMT_DOWN_PAYMENT',
                   'AMT_GOODS_PRICE',
                   'CNT_PAYMENT',
                   'DAYS_DECISION',
                   'HOUR_APPR_PROCESS_START',
                   'RATE_DOWN_PAYMENT'
                   ]:
        PREVIOUS_APPLICATION_AGGREGATION_RECIPIES.append((select, agg))
PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], PREVIOUS_APPLICATION_AGGREGATION_RECIPIES)]

groupby_aggregate_names = []
for groupby_cols, specs in tqdm(PREVIOUS_APPLICATION_AGGREGATION_RECIPIES):
    group_object = previous_application.groupby(groupby_cols)
    for select, agg in tqdm(specs):
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        application = application.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        groupby_aggregate_names.append(groupby_aggregate_name)

application_agg = application[groupby_aggregate_names + ['TARGET']]
application_agg_corr = abs(application_agg.corr())

#### hand crafted features from previous applications

numbers_of_applications = [1, 3, 5]

features = pd.DataFrame({'SK_ID_CURR': previous_application['SK_ID_CURR'].unique()})
prev_applications_sorted = previous_application.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])

group_object = prev_applications_sorted.groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].nunique().reset_index()
group_object.rename(index=str,
                    columns={'SK_ID_PREV': 'previous_application_number_of_prev_application'},
                    inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

prev_applications_sorted['previous_application_prev_was_approved'] = (
        prev_applications_sorted['NAME_CONTRACT_STATUS'] == 'Approved').astype('int')
group_object = prev_applications_sorted.groupby(by=['SK_ID_CURR'])[
    'previous_application_prev_was_approved'].last().reset_index()
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

prev_applications_sorted['previous_application_prev_was_refused'] = (
        prev_applications_sorted['NAME_CONTRACT_STATUS'] == 'Refused').astype('int')
group_object = prev_applications_sorted.groupby(by=['SK_ID_CURR'])[
    'previous_application_prev_was_refused'].last().reset_index()
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

for number in numbers_of_applications:
    prev_applications_tail = prev_applications_sorted.groupby(by=['SK_ID_CURR']).tail(number)

    group_object = prev_applications_tail.groupby(by=['SK_ID_CURR'])['CNT_PAYMENT'].mean().reset_index()
    group_object.rename(index=str, columns={
        'CNT_PAYMENT': 'previous_application_term_of_last_{}_credits_mean'.format(number)},
                        inplace=True)
    features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

    group_object = prev_applications_tail.groupby(by=['SK_ID_CURR'])['DAYS_DECISION'].mean().reset_index()
    group_object.rename(index=str, columns={
        'DAYS_DECISION': 'previous_application_days_decision_about_last_{}_credits_mean'.format(number)},
                        inplace=True)
    features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

    group_object = prev_applications_tail.groupby(by=['SK_ID_CURR'])['DAYS_FIRST_DRAWING'].mean().reset_index()
    group_object.rename(index=str, columns={
        'DAYS_FIRST_DRAWING': 'previous_application_days_first_drawing_last_{}_credits_mean'.format(number)},
                        inplace=True)
    features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

#train = application












# Bureau 1: number of past loans

num_loan = bureau[['SK_ID_CURR', 'DAYS_CREDIT']].groupby(by = ['SK_ID_CURR'])['DAYS_CREDIT'].count().reset_index().rename(index=str, columns={'DAYS_CREDIT': 'num_loan'})
train = train.merge(num_loan, on = ['SK_ID_CURR'], how = 'left')

# Bureau 2: number of past loan types

num_loantype = bureau[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by = ['SK_ID_CURR'])['CREDIT_TYPE'].nunique().reset_index().rename(index=str, columns={'CREDIT_TYPE': 'num_loantype'})
train = train.merge(num_loantype, on = ['SK_ID_CURR'], how = 'left')

# Bureau 3: average number of each loan type

train['ave_loantype'] = train['num_loan']/train['num_loantype']

# Bureau 4: proportion of active loans

def f(x):
    if x == 'Closed':
        y = 0
    else:
        y = 1    
    return y

bureau['CREDIT_ACTIVE_BINARY'] = bureau.apply(lambda x: f(x.CREDIT_ACTIVE), axis = 1)

### Calculate mean number of loans that are ACTIVE per CUSTOMER 
per_active = bureau.groupby(by = ['SK_ID_CURR'])['CREDIT_ACTIVE_BINARY'].mean().reset_index().rename(index=str, columns={'CREDIT_ACTIVE_BINARY': 'per_active'})
bureau = bureau.merge(per_active, on = ['SK_ID_CURR'], how = 'left')
del bureau['CREDIT_ACTIVE_BINARY']
import gc
gc.collect()

# Bureau 6: 

bureau['CREDIT_ENDDATE_BINARY'] = bureau['DAYS_CREDIT_ENDDATE']

def g(x):
    if x<0:
        y = 0
    else:
        y = 1   
    return y

bureau['CREDIT_ENDDATE_BINARY'] = bureau.apply(lambda x: g(x.DAYS_CREDIT_ENDDATE), axis = 1)
print("New Binary Column calculated")

grp = bureau.groupby(by = ['SK_ID_CURR'])['CREDIT_ENDDATE_BINARY'].mean().reset_index().rename(index=str, columns={'CREDIT_ENDDATE_BINARY': 'CREDIT_ENDDATE_PERCENTAGE'})
bureau = bureau.merge(grp, on = ['SK_ID_CURR'], how = 'left')

del bureau['CREDIT_ENDDATE_BINARY']
gc.collect()










# Pop y and ID

#train['TARGET'] = abs(train['TARGET']-1)
y = train.pop('TARGET')
cus_id = train.pop('SK_ID_CURR')

# Drop features not needed

del train['FLAG_MOBIL']
del train['FLAG_DOCUMENT_2']
del train['FLAG_DOCUMENT_7']
del train['FLAG_DOCUMENT_10']
del train['FLAG_DOCUMENT_12']
del train['REGION_RATING_CLIENT_W_CITY']
del train['LIVINGAPARTMENTS_AVG']
del train['LIVINGAREA_AVG']
del train['YEARS_BEGINEXPLUATATION_AVG']
del train['OBS_60_CNT_SOCIAL_CIRCLE']
del train['DEF_60_CNT_SOCIAL_CIRCLE']

mod_df = train.filter(regex = 'MODE')
med_df = train.filter(regex = 'MEDI')
train = train.drop(columns = mod_df)
train = train.drop(columns = med_df)

# Process address not match

notmatch = train.filter(regex='NOT')
train = train.drop(columns=notmatch)

notmatch_city = notmatch.filter(regex='CITY').sum(axis=1).tolist()
notmatch_region = notmatch.filter(regex='REGION').sum(axis=1).tolist()
notmatch.LIVE_REGION_NOT_WORK_REGION = abs(notmatch.LIVE_REGION_NOT_WORK_REGION - 1) #important
notmatch_all = notmatch.sum(axis=1).tolist()
train['notmatch_city'] = notmatch_city
train['notmatch_regions'] = notmatch_region
train['notmatch_all'] = notmatch_all ### GOOD!

# Process documents recieved

documents = train.filter(regex='FLAG_DOCUMENT')
train = train.drop(columns = documents)

documents.FLAG_DOCUMENT_3 = abs(documents.FLAG_DOCUMENT_3 - 1)
documents.FLAG_DOCUMENT_13 = abs(documents.FLAG_DOCUMENT_13 - 1)
documents.FLAG_DOCUMENT_16 = abs(documents.FLAG_DOCUMENT_16 - 1)
documents.FLAG_DOCUMENT_18 = abs(documents.FLAG_DOCUMENT_18 - 1)
documents.FLAG_DOCUMENT_19 = abs(documents.FLAG_DOCUMENT_19 - 1)

doc_number = documents.sum(axis=1).tolist()
train['doc_number'] = doc_number

# How much more money did the client borrow than the actual needs for house?

excess_amt = train.AMT_CREDIT - train.AMT_GOODS_PRICE
excess = excess_amt > 0
train['excess_amt'] = excess_amt
train['excess'] = excess

del train['AMT_GOODS_PRICE']
del train['AMT_CREDIT']

# Missing house informations

house_avg = train.filter(regex='AVG')

train['missing_apartments'] = train.APARTMENTS_AVG.isnull().astype(int)
train['missing_basementarea'] = train.BASEMENTAREA_AVG.isnull().astype(int)
train['missing_yearsbuild'] = train.YEARS_BUILD_AVG.isnull().astype(int)
train['missing_commonarea'] = train.COMMONAREA_AVG.isnull().astype(int)
train['missing_elevators'] = train.ELEVATORS_AVG.isnull().astype(int)
train['missing_entrances'] = train.ENTRANCES_AVG.isnull().astype(int)
train['missing_floorsmax'] = train.FLOORSMAX_AVG.isnull().astype(int)
train['missing_floorsmin'] = train.FLOORSMIN_AVG.isnull().astype(int)
train['missing_landarea'] = train.LANDAREA_AVG.isnull().astype(int)
train['missing_nonlivingapartments'] = train.NONLIVINGAPARTMENTS_AVG.isnull().astype(int)
train['missing_nonlivingarea'] = train.NONLIVINGAREA_AVG.isnull().astype(int)

train = train.drop(columns=house_avg)

# Missing External resources

from sklearn.impute import SimpleImputer

# The rest

num2 = train[['AMT_ANNUITY', 'CNT_FAM_MEMBERS', 'DAYS_LAST_PHONE_CHANGE',]]
imp = SimpleImputer(strategy='median')
num_imp = pd.DataFrame(imp.fit_transform(num2), columns = ['amt_ann', 'cnt_fam', 'days_phone_change'])
train = train.drop(columns=num2)
train = pd.concat([train, num_imp], axis=1)

# Missing observed social circle

obs = train[['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE']]
imp = SimpleImputer(strategy='median')
obs_imp = pd.DataFrame(imp.fit_transform(obs), columns = ['obs30', 'def30'])
train = train.drop(columns=obs)
train = pd.concat([train, obs_imp], axis=1)

#  Missing categorical variables

accomp = train[['NAME_TYPE_SUITE', 'OCCUPATION_TYPE']]
imp = SimpleImputer(strategy='most_frequent')
accomp_imp = pd.DataFrame(imp.fit_transform(accomp), columns = ['accomp', 'occup'])
train = train.drop(columns=accomp)
train = pd.concat([train, accomp_imp], axis=1)

# Missing requested number

req = train[['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
             'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
             'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']]
imp = SimpleImputer(strategy='most_frequent')
req_imp = pd.DataFrame(imp.fit_transform(req), columns = ['hour', 'day',
                                                          'week', 'mon',
                                                          'qrt', 'year'])
train = train.drop(columns=req)
train = pd.concat([train, req_imp], axis=1)

# Missing excessive amount and car age

num = train[['OWN_CAR_AGE', 'excess_amt']]
imp = SimpleImputer(strategy='median')
num_imp = pd.DataFrame(imp.fit_transform(num), columns = ['car_age', 'excess_amt'])
train = train.drop(columns=num)
train = pd.concat([train, num_imp], axis=1)

# Label Encoding (for trees)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
cat = train[['NAME_CONTRACT_TYPE', 'CODE_GENDER',
                             'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                             'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
                             'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                             'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
                             'accomp', 'occup']]
train = train.drop(columns=cat)
cat_encoded = cat.apply(LabelEncoder().fit_transform)
train = pd.concat([train, cat_encoded], axis=1)

# Missing past load informations

num = train[['num_loan', 'num_loantype', 'ave_loantype']]
imp = SimpleImputer(fill_value=0)
num_imp = pd.DataFrame(imp.fit_transform(num), columns = ['num_loan_done', 'num_loantype_done', 'ave_loantype_done'])
train = train.drop(columns=num)
train = pd.concat([train, num_imp], axis=1)

# Missing External resources

## e2

e2_imp_set = train[['REGION_POPULATION_RELATIVE', 'REGION_RATING_CLIENT',
                    'HOUR_APPR_PROCESS_START', 'missing_apartments',
                    'missing_elevators', 'missing_entrances',
                    'missing_floorsmax', 'amt_ann', 'days_phone_change']]

e1 = train.pop('EXT_SOURCE_1')
e2 = train.pop('EXT_SOURCE_2')
e3 = train.pop('EXT_SOURCE_3')

e2_complete = e2_imp_set[e2.isnull()==False]

e2_missing_y = e2[e2.isnull()==True]
e2_complete_y = e2[e2.isnull()==False]

from sklearn.linear_model import LinearRegression

e2_imputed = list()
regressor = LinearRegression()
regressor.fit(e2_complete, e2_complete_y)
e2_missing_pred = pd.Series(regressor.predict(e2_imp_set))

train['e2'] = e2.mask(e2.isna(), e2_missing_pred)

## e1
train['EXT_SOURCE_1'] = e1

e1_imp_set = train[['CNT_CHILDREN', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION',
                    'DAYS_ID_PUBLISH', 'FLAG_EMP_PHONE', 'notmatch_city',
                    'notmatch_all', 'days_phone_change', 'CODE_GENDER', 'e2',
                    'NAME_INCOME_TYPE', 'NAME_HOUSING_TYPE', 'ORGANIZATION_TYPE']]

e1 = train.pop('EXT_SOURCE_1')

e1_complete = e1_imp_set[e1.isnull()==False]

e1_missing_y = e1[e1.isnull()==True]
e1_complete_y = e1[e1.isnull()==False]

e1_imputed = list()
regressor = LinearRegression()
regressor.fit(e1_complete, e1_complete_y)
e1_missing_pred = pd.Series(regressor.predict(e1_imp_set))
train['e1'] = e1.mask(e1.isna(), e1_missing_pred)

## e3

train['EXT_SOURCE_3'] = e3

e3_imp_set = train[['REGION_POPULATION_RELATIVE', 'HOUR_APPR_PROCESS_START',
                    'missing_apartments', 'missing_elevators',
                    'missing_entrances', 'missing_floorsmax', 'amt_ann',
                    'days_phone_change', 'e1']]

e3 = train.pop('EXT_SOURCE_3')

e3_complete = e3_imp_set[e3.isnull()==False]

e3_missing_y = e3[e3.isnull()==True]
e3_complete_y = e3[e3.isnull()==False]

e3_imputed = list()
regressor = LinearRegression()
regressor.fit(e3_complete, e3_complete_y)
e3_missing_pred = pd.Series(regressor.predict(e3_imp_set))
train['e3'] = e3.mask(e3.isna(), e3_missing_pred)

### previous applications data imputations

num3 = train.iloc[:, 14:59]
imp = SimpleImputer(strategy='median')
num_imp = pd.DataFrame(imp.fit_transform(num3))
train = train.drop(columns=num3)
train = pd.concat([train, num_imp], axis=1)


####### train191126 written here



## train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
            train, y, test_size=0.2, random_state=0)

## Scaling
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

## Data balancing: Over-sampling
train = pd.concat([X_train, y_train], axis=1)
fail = train[train.TARGET==0]
success = train[train.TARGET==1]
success_upsampled = resample(success,
                            replace=True,
                            n_samples=len(fail),
                            random_state=0)
upsampled = pd.concat([fail, success_upsampled])
y_train = upsampled.TARGET
X_train = upsampled.drop('TARGET', axis=1)



# Modeling: bagging trees

from sklearn.ensemble import BaggingClassifier

classifier = BaggingClassifier(random_state=123,
                               n_estimators=1000,
                               max_samples=120,
                               n_jobs=-1,
                               verbose=True)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
confusion_matrix(y_test, y_pred)

print(round(accuracy_score(y_test, y_pred), 2)*100)
print(classification_report(y_test, y_pred))
y_score = classifier.predict_proba(X_test)
print(roc_auc_score(y_test, y_score[:, 1]))

### feature importance

y = train.pop('TARGET')
feature_names = list(train.columns)
feature_importance_values = classifier.feature_importances_
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
feature_importances = feature_importances.sort_values('importance', ascending = False).reset_index()

# Modelling: boosting trees

from sklearn.ensemble import GradientBoostingClassifier

classifier = GradientBoostingClassifier(random_state=0,
                                        n_estimators=800,
                                        subsample=0.8)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
confusion_matrix(y_test, y_pred)

print(round(accuracy_score(y_test, y_pred), 2)*100)
print(classification_report(y_test, y_pred))
y_score = classifier.predict_proba(X_test)
print(roc_auc_score(y_test, y_score[:, 1]))

######## Comparison between logistic models wtih standardised/instandardised models

top_features= feature_importances.feature[0:20]
train = train[top_features]

# Modeling: logistic - not standardised

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)

confusion_matrix(y_test, y_pred)
print(round(accuracy_score(y_test, y_pred), 2)*100)
print(classification_report(y_test, y_pred))
y_score = classifier.predict_proba(X_test)
print(roc_auc_score(y_test, y_score[:, 1]))

# Modeling: logistic - standardised

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train_scaled, y_train)
y_pred=classifier.predict(X_test_scaled)
confusion_matrix(y_test, y_pred)
print(round(accuracy_score(y_test, y_pred), 2)*100)
print(classification_report(y_test, y_pred))
y_score = classifier.predict_proba(X_test_scaled)
print(roc_auc_score(y_test, y_score[:, 1]))

########################################################################

# Modelling: SVM

from sklearn.svm import SVC

classifier = SVC(random_state=0, kernel='linear')
classifier.fit(X_train_scaled, y_train)
y_pred=classifier.predict(X_test_scaled)
confusion_matrix(y_test, y_pred)
print(round(accuracy_score(y_test, y_pred), 2)*100)
print(classification_report(y_test, y_pred))

# Modelling: Random Forest

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
confusion_matrix(y_test, y_pred)
print(round(accuracy_score(y_test, y_pred), 2)*100)
print(classification_report(y_test, y_pred))

y_score = classifier.predict_proba(X_test)
print(roc_auc_score(y_test, y_score[:, 1]))

# Extract feature importances

y = train.pop('TARGET')
feature_names = list(train.columns)
feature_importance_values = classifier.feature_importances_
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
feature_importances = feature_importances.sort_values('importance', ascending = False).reset_index()

# only keep the first 30 important features

train_new = train[feature_importances.feature[0:30]]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
            train_new, y, test_size=0.2, random_state=0)

from sklearn.utils import resample

train = pd.concat([X_train, y_train], axis=1)
fail = train[train.TARGET==0]
success = train[train.TARGET==1]
success_upsampled = resample(success,
                            replace=True,
                            n_samples=len(fail),
                            random_state=0)
upsampled = pd.concat([fail, success_upsampled])
y_train = upsampled.TARGET
X_train = upsampled.drop('TARGET', axis=1)

classifier = BaggingClassifier(random_state=123,
                               n_estimators=1000,
                               max_samples=120,
                               n_jobs=-1,
                               verbose=True)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
confusion_matrix(y_test, y_pred)

print(round(accuracy_score(y_test, y_pred), 2)*100)
print(classification_report(y_test, y_pred))
y_score = classifier.predict_proba(X_test)
print(roc_auc_score(y_test, y_score[:, 1]))

X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train_scaled, y_train)
y_pred=classifier.predict(X_test_scaled)
confusion_matrix(y_test, y_pred)
print(round(accuracy_score(y_test, y_pred), 2)*100)
print(classification_report(y_test, y_pred))
y_score = classifier.predict_proba(X_test_scaled)
print(roc_auc_score(y_test, y_score[:, 1]))

flat_list = [item for sublist in y_score for item in sublist][slice(1, 9601, 2)]
#flat_list = [item for sublist in y_score for item in sublist][slice(1, 12001, 2)]

precision = list()
recall = list()
f1 = list()

for j in np.arange(0, 1, 0.01):
    y_pred = list()
    for i in range(0, 4800):
  #  for i in range(0, 6000):
        if flat_list[i] > j:  
            y_pred.append(1)
        else: y_pred.append(0)

#confusion_matrix(y_test, y_pred)
#print(round(accuracy_score(y_test, y_pred), 2)*100)
#print(classification_report(y_test, y_pred))

    report = classification_report(y_test, y_pred, output_dict=True)
    recall.append(report['1']['recall'])
    precision.append(report['1']['precision'])
    f1.append(report['1']['f1-score'])

import matplotlib.pyplot as plt

# Data
df = pd.DataFrame({'x': np.arange(0, 1, 0.01), 'precision': precision, 'recall': recall, 'f1': f1 })

# multiple line plot
plt.plot( 'x', 'precision', data=df, marker='', color='blue', linewidth=2)
plt.plot( 'x', 'recall', data=df, marker='', color='green', linewidth=2)
plt.plot( 'x', 'f1', data=df, marker='', color='olive', linewidth=2)
plt.legend()














################################################


import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
#from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('bureau.csv', nrows = num_rows)
    bb = pd.read_csv('bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg









