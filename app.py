import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import numpy as np
import gc
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
import math
import pickle
import os
import os.path
import sqlite3
import flask

from flask import Flask, jsonify, request
from lightgbm import LGBMClassifier
from sqlalchemy import create_engine
from sklearn.metrics import roc_auc_score
from scipy.stats import randint as sp_randint
from sklearn.model_selection import KFold, StratifiedKFold
from prettytable import PrettyTable
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.linear_model import SGDClassifier
from collections import Counter
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from bayes_opt import BayesianOptimization
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from datetime import datetime

from hcdr_model import initial_function_definition

if os.path.isfile('pickles/test_data')==False:

    train_data = initial_function_definition.reduce_memory_usage(pd.read_csv('home-credit-default-risk/application_train.csv'))
    test_data = initial_function_definition.reduce_memory_usage(pd.read_csv('home-credit-default-risk/application_test.csv'))
    bureau_data = initial_function_definition.reduce_memory_usage(pd.read_csv('home-credit-default-risk/bureau.csv'))
    bureau_balance = initial_function_definition.reduce_memory_usage(pd.read_csv('home-credit-default-risk/bureau_balance.csv'))

    bureau_data_fe = initial_function_definition.FE_bureau_data_1(bureau_data)

    #One Hot Encoding the Bureau Datasets
    bureau_data, bureau_data_columns = initial_function_definition.one_hot_encode(bureau_data_fe)
    bureau_balance, bureau_balance_columns = initial_function_definition.one_hot_encode(bureau_balance)

    bureau_data_balance_final = initial_function_definition.FE_bureau_data_2(bureau_data, bureau_balance,bureau_data_columns,bureau_balance_columns)

    previous_application = initial_function_definition.reduce_memory_usage(pd.read_csv('home-credit-default-risk/previous_application.csv'))
    previous_application = initial_function_definition.preprocess_previous_application(previous_application)

    pos_cash_balance = initial_function_definition.reduce_memory_usage(pd.read_csv('home-credit-default-risk/POS_CASH_balance.csv'))
    installments_payments = initial_function_definition.reduce_memory_usage(pd.read_csv('home-credit-default-risk/installments_payments.csv'))
    credit_card_balance = initial_function_definition.reduce_memory_usage(pd.read_csv('home-credit-default-risk/credit_card_balance.csv'))


    start = datetime.now()

    test_data = initial_function_definition.fix_nulls_outliers(test_data)
    test_data_temp_1 = initial_function_definition.FE_application_data(test_data)
    bureau_data_balance_final = initial_function_definition.FE_bureau_data_2(bureau_data, bureau_balance,bureau_data_columns,bureau_balance_columns)
    test_data_temp_2 = test_data_temp_1.join(bureau_data_balance_final, how='left', on='SK_ID_CURR')


    test_data_temp_2 = initial_function_definition.FE_previous_application_days_decision(test_data,test_data_temp_2,previous_application)
    test_data_temp_2 = initial_function_definition.FE_pos_cash_balance_months_balance(test_data,test_data_temp_2, pos_cash_balance)
    test_data_temp_2 = initial_function_definition.FE_installments_payments_days_instalment(test_data,test_data_temp_2,installments_payments)
    test_data_temp_2 = initial_function_definition.FE_credit_card_balance_months_balance(test_data,test_data_mod_temp_2,credit_card_balance)

    #Removing any duplicate features, if any are present in the final dataset
    test_data = test_data_temp_2.loc[:,~test_data_temp_2.columns.duplicated()]

    
    print("Time taken to run this cell :", datetime.now() - start)

else:

    test_data = pd.read_pickle('pickles/test_data')

     
features_top_df_train = pd.read_pickle('pickles/features_top_df_train.pkl')
features_top_df_test = test_data[features_top_df_train.columns]
features_top_df_test['SK_ID_CURR'] = test_data['SK_ID_CURR']
features_top_df_test['TARGET'] = np.nan


app = Flask(__name__)

#home page
@app.route('/', methods = [])
def hello_world():
    return 'Hello World!'


#prediction page
@app.route('/index')
def index():
    return flask.render_template('index.html')


#results page
@app.route('/predict', methods = ['POST'])
def predict():

    conn = sqlite3.connect('Home_Credit_DB_Connection.db')
    sk_id_curr = request.form.to_dict()['SK_ID_CURR']
    sk_id_curr = int(sk_id_curr)

    test_datapoint = pd.read_sql_query(f'SELECT * FROM test_data_feats WHERE SK_ID_CURR == {sk_id_curr}', conn)
    test_datapoint = test_datapoint.replace([None], np.nan)

    with open('lgbm/lgbm_model_500f_3.pickle','rb') as f:
        lgbm_model = pickle.load(f)

    if os.path.isfile('lgbm/lgbm_best_threshold_500f_api.pkl')==False:

        feats = [f for f in features_top_df_train.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
        test_predict = np.zeros(features_top_df_test.shape[0])
        test_predict += lgbm_model.predict_proba(features_top_df_test[feats], num_iteration=lgbm_model.best_iteration_)[:, 1] / 5
    else:

        with open('lgbm/lgbm_test_predict_500f.pkl','rb') as f:
            test_predict = pickle.load(f)

    threshold = 0.3741018248484985

    test_predict_rounded = np.round(test_predict,4)
    predicted_class_label = np.where(test_predict_rounded > threshold, 1, 0)

    select_index = list(np.where(test_data["SK_ID_CURR"] == sk_id_curr)[0]) 
    final_class_label = predicted_class_label[select_index[0]]
    final_test_predict_rounded = test_predict_rounded[select_index[0]]

    if final_class_label == 1:
        prediction = 'The customer with this ID is a Potential Defaulter with a probability of {}'.format(final_test_predict_rounded)
    else:
        prediction = 'The customer with this ID is not a Potential Defaulter with a probability of {}'.format(final_test_predict_rounded)
        predicted_proba = 1 - final_test_predict_rounded

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.debug=True
    app.run(host='0.0.0.0', port=8080)


