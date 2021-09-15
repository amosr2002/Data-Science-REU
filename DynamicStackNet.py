#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 13:37:10 2021

@author: anneliese.mm
"""


import pandas as pd
import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import tensorflow as tf
import pandas as pd
import statsmodels as sm
import numpy as np
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import math
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from train_test import X_train, X_test, outcomes_train, outcomes_test
from sklearn.model_selection import RandomizedSearchCV
from xgboost.sklearn import XGBRegressor
from sklearn.base import clone


# # Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# # Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
bootstrap = [True, False]

df = pd.read_csv('MasterIntersection.csv')
#df = pd.read_csv('Masterintersection_v2.csv')
df = df.drop(['Unnamed: 0'], axis=1)

vars = df['Somatic_Complaints_TScore'][df['Somatic_Complaints_TScore'] != 'None']
#vars = df['Somatic_Complaints_TScore'][df['Somatic_Complaints_TScore'].notna()]
#vars = df['Somatic_Problems_TScore'][df['Somatic_Problems_TScore'] != 'None'].loc(axis = 0)[range(10)]
x_train = X_train[vars]
y_train = outcomes_train['Somatic_Complaints_TScore']

from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import BayesianRidge
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, DotProduct
rf = RandomForestRegressor()
random_grid = {
               'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'n_estimators':n_estimators,
              'max_features':max_features,
              'min_samples_leaf':min_samples_leaf,
              'bootstrap':bootstrap
               
               }
models = ['BayesianRidge', 'DecisionTreeRegressor', 'ExtraTreesRegressor', 'GaussianProcessRegressor', 'KNN',   'Randomforest', 'XGBRegressor']


model_names = []
model_actual = []
errors = {key: [] for key in models}
kfold_data = {key: [] for key in models}
trained_models = {key: [] for key in models}
#trained_models = {key: [] for key in models}

kf = KFold(n_splits=5)
algos = {
    

       'BayesianRidge': {
        'model':BayesianRidge(),
        'params':{'alpha_1': 10.0**np.arange(-6,1,2),
              'alpha_2': 10.0**np.arange(-6,1,2),
              'lambda_1':10.0**np.arange(-6,1,2),
              'lambda_2':10.0**np.arange(-6,1,2)}
    # },
    }, 
   
      'DecisionTreeRegressor' :{
        'model':DecisionTreeRegressor(random_state=123),
        'params':{'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'max_features':max_features,
              'min_samples_leaf':min_samples_leaf}
    }, 
    'ExtraTreesRegressor':{
        'model':ExtraTreesRegressor(random_state=123),
        'params': random_grid
    },
    'GaussianProcessRegressor': {
        'model':GaussianProcessRegressor(random_state=123),
        'params': {'kernel': [ConstantKernel(), RBF(), DotProduct()]}
    }
  ,
  'KNN': {
        'model': KNeighborsRegressor(),
        'params': {'n_neighbors': np.arange(1, 12, 2),
          'weights': ['uniform', 'distance']
            }
        },
    
    
  # 'SVM': {
  #       'model': SVR(),
  #       'params': {
  #         'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
  #         'C' : [1,5,10],
  #         'degree' : [3,8], 'coef0' : [0.01,10,0.5],'gamma' : ['auto','scale']}
  #           },
         
  'Randomforest' : {
        'model': RandomForestRegressor(),
        'params': random_grid
        },
  
  'XGBRegressor': {
      'model' : XGBRegressor(),
      'params':{'nthread':[4], #when use hyperthread, xgboost may become slower
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'reg_alpha': [1.1, 1.2, 1.3],
    'reg_lambda': [1.1, 1.2, 1.3],
    'n_estimators': [400, 700, 1000]}
      }
    
    }
    

def find_best_model_using_gridsearchcv(X,y,algos):
    #Finds the best parameters for each model and generates an array of gridsearch dataframes
    #Adds fitted models to a dictionary
  
    arr1 = []
    arr2 = []
    pd_arr = []
    model_names= []
    
    for algo_name, config in algos.items():
        #gs =  GridSearchCV(config['model'], config['params'], scoring='neg_root_mean_squared_error', cv=kf, return_train_score=False ,n_jobs=-1)
        print(config['model'])
        gs = RandomizedSearchCV(estimator=config['model'], param_distributions=config['params'], scoring='r2',refit=False, cv=kf, return_train_score=False)
        gs.fit(X,y)
        df1 = pd.DataFrame(gs.cv_results_)
        pd_arr.append(df1)
        print("We just appended the df respective to its model")
        
        model_names.append(algo_name)
        #Adding the model names to its respective data frame
        for model in range(len(pd_arr)):
          pd_arr[model]['model'] = model_names[model]

        for train,test in gs.cv.split(X, y):
          arr1.append(train)
          arr2.append(test)
    train_array=[]

    print("about to add train/test")
    #Adding train/test data to the dataframes
    for train1 in range(gs.cv.get_n_splits(X)):
      for dataframe in range(len(pd_arr)):
        t=[]
        for row in range(pd_arr[dataframe].shape[0]):
          t.append(arr1[train1])
        pd_arr[dataframe][f"train, {train1}"] = t
    
    for train2 in range(gs.cv.get_n_splits(X)):
      for dataframe in range(len(pd_arr)):
        t=[]
        for row in range(pd_arr[dataframe].shape[0]):
          t.append(arr2[train2])
        pd_arr[dataframe][f"test, {train2}"] = t
    
    #Adding trained models to the trained_models dictionary
    for arr in range(len(pd_arr)):
        
        df = pd_arr[arr]
        params = df[df['rank_test_score'] == 1]['params'].values[0]
        data = df[df['rank_test_score'] == 1]['train, {}'.format(splitno(pd_arr, arr))].values[0]
        
        #temp.append(algo)
        trained_models[list(algos.keys())[arr]].append(clone(algos.get(list(algos.keys())[arr]).get('model')).set_params(**params).fit(X.iloc[list(data)] , y.iloc[list(data)]))

        
    
      
    return trained_models, pd_arr

import numpy as np 
def metric(tunedata, i, xtrain, ytrain):
    """Gets the metric from the model's dataframe
    """

    frame = tunedata[i]
    max = -1*np.inf
    for x1 in range(kf.get_n_splits(xtrain)):
        if((frame[frame['rank_test_score'] == 1]["split{}_test_score".format(x1)].values > max)).any():
            max = frame[frame['rank_test_score'] == 1]["split{}_test_score".format(x1)].values[0]
    #print("i: ", i)
    #print("max: ", max)
    return max

def splitno(tunedata, i):
  #Gets the split number of the best score corresponding to the best rank
    frame = tunedata[i]
    max = -1*np.inf
    no = 0
    for x1 in range(kf.get_n_splits(x_train)):
        #print(frame[frame['rank_test_score'] == 1]['rank_test_score'])
        if((frame[frame['rank_test_score'] == 1]["split{}_test_score".format(x1)].values > max)).any():
            no = x1
            max = frame[frame['rank_test_score'] == 1]["split{}_test_score".format(x1)].values[0]

    return no



def compute_error(trained_models, tunedata, models):
    """ calculate errors for list of pre-trained models and test data"""
    for i in range(len(models)):
        #print("index", i)
        #trained = list(trained_models.keys())[i]
        #pred = get_pred(trained, kfold_data)
        
        
        #gotta fix metric first
        #print("blah: ", errors[models[i]])
        errors[models[i]].append(metric(tunedata, i, x_train, y_train))
        #print("error:", errors[list(errors.keys())[i]].append(metric(tunedata, i, x_train, y_train)))

    print("errors: ", errors)
    return(errors)

def get_hold(tunedata, i):
    #print("orig x_train: ", x_train.shape)

    xtrain = x_train.iloc[tunedata[i][tunedata[i]['rank_test_score'] == 1]['train, {}'.format(splitno(tunedata,i))].values[0]]
    ytrain = y_train.iloc[tunedata[i][tunedata[i]['rank_test_score'] == 1]['train, {}'.format(splitno(tunedata,i))].values[0]]
    xtest = x_train.iloc[tunedata[i][tunedata[i]['rank_test_score'] == 1]['test, {}'.format(splitno(tunedata,i))].values[0]]
    #print("xtest shape: ", xtest.shape)
    ytest = y_train.iloc[tunedata[i][tunedata[i]['rank_test_score'] == 1]['test, {}'.format(splitno(tunedata,i))].values[0]]

    return(xtrain,ytrain,xtest,ytest)

def get_pred(model, kfold_data):
    """Return predictions for list of pre-trained models and test data
    """
    xhold = kfold_data.get(model)[0][2]
    xhold = xhold.values
    print(xhold)
    print(xhold.shape)
    yhold = kfold_data.get(model)[0][3]

    print("trained_models: ", trained_models)

    trained = trained_models.get(model)[0]
    print("trained: ", trained)
    pred = trained.predict(xhold)
    return(pred, yhold)

def get_pred_2(model, xdata):
    """Return predictions for list of pre-trained models and test data
    """
    #print("model" , model)
    trained = trained_models.get(model)[-1]
    pred = trained.predict(xdata)
    return(pred)

def fill_kfoldMatrix(tunedata):
    #print( "len keys: ", kfold_data.keys())
    #print( "len keys: ", len(kfold_data.keys()))
    #print("tunedata: ", tunedata)
    for i in range(len(kfold_data.keys())):
        #print("i: ", i)
        xtrain,ytrain,xtest,ytest = get_hold(tunedata, i)
        array = [xtrain,ytrain,xtest,ytest]
        kfold_data[list(kfold_data.keys())[i]].append(array)
    
    return(kfold_data)
        

def top_half(errors):
    """Returns upper half of accurate models
    Args:
        model_acc (dictionary)
    """
    xc = errors.copy()
    for x in xc:
      num = xc[x][-1]
      xc[x] = num
    sort_errors = dict(sorted(xc.items(), key=lambda x: x[1], reverse=True))
    #errors = sort_errors
    tophalf = list(sort_errors.keys())[:math.floor(len(sort_errors.keys())/2)]
    
    wins = {key: None for key in tophalf}
    
    return(tophalf, wins)

    pass

def num_wins(wins):
    """ counts number of Trues from compare_error() """
    count = 0
    for i in range(len(wins.keys())):
        if wins[list(wins.keys())[i]]:
            count += 1

    win_num = count
    return(win_num)
    pass

def done(win_num):
    if win_num <=2:
        return(True)
    else:
        return(False)
    """Checks if number of longest error value lists <= 2
    """
    pass

def compare_error(tophalf, errors, wins, k):
    """ compares two most recent error values for longest lists"""
    for i in range(len(tophalf)):
        #k=2 normally
        compare = errors[tophalf[i]][len(errors[tophalf[i]]) - 1] > errors[tophalf[i]][len(errors[tophalf[i]]) - k]
        wins[tophalf[i]] = compare
    return(wins)
    pass

def move_back(tophalf, errors, wins, xtrain_copy, ytrain_copy, algos_copy):
    """ Deletes most recent error, most recent trained model from top half models if failed compare error
    """
    retune_names = tophalf.copy()
    #print("retune names: ", retune_names)
    
    back_count = 0
    for i in range(len(tophalf)):
      print("i: ", i)
      if not (wins[tophalf[i]]):
        #delete most recent error
        errors[tophalf[i]].pop(-1)
        #deletes most recent trained model
        trained_models[tophalf[i]].pop(-1)
        #update model
        retune_names.pop(i)
        back_count += 1

        #remove from tophalf
        tophalf.remove(tophalf[i])
        #print("back count: ", back_count)
    #print("retune names", retune_names)
    if (len(retune_names)> 0 and back_count > 0):
      retune_models(retune_names, algos_copy, xtrain_copy, ytrain_copy)

    return(errors, trained_models)
    pass



def retune_models(remodels, algos, xtrain, ytrain ):
    """
    Re- tunes remaining models in outer layer after done_moving_back returns true
    """
    algos_copy = algos.copy()
    model_actual = []
    xtrain_copy = xtrain.copy()
    ytrain_copy = ytrain.copy()

    #print(algos_copy.keys())
    for model_name in (list(algos_copy.keys())):
        
        #print("model_name", model_name)
        if(model_name not in remodels):
            algos_copy.pop(model_name)
            column_name = 'pred_' + model_name 
            xtrain_copy[column_name] = get_pred_2(model_name, xtrain)


            #ytrain_copy[column_name] = ytrain
            
    #values = xtrain_copy.values

    #min_max_scaler = preprocessing.MinMaxScaler()
    #x_scaled = min_max_scaler.fit_transform(values)
    #xtrain_copy = pd.DataFrame(x_scaled, columns=xtrain_copy.columns)
    trained_models, pd_arr = find_best_model_using_gridsearchcv(xtrain_copy, ytrain_copy, algos_copy)
    return(trained_models, pd_arr,xtrain_copy, ytrain_copy, algos_copy)

n = 0

def dynamic_stacknet(x_train, y_train, algos, n, errors):
    if n == 0:
      trained_models, pd_arr = find_best_model_using_gridsearchcv(x_train, y_train, algos)
      #print(trained_models)
      #kfold_data = fill_kfoldMatrix(pd_arr)
      errors = compute_error(trained_models, pd_arr, list(algos.keys()))

    tophalf, wins = top_half(errors)
    print(tophalf)
    
    trained_models, pd_arr,xtrain_copy, ytrain_copy, algos_copy = retune_models(tophalf, algos, x_train, y_train)
    errors = compute_error(trained_models, pd_arr, tophalf)
    
    wins = compare_error(tophalf, errors, wins, 2)
    print(wins)
    print(errors)
  
    errors, trained_models = move_back(tophalf, errors, wins, xtrain_copy, ytrain_copy, algos)
    win_num = num_wins(wins)

    done_result = done(win_num)
    print(win_num)
    if not done_result:
        #what is the xtrain and ytrain for the recursive call? 
        print("recursive time")
        n+= 1
        errors_copy = { your_key: errors[your_key] for your_key in tophalf }
        dynamic_stacknet(xtrain_copy, ytrain_copy, algos_copy, n, errors_copy)
    return(trained_models, xtrain_copy)

    
import numpy as np

class WeightEnsembleModels:
    def __init__(self, models, Xtrain, Ytrain, X):
        self.__models = models
        self.__X = X
        self.__Xtrain = Xtrain
        self.__Ytrain = Ytrain

    def getModels(self):
        return self.__models

    def setModels(self, models):
        self.__models = models

    def getX(self):
        return self.__X

    def setX(self, X):
        self.__X = X

    def getXtrain(self):
        return self.__Xtrain

    def getYtrain(self):
        return self.__Ytrain

    def compute_weight(self):
        base_models = np.column_stack([model.predict(self.getXtrain()) for model in self.getModels()])

        # Convex Optimization Problem --> minimize (Ytest - base_models*weight)**2
        weight = cp.Variable(len(self.getModels()))
        objective = cp.Minimize(cp.sum_squares((base_models * weight) - self.getYtrain()))
        constraints = [weight >= 0, sum(weight) == 1]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        return weight.value

    def predict(self):
        base_models_X = np.column_stack([model.predict(self.getX()) for model in self.getModels()])
        prediction = np.dot(base_models_X, self.compute_weight())
        return prediction
    
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score 

trained_models, xtrain_copy = dynamic_stacknet(x_train, y_train, algos, n , errors)

def get_pred_3(model, xdata):
    """Return predictions for list of pre-trained models and test data
    """
    #print("model" , model)
    pred = model.predict(xdata)
    return(pred)

def stackresults_average(trained_models, x_test, y_test):
  finalpred_arr = []
  #sort trained dictionary by array length
  train_copy = trained_models.copy()
  xcopy = x_test.copy()
  list1 = []
  for x in train_copy:
    tm = {}
    tm[len(train_copy[x])]=[train_copy[x][-1], x]
    list1.append(tm)
  
  print(len(xcopy.columns))
  

  #create new dictionary split into key: length of array, value : most recent array for all arrays of that length
  
  
  res = {}
  for dicts in list1:
    for lists in dicts:
      if lists in res:
        res[lists] += (dicts[lists])
      else:
        res[lists] = dicts[lists]

  res = dict(sorted(res.items(), key = lambda x:x[0], reverse = False))
  
  print(res)
  #work thru keys of dictionary append to xtrain, make predictions
  for key, value in res.items():
    print("key", key)
    count = key

    print("len value",len(value))

    xcopy_2 = xcopy.copy()
    print("xcopy", len(xcopy.columns))
    print("xcopy_2", len(xcopy.columns))
    for k in [*range(len(value))][::2]:
      print("value[k]", value[k+1])
      temp = get_pred_3(value[k], xcopy)
      #print("temp", temp)
      column_name = 'pred_' + value[k+1]
      xcopy_2[column_name] = temp
    
    if key != max(list(res.keys())):
      xcopy = xcopy_2
      print("xcopy", len(xcopy.columns))
  #values = xcopy.values
  #min_max_scaler = preprocessing.MinMaxScaler()
  #x_scaled = min_max_scaler.fit_transform(values)
  #xcopy = pd.DataFrame(x_scaled, columns=xcopy.columns)
  final_models = res[list(res.keys())[-1]][::2]
  print("final models", final_models)
  print("len final models", len(final_models) )
  final_pred = 0
  #weight = WeightEnsembleModels(final_models, x_train, y_train, x_test)
  
  if len(final_models) == 1:
    final_pred = get_pred_3(final_models[0],xcopy)
  else:
    pred_vec = []
    for i in range(len(final_models)):
      pred_vec.append(get_pred_3(final_models[i],xcopy))
    multiplied_list = [element * (1/len(final_models)) for element in pred_vec]
    final_pred = list(map(sum, zip(*multiplied_list)))
    print(list(final_pred))
  
  final_accuracy_mse = (mean_squared_error(y_test, final_pred))**0.5
  final_accuracy_r2 = 1 - ( 1-r2_score(y_test, final_pred) ) * ( len(y_test) - 1 ) / ( len(y_test) - x_test.shape[1] - 1 )


  return(final_pred, final_accuracy_mse, final_accuracy_r2)
  




def stackresults_convex(trained_models, x_test, y_test):
  finalpred_arr = []
  #sort trained dictionary by array length
  train_copy = trained_models.copy()
  xcopy = x_test.copy()
  list1 = []
  for x in train_copy:
    tm = {}
    tm[len(train_copy[x])]=[train_copy[x][-1], x]
    list1.append(tm)
  
  print(len(xcopy.columns))
  

  #create new dictionary split into key: length of array, value : most recent array for all arrays of that length
  
  
  res = {}
  for dicts in list1:
    for lists in dicts:
      if lists in res:
        res[lists] += (dicts[lists])
      else:
        res[lists] = dicts[lists]

  res = dict(sorted(res.items(), key = lambda x:x[0], reverse = False))
  
  print(res)
  #work thru keys of dictionary append to xtrain, make predictions
  for key, value in res.items():
    print("key", key)
    count = key

    print("len value",len(value))

    xcopy_2 = xcopy.copy()
    print("xcopy", len(xcopy.columns))
    print("xcopy_2", len(xcopy.columns))
    for k in [*range(len(value))][::2]:
      print("value[k]", value[k+1])
      temp = get_pred_3(value[k], xcopy)
      #print("temp", temp)
      column_name = 'pred_' + value[k+1]
      xcopy_2[column_name] = temp
    
    if key != max(list(res.keys())):
      xcopy = xcopy_2
      print("xcopy", len(xcopy.columns))
  
  #values = xcopy.values
  #min_max_scaler = preprocessing.MinMaxScaler()
  #x_scaled = min_max_scaler.fit_transform(values)
  #xcopy = pd.DataFrame(x_scaled, columns=xcopy.columns)
  final_models = res[list(res.keys())[-1]][::2]
  print("final models", final_models)
  print("len final models", len(final_models) )
  final_pred = 0
  #weight = WeightEnsembleModels(final_models, x_train, y_train, x_test)
  
  if len(final_models) == 1:
    final_pred = get_pred_3(final_models[0],xcopy)
 
  else:
    final_models1 = []
    #weight = WeightEnsembleModels(final_models1, x_train, y_train, x_test)
    weight_ensemble_learner = WeightEnsembleModels(models=final_models, Xtrain=xtrain_copy, Ytrain=y_train, X=xcopy) #Xtest is 30% Testing Dataset
    print(xcopy.shape[0])
    print(xcopy.shape[1])
    weight = weight_ensemble_learner.compute_weight() # Compute the optimal weight using the training dataset (70%)
    print(weight)
    Y_weight_pred = weight_ensemble_learner.predict() # Compute the predicted Y on Xtest (30%)

  weight_learn_rmse = (mean_squared_error( y_test, Y_weight_pred))**0.5 # Compute RMSE

  weight_learn_adjust_R2 = 1 - ( 1-r2_score(y_test, Y_weight_pred) ) * ( len(y_test) - 1 ) / ( len(y_test) - xcopy.shape[1] - 1 ) # Compute Adjusted R2
  
  #final_accuracy_mse = (mean_squared_error(y_test, final_pred))**0.5
  #final_accuracy_r2 = 1 - ( 1-r2_score(y_test, final_pred) ) * ( len(y_test) - 1 ) / ( len(y_test) - x_test.shape[1] - 1 )


  return(Y_weight_pred, weight_learn_rmse, weight_learn_adjust_R2)


  