# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:11:52 2021

@author: balle
"""


import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from train_test import X_train, X_test, outcomes_train, outcomes_test
import tensorflow as tf
import pandas as pd
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Activation, Dense
import math


def build_model(x):
    
    model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(x,)),
    layers.Dense(2/3*(x+1), activation=tf.nn.relu),
    tf.keras.layers.Dense(1)])
    
#     model = Sequential()

# # Adding the input layer and the first hidden layer
#     model.add(Dense(math.floor(2/3*(x+1)), activation = 'relu', input_dim = x))
    
# # Adding the second hidden layer
#     model.add(Dense(units = 1))
    
    
    #optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    # model = keras.Sequential([layers.Dense(math.floor((2/3)*x+1), activation=tf.nn.relu, input_shape=(x,)), 
                               
    #                             layers.Dense(1)])
    
   #optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer='adam', metrics=['mae','mse'])
    return model



class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 ==0: print('')
        print('.', end='')
        
EPOCHS = 140

def forward_selection(data, target, xtest, ytest, significance_level=0.01):
    initial_features = data.columns.tolist()
    best_features = []
    mse1 = tf.keras.losses.MeanSquaredError()
    mse_min = 0
    min_mse_value =0
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        mse = pd.Series(index=remaining_features)
        #print(remaining_features)
        
        old_mse = min_mse_value
        for new_column in remaining_features:
            
               
            #print(mse_min)
            model = build_model(len(best_features+[new_column]))
            #print(len(new_column))
            
            #print(model.layers[0].input_shape)
            #print(len(best_features))
            model.fit(data[best_features+[new_column]], target,
                  epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[PrintDot()])
            
            mse[new_column] = model.evaluate(xtest[best_features+[new_column]], ytest)[2]
            print(mse)
        min_mse_value = mse.min()
        #print("mse diff" , old_mse -min_mse_value)
        
            
        if((old_mse - min_mse_value > significance_level)&(len(best_features)>0)):
            best_features.append(mse.idxmin())
        elif(len(best_features) == 0):
                best_features.append(mse.idxmin())
        else:
            break
    return best_features


def backward_elimination(data, target, xtest, ytest, significance_level = 0.01):
    features = data.columns.tolist()
    
    og_model = build_model(len(features))
    og_model.fit(data[features], target, epochs = EPOCHS, validation_split = 0.2, verbose = 0, callbacks = [PrintDot()])
    og_mse = og_model.evaluate(xtest[features], ytest)[2]
    #min_mse_value = og_mse
    min_mse_value = 144*math.log(og_mse)+(2*og_model.count_params())
    print(og_mse)
    while(len(features)>0):
        mse = pd.Series(index = features)
        #old_mse = min_mse_value
        old_aic = min_mse_value
        print('old aic '+ str(old_aic))
        for column in features:
            temp = features.copy()
            temp.remove(column)
            model = build_model(len(temp))
            model.fit(data[temp], target, epochs = EPOCHS, validation_split = 0.2, verbose = 0, callbacks = [PrintDot()])
            #mse[column] = model.evaluate(xtest[temp], ytest)[2]
            mse[column] = 144*math.log(model.evaluate(xtest[temp], ytest)[2])+(2*model.count_params())
            print(mse)
            print("number of params", model.count_params())
            print(temp)
        min_mse_value = mse.min()
        aic_diff = mse.min() - old_aic
        print("params", model.count_params())
        #new_aic = 144*math.log(min_mse_value)+(2*model.count_params())
        #aic_diff = new_aic-old_aic
        #print(mse_diff)
        print(aic_diff)
        if((abs(aic_diff) > significance_level)&(aic_diff<0)):
            excluded_feature = mse.idxmin()
            features.remove(excluded_feature)
        else:
            break 
    return features
    
def stepwise_selection(data, target, xtest, ytest, SL_in=0.05,SL_out = 0.05):
    initial_features = data.columns.tolist()
    best_features1 = []
    min_mse_value = 0
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features1))
        mse = pd.Series(index=remaining_features)
        old_mse = min_mse_value
        for new_column in remaining_features:
            model = build_model(len(best_features1+[new_column]))
            #print(len(new_column))
            
            #print(model.layers[0].input_shape)
            #print(len(best_features))
            model.fit(data[best_features1+[new_column]], target,
                  epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[PrintDot()])
            mse[new_column] = model.evaluate(xtest[best_features1+[new_column]], ytest)[2]
        min_mse_value = mse.min()
        if((old_mse - min_mse_value > SL_in)&(len(best_features1)>0)):
            best_features1.append(mse.idxmin())
        elif(len(best_features1) == 0):
            best_features1.append(mse.idxmin())
            
            while(len(best_features1)>0):
                best_features_with_constant = sm.add_constant(data[best_features1])
                p_values = sm.OLS(target, best_features_with_constant).fit().pvalues[1:]
                max_p_value = p_values.max()
                if(max_p_value >= SL_out):
                    excluded_feature = p_values.idxmax()
                    best_features1.remove(excluded_feature)
                else:
                    break 
        else:
            break
    return best_features1