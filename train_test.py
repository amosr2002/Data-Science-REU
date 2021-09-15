#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 12:56:54 2021

@author: anneliese.mm
"""
#Packages
import pandas as pd
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Normalized Data
df = pd.read_csv('Normalized_Data.csv')


#Looping
X_train = pd.DataFrame()
X_test = pd.DataFrame()
outcomes_test = pd.DataFrame()
outcomes_train = pd.DataFrame()

seed = 10
cancers = {'ALL' : (df['STS'] == 0)&(df['osteosarcoma'] == 0), 'STS' : (df['STS'] == 1)&(df['osteosarcoma'] == 0), 'osteo' : (df['STS'] == 0)&(df['osteosarcoma'] == 1) }


for j in cancers:
    df_cancer = df[cancers[j]]
    
    # X and Outcomes
    X = df_cancer.drop(['Patient_ID','Unnamed: 0', 'Anxious__Depressed_TScore', 'Withdrawn_TScore',
       'Somatic_Complaints_TScore', 'Thought_Problems_TScore',
       'Attention_Problems_TScore', 'Depressive_Problems_TScore',
       'Somatic_Problems_TScore', 'Avoidant_Personality_Problems_TS',
       'Sluggish_Cognitive_Tempo_TScore'], axis = 1 ) 
    
    outcomes = df_cancer[['Anxious__Depressed_TScore', 'Withdrawn_TScore',
       'Somatic_Complaints_TScore', 'Thought_Problems_TScore',
       'Attention_Problems_TScore', 'Depressive_Problems_TScore',
       'Somatic_Problems_TScore', 'Avoidant_Personality_Problems_TS',
       'Sluggish_Cognitive_Tempo_TScore']]
    
    # y= outcomes.iloc[:,0]
    
    #train test split
    Xc_train, Xc_test, outcomesc_train, outcomesc_test = train_test_split(X, outcomes, test_size=0.30, random_state=seed)
    
    #Adding together
    X_train = pd.concat([X_train, Xc_train], axis = 0)
    X_test = pd.concat([X_test, Xc_test], axis = 0)
    outcomes_train = pd.concat([outcomes_train, outcomesc_train], axis = 0)
    outcomes_test = pd.concat([outcomes_test, outcomesc_test], axis = 0)
    
    
    
    


