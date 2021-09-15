# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 19:34:53 2021

@author: balle
"""


import pandas as pd

df = pd.read_csv('Cleaned_Cancer_Behavior_Dataset.csv')
df = df.drop(['Unnamed: 0', 'AgeGroup'], axis=1)

dummies = pd.get_dummies(df.Gender).iloc[:,1:]
dummies2 = pd.get_dummies(df.Cancer).iloc[:,1:]


df = pd.concat([df, dummies, dummies2], axis=1)
df = df.drop(['Cancer', 'Gender'], axis=1)



#df = df.rename(columns={ df.columns[81]: "drinking_1"})
#df = df.rename(columns={ df.columns[82]: "drinking_2"})
#df = df.rename(columns={ df.columns[81]: "employed_1"})

dummiesNew = pd.get_dummies(df.Employed).iloc[:,1:]
df = df.drop(['Employed'], axis=1)
df = pd.concat([df, dummiesNew], axis=1)
df = df.rename(columns={ df.columns[80]: "Employed_Patient"})
df = df.rename(columns={ df.columns[81]: "Unemployed_Patient"})
df = df.rename(columns={ df.columns[77]: "Gender"})

df = df.drop(['Unemployed_Patient'], axis=1)

df.to_csv('Data_Cleaned_Cancer_Data.csv')