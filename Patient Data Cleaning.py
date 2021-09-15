# -*- coding: utf-8 -*-
"""
Created on Mon May 31 13:25:23 2021
@author: balle
"""
import pandas as pd
import numpy as np

df = pd.read_csv('Copy_of_Cancer_Patients_Datasets.csv')

df = df.drop(['AssessedPersonId',
'FormCode',
'InformantId',
'DateOnForm',
'Status',
'Society',
'Age',
'Relationship',
'Personal_Strengths_Total',
'Personal_Strengths_TScore',
'Personal_Strengths_Percentile',
'Anxious__Depressed_Total',
'Anxious__Depressed_Percentile',
'Withdrawn_Total',
'Withdrawn_Percentile',
'Somatic_Complaints_Total',
'Somatic_Complaints_Percentile',
'Thought_Problems_Total',
'Thought_Problems_Percentile',
'Attention_Problems_Total',
'Attention_Problems_Percentile',
'Aggressive_Behavior_Total',
'Aggressive_Behavior_Percentile',
'Rule_Breaking_Behavior_Total',
'Rule_Breaking_Behavior_Percentil',
'Intrusive_Total',
'Intrusive_Percentile',
'Internalizing_Problems_Total',
'Internalizing_Problems_Percentil',
'Externalizing_Problems_Total',
'Externalizing_Problems_Percentil',
'Total_Problems_Total',
'Total_Problems_TScore',
'Total_Problems_Percentile',
'Critical_Items_Total',
'Critical_Items_TScore',
'Critical_Items_Percentile',
'Tobacco_Times_Per_Day_Total',
'Tobacco_Times_Per_Day_TScore',
'Tobacco_Times_Per_Day_Percentile',
'Alcohol_Days_Drunk_Total',
'Alcohol_Days_Drunk_TScore',
'Alcohol_Days_Drunk_Percentile',
'Drugs_Days_Used_Total',
'Drugs_Days_Used_TScore',
'Drugs_Days_Used_Percentile',
'Mean_Substance_Use_Total',
'Mean_Substance_Use_TScore',
'Mean_Substance_Use_Percentile',
'Depressive_Problems_Total',
'Depressive_Problems_Percentile',
'Anxiety_Problems_Total',
'Anxiety_Problems_Percentile',
'Somatic_Problems_Total',
'Somatic_Problems_Percentile',
'Avoidant_Personality_Problems_To',
'Avoidant_Personality_Problems_Pe',
'AD_H_Problems_Total',
'AD_H_Problems_Percentile',
'Antisocial_Personality_Total',
'Antisocial_Personality_Percentil',
'Inattention_Problems_Subscale_To',
'Inattention_Problems_Subscale_Pe',
'Hyperactivity_Impulsivity_Proble',
'Hyperactivity_Impulsivity_Probl1',
'Hyperactivity_Impulsivity_Probl2',
'Sluggish_Cognitive_Tempo_Total',
'Sluggish_Cognitive_Tempo_Percent',
'Obsessive_Compulsive_Problems_To',
'Obsessive_Compulsive_Problems_Pe',
'asr_today',
'asr_your_work',
'Paitnet_Index',
'Family_total',
'Total_Fatigue',
'Luteinizing_hormone',
'FSH',
'Testosterone',
'Estradiol',
'Thyroxine',
'TSH',
'Testosterone',
'Estradiol',
'Thyroxine',
'TSH',
'Sodium',
'Potassium',
'Urea',
'Protein__total',
'Albumin',
'Bilirubin_total',
'Alkaline_Phosphatase',
'Alanine_Aminotransferase',
'Phosphate',
'Haemoglobin',
'WBC',
'Platelet',
'MCV',
'MCH',
'MCHC',
'RBC',
'HCT',
'RDW', 'Obs', 'MPV', 'Creatinine', 'Calcium_adjusted'], axis=1)



def to_float(x):
    return float(x)

def to_int(x):
    return int(x)


x = (df.select_dtypes(include=[np.object]).dtypes)
y = x.reset_index()



df.loc[0:206, list(y.iloc[:, 0])]

#df[df.isnull().any(axis=1)]
#df[(df == '.').any(axis=1)]
#df[(df != '.').all(axis=1)]

#Replaces a blank entry with the average of the ages in the range
df.loc[103, ['Current_age']] = df.loc[103, ['Current_age']].replace(['.'], 36.52)

#Replaces incorrect values for row 133 for the diagnosis age and time postdx columns
df['Age_diagnosis'][133] = 14.64657534
df['Time_postdx'][133] = 19.47671233

df['Drinking'][df.Patient_ID == 'STS76'] = 1


#Converts the current_age column into floating point values
df['Current_age'] = df['Current_age'].apply(to_float)


#Adds an AgeGroup Column
bins = [0,18,25,35,120]
labels = ['0-18','18-25','25-35', '35+']
df['AgeGroup'] = pd.cut(df['Current_age'], bins=bins, labels=labels, right=False)


#Replaces blank values in the Education_years column with an average
list1 = list(df[(df.Education_years.notnull()) & (df.Education_years != '.') & (df.Gender == 'F') & (df.Current_age > 35) & (df.Cancer == 'ALL')]['Education_years'])
list1 = list(map(int, list1))
df['Education_years'][103] = sum(list1) / len(list1)

#Replaces blank values in the Education_years column with an average
list3 = list(df[(df.Education_years.notnull()) & (df.Education_years != '.') & (df.Gender == 'M') & (df.AgeGroup =='25-35') & (df.Cancer == 'STS')]['Education_years'])
list3 = list(map(int, list3))
df['Education_years'][(df.Education_years.isnull()) & (df.Gender == 'M') & (df.Cancer == 'STS') & (df.AgeGroup == '25-35')] = sum(list3)/len(list3)

#Replaces blank values in the Education_years column with an average
list4 = list(df[(df.Education_years.notnull()) & (df.Education_years != '.') & (df.Gender == 'F') & (df.AgeGroup =='18-25') & (df.Cancer == 'STS')]['Education_years'])
list4 = list(map(int, list4))
df['Education_years'][(df.Education_years.isnull()) & (df.Gender == 'F') & (df.AgeGroup =='18-25') & (df.Cancer == 'STS')] = sum(list4) / len(list4)

#Replaces blank values in the Education_years column with an average
list5 = list(df[(df.Education_years.notnull()) & (df.Education_years != '.') & (df.Gender == 'F') & (df.AgeGroup =='25-35') & (df.Cancer == 'STS')]['Education_years'])
list5 = list(map(int, list5))
df['Education_years'][(df.Education_years.isnull()) & (df.Gender == 'F') & (df.AgeGroup =='25-35') & (df.Cancer == 'STS')] = sum(list5) / len(list5)

#Replaces blank values in the Education_years column with an average
list6 = list(df[(df.Education_years.notnull()) & (df.Education_years != '.') & (df.Gender == 'F') & (df.AgeGroup =='25-35') & (df.Cancer == 'osteosarcoma')]['Education_years'])
list6 = list(map(int, list6))
df['Education_years'][(df.Education_years.isnull()) & (df.Gender == 'F') & (df.Cancer == 'osteosarcoma') & (df.AgeGroup == '25-35')] = sum(list6)/len(list6)

#Replaces blank values in the Education_years column with an average
list7 = list(df[(df.Education_years.notnull()) & (df.Education_years != '.') & (df.Gender == 'M') & (df.AgeGroup =='25-35') & (df.Cancer == 'osteosarcoma')]['Education_years'])
list7 = list(map(int, list7))
df['Education_years'][(df.Education_years.isnull()) & (df.Gender == 'M') & (df.Cancer == 'osteosarcoma') & (df.AgeGroup == '25-35')] = sum(list7)/len(list7)

#Replaces blank values in the Education_years column with an average
list8 = list(df[(df.Education_years.notnull()) & (df.Education_years != '.') & (df.Gender == 'F') & (df.AgeGroup =='35+') & (df.Cancer == 'osteosarcoma')]['Education_years'])
list8 = list(map(int, list8))
df['Education_years'][(df.Education_years.isnull()) & (df.Gender == 'F') & (df.Cancer == 'osteosarcoma') & (df.AgeGroup == '35+')] = sum(list8)/len(list8)

#Replaces blank values in the Education_years column with an average
list9 = list(df[(df.Education_years.notnull()) & (df.Education_years != '.') & (df.Gender == 'F') & (df.AgeGroup =='0-18') & (df.Cancer == 'STS')]['Education_years'])
list9 = list(map(int, list9))
df['Education_years'][(df.Education_years.isnull()) & (df.Gender == 'F') & (df.Cancer == 'STS') & (df.AgeGroup == '0-18')] = sum(list9)/len(list9)

#Replaces blank values in the Education_years column with an average
list10 = list(df[(df.Education_years.notnull()) & (df.Education_years != '.') & (df.Gender == 'M') & (df.AgeGroup =='18-25') & (df.Cancer == 'osteosarcoma')]['Education_years'])
list10 = list(map(int, list10))
df['Education_years'][(df.Education_years.isnull()) & (df.Gender == 'M') & (df.Cancer == 'osteosarcoma') & (df.AgeGroup == '18-25')] = sum(list10)/len(list10)

#Replaces blank values in the Education_years column with an average
list11 = list(df[(df.Education_years.notnull()) & (df.Education_years != '.') & (df.Gender == 'M') & (df.AgeGroup =='18-25') & (df.Cancer == 'STS')]['Education_years'])
list11 = list(map(int, list11))
df['Education_years'][(df.Education_years.isnull()) & (df.Gender == 'M') & (df.Cancer == 'STS') & (df.AgeGroup == '18-25')] = sum(list11)/len(list11)

#Replaces blank values in the Education_years column with an average
list12 = list(df[(df.Education_years.notnull()) & (df.Education_years != '.') & (df.Gender == 'F') & (df.AgeGroup =='18-25') & (df.Cancer == 'osteosarcoma')]['Education_years'])
list12 = list(map(int, list12))
df['Education_years'][(df.Education_years.isnull()) & (df.Gender == 'F') & (df.Cancer == 'osteosarcoma') & (df.AgeGroup == '18-25')] = sum(list12)/len(list12)

#Replaces blank values in the Education_years column with an average
df['Education_years'] = df['Education_years'].apply(to_float)

#Replaces blank values in the Family_Mutuality column with an average
list2 = list(df[(df.Family_Mutuality.notnull()) & (df.Family_Mutuality != '.') & (df.Gender == 'F') & (df.Current_age > 35) & (df.Cancer == 'ALL')]['Family_Mutuality'])
list2 = list(map(int, list2))
df['Family_Mutuality'][103] = sum(list2) / len(list2)

#Replaces blank values in the Family_Communication column with an average
list13 = list(df[(df.Family_Communication.notnull()) & (df.Family_Communication != '.') & (df.Gender == 'F') & (df.Current_age > 35) & (df.Cancer == 'ALL')]['Family_Communication'])
list13 = list(map(int, list13))
df['Family_Communication'][103] = sum(list13) / len(list13)

#Replaces blank values in the Family_Control column with an average
list14 = list(df[(df.Family_Control.notnull()) & (df.Family_Control != '.') & (df.Gender == 'F') & (df.Current_age > 35) & (df.Cancer == 'ALL')]['Family_Control'])
list14 = list(map(int, list14))
df['Family_Control'][103] = sum(list14) / len(list14)


#Replaces blank values in the Family_Conflict column with an average
list15 = list(df[(df.Family_Conflict.notnull()) & (df.Family_Conflict != '.') & (df.Gender == 'F') & (df.Current_age > 35) & (df.Cancer == 'ALL')]['Family_Conflict'])
list15 = list(map(int, list15))
df['Family_Conflict'][103] = sum(list15) / len(list15)

#Replaces blank values in the Family_Concern column with an average
list16 = list(df[(df.Family_Concern.notnull()) & (df.Family_Concern != '.') & (df.Gender == 'F') & (df.Current_age > 35) & (df.Cancer == 'ALL')]['Family_Concern'])
list16 = list(map(int, list16))
df['Family_Concern'][103] = sum(list16) / len(list16)

#Replaces blank values in the Physical_activity column with an average
list17 = list(df[(df.Physical_activity.notnull()) & (df.Physical_activity != '.') & (df.Gender == 'F') & (df.Current_age > 35) & (df.Cancer == 'ALL')]['Physical_activity'])
list17 = list(map(int, list17))
df['Physical_activity'][103] = sum(list17) / len(list17)


#Replaces blank values in the General_Fatigue column with an average
list18 = list(df[(df.General_Fatigue.notnull()) & (df.General_Fatigue != '.') & (df.Gender == 'F') & (df.Current_age > 35) & (df.Cancer == 'ALL')]['General_Fatigue'])
list18 = list(map(float, list18))
df['General_Fatigue'][103] = sum(list18) / len(list18)

#Replaces blank values in the Sleep_Fatigue column with an average
list19 = list(df[(df.Sleep_Fatigue.notnull()) & (df.Sleep_Fatigue != '.') & (df.Gender == 'F') & (df.Current_age > 35) & (df.Cancer == 'ALL')]['Sleep_Fatigue'])
list19 = list(map(float, list19))
df['Sleep_Fatigue'][103] = sum(list19) / len(list19)

#Replaces blank values in the Cognitive_Fatigue column with an average
list20 = list(df[(df.Cognitive_Fatigue.notnull()) & (df.Cognitive_Fatigue != '.') & (df.Gender == 'F') & (df.Current_age > 35) & (df.Cancer == 'ALL')]['Cognitive_Fatigue'])
list20 = list(map(float, list20))
df['Cognitive_Fatigue'][103] = sum(list20) / len(list20)


#Creates a dataframe with the averages for each age, cancer, and gender category
df1 = df.loc[(df.BMI.notnull()) & (df.BMI != '.')]
df1['BMI'][111] = 21.5
df1['BMI'] = df1['BMI'].apply(to_float)
df1 = df1.groupby(['Cancer','AgeGroup', 'Gender']).mean()


df['BMI'][111] = 21.5

#Replaces blank values in the BMI column with averages
df['BMI'][((df.BMI.isnull()) | (df.BMI =='.') ) & (df.Gender == 'M') & (df.Cancer == 'ALL') & (df.AgeGroup == '35+')] = 26.650000
df['BMI'][((df.BMI.isnull()) | (df.BMI =='.') ) & (df.Gender == 'F') & (df.Cancer == 'ALL') & (df.AgeGroup == '18-25')] = 20.929444
df['BMI'][((df.BMI.isnull()) | (df.BMI =='.') ) & (df.Gender == 'F') & (df.Cancer == 'ALL') & (df.AgeGroup == '25-35')] = 22.131429
df['BMI'][((df.BMI.isnull()) | (df.BMI =='.') ) & (df.Gender == 'M') & (df.Cancer == 'ALL') & (df.AgeGroup == '18-25')] = 22.429000
df['BMI'][((df.BMI.isnull()) | (df.BMI =='.') ) & (df.Gender == 'M') & (df.Cancer == 'ALL') & (df.AgeGroup == '25-35')] = 24.111818
df['BMI'][((df.BMI.isnull()) | (df.BMI =='.') ) & (df.Gender == 'F') & (df.Cancer == 'ALL') & (df.AgeGroup == '35+')] = 22.420000
df['BMI'][((df.BMI.isnull()) | (df.BMI =='.') ) & (df.Gender == 'M') & (df.Cancer == 'STS') & (df.AgeGroup == '25-35')] = 21.687143
df['BMI'][((df.BMI.isnull()) | (df.BMI =='.') ) & (df.Gender == 'F') & (df.Cancer == 'osteosarcoma') & (df.AgeGroup == '25-35')] = 19.920833

#Connverts the Column DataType to float
df['BMI'] = df['BMI'].apply(to_float)

#Creates a dataframe with the averages for each category
df2 = df.loc[(df.Heart_Rate.notnull()) & (df.Heart_Rate != '.')]
df2['Heart_Rate'] = df2['Heart_Rate'].apply(to_float)
df2 = df2.groupby(['Cancer','AgeGroup', 'Gender']).mean()

#Replaces blank values in the Heart_Rate column with averages
df['Heart_Rate'][((df.Heart_Rate.isnull()) | (df.Heart_Rate =='.') ) & (df.Gender == 'M') & (df.Cancer == 'ALL') & (df.AgeGroup == '35+')] = 90.000000
df['Heart_Rate'][((df.Heart_Rate.isnull()) | (df.Heart_Rate =='.') ) & (df.Gender == 'F') & (df.Cancer == 'ALL') & (df.AgeGroup == '18-25')] = 87.294118
df['Heart_Rate'][((df.Heart_Rate.isnull()) | (df.Heart_Rate =='.') ) & (df.Gender == 'F') & (df.Cancer == 'ALL') & (df.AgeGroup == '25-35')] = 92.090909
df['Heart_Rate'][((df.Heart_Rate.isnull()) | (df.Heart_Rate =='.') ) & (df.Gender == 'M') & (df.Cancer == 'ALL') & (df.AgeGroup == '18-25')] = 88.000000
df['Heart_Rate'][((df.Heart_Rate.isnull()) | (df.Heart_Rate =='.') ) & (df.Gender == 'M') & (df.Cancer == 'ALL') & (df.AgeGroup == '25-35')] = 76.357143
df['Heart_Rate'][((df.Heart_Rate.isnull()) | (df.Heart_Rate =='.') ) & (df.Gender == 'M') & (df.Cancer == 'STS') & (df.AgeGroup == '25-35')] = 74.750000
#Connverts the Column DataType to float
df['Heart_Rate'] = df['Heart_Rate'].apply(to_float)

#Replaces blank values in the BP_Systolic column with averages
df3 = df.loc[(df.BP_Systolic.notnull()) & (df.BP_Diastolic != '.')]
df3['BP_Systolic'] = df3['BP_Systolic'].apply(to_float)
df3 = df3.groupby(['Cancer','AgeGroup', 'Gender']).mean()
df['BP_Systolic'][((df.BP_Systolic.isnull()) | (df.BP_Systolic =='.') ) & (df.Gender == 'M') & (df.Cancer == 'ALL') & (df.AgeGroup == '35+')] =  136.750000
df['BP_Systolic'][((df.BP_Systolic.isnull()) | (df.BP_Systolic =='.') ) & (df.Gender == 'F') & (df.Cancer == 'ALL') & (df.AgeGroup == '35+')] =  118.750000
df['BP_Systolic'][((df.BP_Systolic.isnull()) | (df.BP_Systolic =='.') ) & (df.Gender == 'F') & (df.Cancer == 'ALL') & (df.AgeGroup == '18-25')] =  110.857143
df['BP_Systolic'][((df.BP_Systolic.isnull()) | (df.BP_Systolic =='.') ) & (df.Gender == 'M') & (df.Cancer == 'ALL') & (df.AgeGroup == '18-25')] =   126.083333
df['BP_Systolic'][((df.BP_Systolic.isnull()) | (df.BP_Systolic =='.') ) & (df.Gender == 'M') & (df.Cancer == 'ALL') & (df.AgeGroup == '25-35')] =   131.562500
df['BP_Systolic'][((df.BP_Systolic.isnull()) | (df.BP_Systolic =='.') ) & (df.Gender == 'F') & (df.Cancer == 'ALL') & (df.AgeGroup == '25-35')] =   112.666667
df['BP_Systolic'][((df.BP_Systolic.isnull()) | (df.BP_Systolic =='.') ) & (df.Gender == 'M') & (df.Cancer == 'STS') & (df.AgeGroup == '25-35')] = 123.375000
df['BP_Systolic'] = df['BP_Systolic'].apply(to_float)

#Replaces blank values in the BP_Diastolic column with averages
df4 = df.loc[(df.BP_Diastolic.notnull()) & (df.BP_Diastolic != '.')]
df4['BP_Diastolic'] = df4['BP_Diastolic'].apply(to_float)
df4 = df4.groupby(['Cancer','AgeGroup', 'Gender']).mean()
df['BP_Diastolic'][((df.BP_Diastolic.isnull()) | (df.BP_Diastolic =='.') ) & (df.Gender == 'M') & (df.Cancer == 'ALL') & (df.AgeGroup == '35+')] =  83.250000
df['BP_Diastolic'][((df.BP_Diastolic.isnull()) | (df.BP_Diastolic =='.') ) & (df.Gender == 'F') & (df.Cancer == 'ALL') & (df.AgeGroup == '35+')] =  73.750000
df['BP_Diastolic'][((df.BP_Diastolic.isnull()) | (df.BP_Diastolic =='.') ) & (df.Gender == 'F') & (df.Cancer == 'ALL') & (df.AgeGroup == '18-25')] =  69.571429
df['BP_Diastolic'][((df.BP_Diastolic.isnull()) | (df.BP_Diastolic =='.') ) & (df.Gender == 'M') & (df.Cancer == 'ALL') & (df.AgeGroup == '18-25')] =   72.208333
df['BP_Diastolic'][((df.BP_Diastolic.isnull()) | (df.BP_Diastolic =='.') ) & (df.Gender == 'M') & (df.Cancer == 'ALL') & (df.AgeGroup == '25-35')] =    79.562500
df['BP_Diastolic'][((df.BP_Diastolic.isnull()) | (df.BP_Diastolic =='.') ) & (df.Gender == 'F') & (df.Cancer == 'ALL') & (df.AgeGroup == '25-35')] =   70.400000
df['BP_Diastolic'][((df.BP_Diastolic.isnull()) | (df.BP_Diastolic =='.') ) & (df.Gender == 'M') & (df.Cancer == 'STS') & (df.AgeGroup == '25-35')] = 70.250000
df['BP_Diastolic'] = df['BP_Diastolic'].apply(to_float)

#Replaces blank values in these columns
df['CHC_yesORno'][(df.CHC_yesORno == '.') | (df.CHC_yesORno.isnull())] = 0
df['RTX'][(df.RTX == '.') | (df.RTX.isnull())] = 0
df['CRT'][(df.CRT == '.') | (df.CRT.isnull())] = 0
df['HSCT'][(df.HSCT == '.') | (df.HSCT.isnull())] = 0
df['Surgery'][(df.Surgery == '.') | (df.Surgery.isnull())] = 0
df['FSH_AB'][(df.FSH_AB == '.') | (df.FSH_AB.isnull())] = 0
df['Testosterone_AB'][(df.Testosterone_AB == '.') | (df.Testosterone_AB.isnull())] = 0
df['Estradiol_AB'][(df.Estradiol_AB == '.') | (df.Estradiol_AB.isnull())] = 0


#Replace 2s in drinking column with 1
df['Drinking'][df['Drinking']==2]=1


df= df.apply(pd.to_numeric, errors='ignore')

#Saves the cleaned dataset into a csv file
df.to_csv('Cleaned_Cancer_Behavior_Dataset.csv')