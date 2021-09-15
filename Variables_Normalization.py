import pandas as pd
from sklearn import preprocessing

df = pd.read_csv('Data_Cleaned_Cancer_Data.csv')
df1 = df
df1 = df1.drop(['Unnamed: 0', 'Patient_ID','Anxious__Depressed_TScore',
       'Withdrawn_TScore', 'Somatic_Complaints_TScore',
       'Thought_Problems_TScore', 'Attention_Problems_TScore',
       'Depressive_Problems_TScore', 'Somatic_Problems_TScore',
       'Avoidant_Personality_Problems_TS', 'Sluggish_Cognitive_Tempo_TScore'], axis=1)
values = df1.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(values)
df1 = pd.DataFrame(x_scaled, columns=df1.columns)

df2 = pd.concat([df1, df[['Patient_ID','Anxious__Depressed_TScore',
       'Withdrawn_TScore', 'Somatic_Complaints_TScore',
       'Thought_Problems_TScore', 'Attention_Problems_TScore',
       'Depressive_Problems_TScore', 'Somatic_Problems_TScore',
       'Avoidant_Personality_Problems_TS', 'Sluggish_Cognitive_Tempo_TScore']]], axis=1)


df2['Time_postdx'] = df2['Current_age']-df2['Age_diagnosis']

df2 = df2.drop(['Aggressive_Behavior_TScore', 'Rule_Breaking_Behavior_TScore',
       'Intrusive_TScore', 'Internalizing_Problems_TScore',
       'Externalizing_Problems_TScore', 'Anxiety_Problems_TScore',
       'AD_H_Problems_TScore', 'Antisocial_Personality_TScore',
       'Inattention_Problems_Subscale_TS', 'Obsessive_Compulsive_Problems_TS'], axis=1)
df2.to_csv('Normalized_Data.csv')