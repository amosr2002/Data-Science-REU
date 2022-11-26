# Data-Science-REU

We propose a prognostic machine learning (ML) framework to support the behavioural outcome prediction for cancer survivors. Specifically, our contributions are four-fold: (1) devise a data-driven, clinical domain-guided pipeline to select the best set of predictors among cancer treatments, chronic health conditions, and socio-environmental factors to perform behavioural outcome predictions; (2) use the state-of-the-art two-tier ensemble-based technique to select the best set of predictors for the downstream ML regressor constructions; (3) develop a StackNet Regressor Architecture (SRA) algorithm, i.e., an intelligent meta-modeling algorithm, to dynamically and automatically build an optimized multilayer ensemble-based RA from a given set of ML regressors to predict long-term behavioural outcomes; and (4) conduct a preliminarily experimental case study on our existing study data (i.e., 207 cancer survivors who suffered from either Osteogenic Sarcoma, Soft Tissue Sarcomas, or Acute Lymphoblastic Leukemia before the age of 18) collected by our investigators in a public hospital in Hong Kong. In this pilot study, we demonstrate that our approach outperforms the traditional statistical and computation methods, including Linear and non-Linear ML regressors.

- DynamicStackNet.py
  - Includes the codes for the dynamic stacking architecture. This algorithm automatically ensembles an aribitrary amount of Machine Learning models into different layers
- FeatureSelectionNeuralNetwork.py
  - Includes the code for the neural network architecture and the backwards/forward selection feature selection methods. 
- OneHotEncodingData.py
  - Includes the code for the one hot categorical encoding process. 
- Patient Data Cleaning.py
  - Includes the data cleaning code. Cleaned out null values and imputed some missing values with average value
- Variables_Normalization.py
  - Normalized the variables between 0 and 1
- train_test.py
  - Splits the data into train and test sets
- HEALTHINF2022FinalPaper (3).pdf
  - Includes the full Conference Published Research Paper
