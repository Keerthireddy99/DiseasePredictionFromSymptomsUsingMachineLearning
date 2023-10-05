# Disease Prediction From Symptoms Using Machine Learning

The main aim of this project is to build a machine-learning model with the given datasets that can predict diseases from symptoms.

### Abstract

With the fast advancement of technology and data, the healthcare sector is currently one of the most important study areas. Managing the vast volume of patient data is challenging. Big Data Analytics makes handling this data simpler. There are several methods used all around the world to cure various ailments. A new method that aids in illness detection and prediction is machine learning. In this study, machine learning is used to predict illness based on symptoms. On the presented dataset, machine learning methods like Naive Bayes, Decision Tree, and Random Forest are used to forecast the illness. Through the use of the Python programming language, it is implemented.

### Introduction

Disease Prediction using Machine Learning is a method that forecasts the disease based on the data the user provides. Additionally, it accurately forecasts the user's or the patient's disease based on the data or symptoms entered into the system and returns findings accordingly. If the customer only wants to know the sort of ailment the patient has experienced, and the condition is not particularly significant. It is a system that gives users advice on how to keep their health systems in good shape and offers a technique to identify diseases using this prediction. Today's health sector plays a significant role in treating patients' illnesses, so this is frequently helpful for the sector to inform the user as well as helpful for the user in case he or she chooses not to visit the hospital or other clinics. By entering the symptoms and all other relevant information, the user can understand the disease they are experiencing. 

#### Keywords

Disease Prediction, Machine Learning, PCA, Naive Baye’s Algorithm, Random Forest, Decision Tree, SVM.

#### Data

The dataset has been taken from Kaggle Data set.  
Dataset Link: https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machinelearning 
•	All the features of the dataset are categorical (i.e., each feature/symptom has a binary value of 0 or 1) which implies whether the disease has this symptom or not. 
•	The data file has 133 columns. 132 of these columns are symptoms that a person experiences and last column is the prognosis and a total of 4920 rows/instances. 
 
The head of the data or the first five rows of the data look as the following –  
![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/a3efc93e-2295-4ae6-916b-0f29f8c8289c)

![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/4308a6d0-8dc0-4638-b6f3-da2bdf87b38a)


### Data Preparation
#### Data Analysis

 •	Here the data type of symptoms is int64 and of the target variable(prognosis) is object which are strings.            
 •	These symptoms are mapped to 42 diseases that can classify these set of symptoms to.                       
    The 42 diseases are –                                                                                    
    'Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction', 'Peptic ulcer disease’, 'AIDS', 'Diabetes     ', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine', 'Cervical  spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue’, ‘Typhoid’, 'hepatitis A', 'Hepatitis B', 'Hepatitis C','Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis’, ‘Tuberculosis', 'Common Cold', 'Pneumonia','Dimorphic hemmorhoids(piles)', 'Heartattack’,'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia','Osteoarthristis', 'Arthritis’,'(vertigo) Paroymsal Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis','Impetigo’  
 
 •	The dataset is completely balanced which means that each disease has 120 sets of data/rows 

#### Code:
##### 1:
_disease_counts = df["prognosis"].value_counts()       
disease_counts_
##### 2:
_disease_counts_valid = df_valid["prognosis"].value_counts()                             
disease_counts_valid_
##### 3:
_target_df = pd.DataFrame(df['prognosis'], columns=['prognosis'])                                       
labelencoder = LabelEncoder()                  
target_df['Prognosis_Cat'] = labelencoder.fit_transform(target_df['prognosis'])                 
target_df_
#### Output –
#### 1: Training Data
  
 ![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/3d4a4652-7bb6-4b0b-80dd-7ad42d262682)

 
 
#### 2:  Validation Data 

![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/a578b9bc-6c7a-4e32-a696-f037b3fe74bd)

#### 3: 
![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/2765b83b-ade6-455b-96be-4212b73f4df0)

### Data Exploration 

Data exploration helps one understand how the data is distributed. This can help in understanding the underlying patterns. 

#### Code – 

plt.figure(figsize = (18,8))                     
sns.barplot(x = df["prognosis"].unique(), y = disease_counts.values, data = df)                   
plt.xticks(rotation=90)                     
#plt.savefig('count')                             
plt.show() 

#### Plot – Training Data 

![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/124e58dc-331a-4692-bf23-d43a53ab8b5a)

#### Code –  

plt.figure(figsize = (18,8))                         
sns.barplot(x = df_valid["prognosis"].unique(), y = disease_counts_valid.values, data = df)                        
plt.xticks(rotation=90)                             
plt.show() 

#### Plot – Validation Data 

![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/9c8a6e28-49a8-489c-9063-ba219608795c)

  
### Frequency Graph: 
 The frequency graph of the symptoms is below: 

 ![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/607f6912-9f1c-4214-ac62-0b3161a535d8)

#### Most frequent symptoms (Most significant features) 

![IMG2](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/fc3fdc97-fecd-4110-9b6f-4a2da8161a28)

Fatigue and Vomiting are the most frequent symptoms. They happen to be symptoms of many diseases. 
 
![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/c2a801af-4da5-4b05-981a-4389c8d88602)

![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/bb708aa6-13f2-4046-8ce1-a8eb9cf588a8)

#### Least frequent Symptoms 

![img](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/90f7a0ea-201f-476a-932d-70dd1aef1979)

#### TSNE plot: 
This high dimensional data has been projected to 2 dimensions for visualisation purpose using tsne plot 

![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/d9ea354e-d027-4151-a161-156cdf091e1c)

### Correlation: 

sns.set(rc = {'figure.figsize':(100,100)}) 	                                                             
sns.heatmap(X.corr(), annot = True, fmt='.2g',cmap= 'coolwarm') 		                                                          
#plt.savefig('Corr')

#### Plot: 
 
![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/928b4651-82a6-436d-8cc0-9cc5994493f3)
  
#### Correlation Analysis: 
From the correlation the below are the features/symptoms which are highly correlated (>0.90) –  
 
['cold_hands_and_feets', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 
'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'bruising', 'swollen_legs', 
'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 
'swollen_extremeties', 'drying_and_tingling_lips', 'slurred_speech', 'hip_joint_pain', 
'unsteadiness', 'loss_of_smell', 'continuous_feel_of_urine', 'internal_itching', 
'altered_sensorium', 'belly_pain', 'abnormal_menstruation', 'increased_appetite', 
'polyuria', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 
'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 
'fluid_overload', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 
'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 
'red_sore_around_nose', 'yellow_crust_ooze' ] 

•	We had dropped the above highly correlated features from our dataset.  
•	The current dimensionality of the dataset is – (4920, 89) 

#### Dimensionality reduction using PCA: 

Principal Component Analysis (PCA) is one of the most popular linear dimension reductions. 
Sometimes, it is used alone and sometimes as a starting solution for other dimension reduction methods. 
PCA is a projection-based method which transforms the data by projecting it onto a set of orthogonal axes. 

For our dataset, we had tried to reduce the dimensionalities such that they capture – 99%, 95%, 90%, 85% of variances. The results are shown below – 
#### Explained variances –  
#### Code: 
components = None                                        
pca = PCA(n_components = components)                                   
pca.fit(X)                                         
print("Variances (Percentage):")                                     
print(pca.explained_variance_ratio_ * 100))                                        
#### Output: 
#### Variances (Percentage): 
[1.43459866e+01  1.18471298e+01  7.37690949e+00  6.99273877e+00                                                              
 6.04020735e+00  3.96880033e+00  3.61636054e+00  3.46995409e+00                                                           
 2.92020152e+00  2.80102260e+00  2.44933097e+00  2.36612329e+00                                                              
 2.18516190e+00  2.04049697e+00  1.86424284e+00  1.74772998e+00                                                           
 1.57899240e+00   1.50804812e+00  1.47062218e+00  1.41337232e+00                                                             
 1.31951258e+00   1.24724722e+00  1.12496759e+00  1.08659723e+00                                                               
 1.01791768e+00   9.84893290e-01   9.17916304e-01   8.59302675e-01                                                             
 8.09078163e-01   7.86157597e-01    7.45040307e-01   6.97479538e-01                                                                  
 6.88069784e-01   6.18983580e-01   5.28821480e-01   4.58718848e-01                                                                 
 4.39291949e-01   3.93348890e-01   3.90465351e-01   1.84142273e-01                                                               
 1.68668899e-01   1.36017578e-01   1.24772325e-01   1.16175547e-01                                                             
 1.06895260e-01    1.04350851e-01   1.01271935e-01   9.32352691e-02                                                               
 8.64930098e-02   7.84802211e-02   7.48770454e-02   6.90084664e-02                                                                  
 6.67717859e-02   6.28834328e-02   6.21936011e-02   5.79796333e-02                                                                    
 5.62733037e-02   5.48961570e-02   5.01186473e-02   4.87761585e-02                                                                
 4.68142012e-02   4.54000275e-02   4.49394482e-02   4.49394482e-02                                                                   
 4.49394482e-02   4.49394482e-02   4.49394482e-02   4.49394482e-02                                                                 
 4.49394482e-02   4.39512117e-02   4.31224116e-02   4.09037226e-02                                                                   
 3.98665187e-02   3.88097799e-02   3.60672618e-02   3.50453785e-02                                                                 
 3.33522819e-02   3.09440723e-02   3.00675702e-02   2.90310159e-02                                                                      
 2.81549436e-02   2.62354953e-02   2.53543714e-02   2.22375102e-02                                                                      
 2.19719128e-02   1.95226608e-02   1.32691210e-02   1.02809910e-02                                                                    
 3.49789065e-03] 
 
• From the above output we can see that – 

o The 1st component/feature alone captures 14.34% variability in the data.                                                             
o The 2nd component/feature alone captures 11.84% variability in the data and so on. o The 89 components all together account for 100% variability in the data. 
 
#### Cumulative variances –  
#### Code: 
components = None                                                         
pca = PCA(n_components = components)                                                      
pca.fit(X) print("Cumulative Variances (Percentage):")                                           
print(pca.explained_variance_ratio_.cumsum() * 100)                                       

#### Output: 
##### Cumulative Variances (Percentage): 
[ 14.34598656  26.1931164   33.57002589  40.56276466  46.60297201   50.57177234  54.18813287  57.65808696  60.57828848  63.37931108 
  65.82864205  68.19476534  70.37992724  72.42042421  74.28466706 
  76.03239704  77.61138944  79.11943756  80.59005974  82.00343206 
  83.32294464  84.57019186  85.69515945  86.78175668  87.79967436 
  88.78456765  89.70248395  90.56178662  91.37086479  92.15702238 
  92.90206269  93.59954223  94.28761201  94.90659559  95.43541707 
  95.89413592  96.33342787  96.72677676  97.11724211  97.30138439 
  97.47005328  97.60607086  97.73084319  97.84701873  97.95391399 
  98.05826485  98.15953678  98.25277205  98.33926506  98.41774528 
  98.49262233  98.56163079  98.62840258  98.69128601  98.75347961 
  98.81145925  98.86773255  98.92262871  98.97274735  99.02152351 
  99.06833771  99.11373774  99.15867719  99.20361664  99.24855609 
  99.29349553  99.33843498  99.38337443  99.42831388  99.47226509 
  99.5153875   99.55629122  99.59615774  99.63496752  99.67103478   99.70608016  99.73943244  99.77037652  99.80044409  99.8294751 
  99.85763005  99.88386554  99.90921991  99.93145742  99.95342934 
  99.972952    99.98622112  99.99650211 100.        ]  
  
• From the above output we can see that –                                                
o The 1st component/feature captures the 14.34% variability in data.                                                       
o The 1st and 2nd components/features together capture the 26.19% variability in data and so on.                                                    

#### Scree plot: 
The scree plot helps to visualize the number of components that are needed to capture the various amounts of variability in the data: 
##### Code: 

![IMG 3](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/e4d1f1ba-ead5-480d-9809-8637c7258baa)

### Output: 

![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/f956de0e-0124-4013-942c-952ae4923800)

The next step is to apply PCA to find the desired number of components based on the desired explained variances - 85%, 90%, 95%, 99%: 
#### Code: 
comp = [0.85,0.90,0.95,0.99]                                                      
for i in comp:                                                  
    var_per = int(i*100)                              
    pca = PCA(n_components = i) 
    pca.fit(X)     print("Cumulative Variances (Percentage):")     
    print(np.cumsum(pca.explained_variance_ratio_ * 100))     
    components = len(pca.explained_variance_ratio_)     
    print(f'Number of components: {components}', f'for {var_per}', '% variance')     
    plt.figure(figsize=(10,10))     
    plt.plot(range(1, components + 1), np.cumsum(pca.explained_variance_ratio_ * 100))     
    plt.xlabel("Number of components") 
    plt.ylabel("Explained variance (%)") 
    #plt.savefig(str(var_per))     
    plt.show() 
#### Output: 
•	Number of components to capture in 99% is 60:

##### Cumulative Variances (Percentage):                                               
[14.34598656 26.1931164 33.57002589 40.56276466 46.60297201 50.57177 234 54.18813287 57.65808696 60.57828848 63.37931108 65.82864205 68.1
9476534 70.37992724 72.42042421 74.28466706 76.03239704 77.61138944 79.11943756 80.59005974 82.00343206 83.32294464 84.57019186 85.69515 945 86.78175668 87.79967436 88.78456765 89.70248395 90.56178662 91.3 7086479 92.15702238 92.90206269 93.59954223 94.28761201 94.90659559 95.43541707 95.89413592 96.33342787 96.72677676 97.11724211 97.30138 439 97.47005328 97.60607086 97.73084319 97.84701873 97.95391399 98.0 5826485 98.15953678 98.25277205 98.33926506 98.41774528 98.49262233 98.56163079 98.62840258 98.69128601 98.75347961 98.81145925 98.86773 255 98.92262871 98.97274735 99.02152351] 


![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/28658c41-cbf3-49df-be90-29ee962440f6)

•	Number of components to capture in 95% is 35: 

##### Cumulative Variances (Percentage): 
[14.34598656 26.1931164 33.57002589 40.56276466 46.60297201 50.57177
234 54.18813287 57.65808696 60.57828848 63.37931108 65.82864205 68.1 9476534 70.37992724 72.42042421 74.28466706 76.03239704 77.61138944 
79.11943756 80.59005974 82.00343206 83.32294464 84.57019186 85.69515 945 86.78175668 87.79967436 88.78456765 89.70248395 90.56178662 91.3 7086479 92.15702238 92.90206269 93.59954223 94.28761201 94.90659559 95.43541707] 

![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/eeaf526d-00f6-41a9-b48d-d5f7bea4b600)
  
•	Number of components to capture in 90% is 28: 

##### Cumulative Variances (Percentage): 
   [14.34598656 26.1931164 33.57002589 40.56276466 46.60297201 50.57177234    54.18813287 57.65808696 60.57828848 63.37931108 65.82864205 68.19476534 
   70.37992724 72.42042421 74.28466706 76.03239704 77.61138944 79.11943756 
   80.59005974 82.00343206 83.32294464 84.57019186 85.69515945 86.78175668 
   87.79967436 88.78456765 89.70248395 90.56178662] 
    

![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/cac26726-3c0d-49de-8b65-d42197ce00bb)

•	Number of components to capture in 85% is 23: 

##### Cumulative Variances (Percentage): 
[14.34598656 26.1931164 33.57002589 40.56276466 46.60297201 50.57177 234 54.18813287 57.65808696 60.57828848 63.37931108 65.82864205 68.1 9476534 70.37992724 72.42042421 74.28466706 76.03239704 77.61138944 79.11943756 80.59005974 82.00343206 83.32294464 84.57019186 85.69515
945] 
 

 ![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/ec1c6b3d-39ed-4be1-aaf4-f408195cd7a5)
 
#### Transforming the data and feature importance

We had chosen to reduce the dimensionality from 89 to 23 which captures the 85% variance of t he original data.  
We can see the importance of each original feature to each principal component below – 

![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/870b343e-18d9-4a7d-aaca-8d7d8de57ad6)
  
The importance of each feature is reflected by the above magnitude of the corresponding values i n the above output — the higher magnitude, the higher the importance. 
 The top 4 features that contributes the most to each of the 23 components is shown below – 
 

![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/6d4b82b5-18cc-477c-bfb4-9c8730023f99)
  
#### Data Pre-processing: 
1.	We had dropped the features with high correlation (Correlation > .90). Hence, 42 feature of the total 131 features have been dropped and 89 features are present now in the dataset. 
2.	Next, step is dimensionality reduction using PCA. With PCA we had reduced the dimensionality from 89 to 23 which capture the 85% of the variance. 
 
#### Train Test Spit 
The train-test split is used to evaluate the performance of machine learning algorithms The model is trained on the training data and the prediction made on the test data set. The results can then be cross-checked. 

Splitting of data into training and testing –

•	Training-60%                                                          
•	Testing-40%                                                                
##### Code –  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40)                                                        
#### Parameter Tuning: 
We had selected the following machine learning algorithms to build the model Naïve Baye’s, Random Forest, Decision Tree and SVM.                                 
We had used GridSearchCV algorithm to find the best parameters of each algorithm that best fits our data. 
#### Naïve Baye’s: 
Naïve Bayes Classifier is one of the simple and most effective Classification algorithms which helps in building the fast machine learning models that can make quick predictions. It is a probabilistic classifier, which means it predicts based on the probability of an object. The Naive Bayes classifier assumes the presence of a particular feature in a class is unrelated to the presence of any other feature. It is very easy to build and useful for large datasets. 

Parameters – Var_Smoothing                                                                            
Best parameters – Var_smoothing = 0.001 
##### Code: 
 

![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/ea25a1d2-7cd6-4a0d-b3bf-616e22cc4c16)
  
 
#### SVM (Support Vector Machines): 
Support Vector Machine is a linear model used for classification and regression problems. It can solve linear and non-linear problems. SVM creates a hyperplane that separates the data into classes. Support Vector Machine (SVM) is a supervised machine learning algorithm for classification and regression. Though we say regression problems as well it is best suited for classification. The objective of the SVM algorithm is to find a hyperplane in an N-dimensional space that distinctly classifies the data points. 

Parameters – C, gamma, Kernel                                                 
Best parameters – C = 10, gamma = 0.1, kernel = ‘rbf 
##### Code: 

![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/2d8fb19c-64f7-4362-afc4-1727c1242f42)
  
#### Random Forest: 
Random forest is a Supervised Machine Learning Algorithm that is used widely in Classification and Regression problems. It builds decision trees on different samples and takes their majority vote for classification and average in case of regression. 
One of the most important features of the Random Forest Algorithm is that it can handle the data set containing continuous variables as in the case of regression and categorical variables as in the case of classification. It performs better results for classification problems. 

Parameters – n_estimators, criterion, max_depth, min_samples_leaf, min_samples_split, max_features                                       
Best Parameters - n_estimators = 15, criterion = entropy, max_depth = 6, min_samples_leaf = 1, min_samples_split = 2, max_features = log2 
 

![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/dde90265-452d-4059-a7d3-bb3a8b28a3d7)
  
 
#### Decision Tree: 
A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility.                                                                                                                
Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.                                        
By using a series of straightforward decision trees, a decision tree is a framework that can be used to successfully split up a huge collection of information into smaller sets of records. The individuals in the resultant groups resemble one another more and more with each subsequent division. An example of a decision tree model is a collection of guidelines for segmenting a sizable, varied population into smaller, more homogenous groups that are mutually exclusive about a certain aim. The decision tree is typically used to: Determine the likelihood that a given record belongs to each category when the goal variable is ordinarily categorical.                     
To categorize the record by putting it in the most probable category (or category). The decision tree in this illness prediction system categorizes the symptoms to lessen the difficulty of the dataset. 

Parameters – max_depth, min_samples_leaf, criterion                                                                                
Best parameters – max_depth = 7, min_samples_leaf = 5, criterion = entropy                                                                     
##### Code: 

 ![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/daf8fe8f-84ba-4c39-bf5b-cf7aa9d7ba21)
 
### Model Performance: 
#### Accuracy Score: 

  ![image](https://github.com/Keerthireddy99/DiseasePredictionFromSymptomsUsingMachineLearning/assets/145499897/865d9356-db42-4f65-8d3e-4a1db517aed0)

From our model’s performance, we can see that Naïve Baye’s, SVM and Random Forest have 100% accuracy on validation data. Random Forest model has 100% accuracy on all the 3 data sets – training, testing and validation sets. It is possible that this model may overfit the data. We believe that it is better to take voting from the Naïve Baye’s, SVM and Decision Tree algorithms to predict the disease. 
### Conclusion: 
We have thus come to the conclusion that machine learning may be utilized to track our health in an efficient manner. We may periodically check our health for free and stay healthy. After constructing the machine learning model. The user must select the symptoms from the 131 which are in our dataset based on these our model predicts the disease. After receiving the forecast, the user will get insight into their health and, if necessary, contact the appropriate doctor. Any person in our planet can become healthy. 
