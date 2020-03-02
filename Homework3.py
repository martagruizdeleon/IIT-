# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:57:20 2020

@author: marta
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
from itertools import combinations
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
import matplotlib.pyplot as plt




#QUESTION 1
data = pd.read_csv('claim_history.csv')
rs = 60616

#A)
data_train, data_test = train_test_split( data , test_size=0.25, random_state=rs,stratify = data["CAR_USE"])

car_use_freq = data_train.groupby('CAR_USE')['CAR_USE'].count()
print( 'The frequency of CAR USE is:',car_use_freq)
car_use_prop=np.array([car_use_freq[0]/len(data_train),car_use_freq[1]/len(data_train)])
print('The proportion of commercial is:'+str(car_use_prop[0]))
print('The proportion of Private is:'+str(car_use_prop[1]))
print()
print('------------------------------------------------------------------------')
#B
car_use_freq_test = data_test.groupby('CAR_USE')['CAR_USE'].count()
print( 'The frequency of CAR USE is:',car_use_freq_test)
car_use_prop_test=np.array([car_use_freq_test[0]/len(data_test),car_use_freq_test[1]/len(data_test)])
print()
print('The proportion of commercial is:'+str(car_use_prop_test[0]))
print('The proportion of Private is:'+str(car_use_prop_test[1]))
print('------------------------------------------------------------------------')
#C PROPORTION OF COMMERCIAL IN TRAINING
prob_commercial=car_use_freq[0]/(car_use_freq[0]+car_use_freq_test[0])
print('The probability that a commercial observation is in the training set is:'+str(prob_commercial))
print()
print('------------------------------------------------------------------------')
#D
prob_private=car_use_freq_test[1]/(car_use_freq[1]+car_use_freq_test[1])
print('The probability that a private observation is in the test set is:'+str(prob_private))
print()
#QUESTION 2
def entropy_calc(categories,comb,num_comb,predictor,df,total_private,total_commercial,total):
    entropy_vect=[];
    for i in range(0,num_comb):
        df_comb=df.where(df[predictor].isin(comb[i]))
        df_freq=df_comb.groupby("CAR_USE").count()[predictor]
        #Entropy for branch 1
        try:
            private_b1=df_freq['Private']
        except KeyError:
            private_b1=0
        try:
            commercial_b1=df_freq['Commercial']
        except KeyError:
            commercial_b1=0
        total_comb=private_b1+commercial_b1
        if private_b1>0 and commercial_b1>0:
            e1=-(((private_b1/total_comb)*np.log2(private_b1/total_comb))+((commercial_b1/total_comb)*np.log2(commercial_b1/total_comb)))
        else:
            e1=0
        #Entropy for branch 2:
        private_b2=total_private-private_b1
        commercial_b2=total_commercial-commercial_b1
        total_b2=private_b2+commercial_b2
        if private_b2>0 and commercial_b2>0:
            e2=-(((private_b2/total_b2)*np.log2(private_b2/total_b2))+((commercial_b2/total_b2)*np.log2(commercial_b2/total_b2)))
        else:
            e2=0
        entropy_split=(total_comb/total)*e1+(total_b2/total)*e2
        entropy_vect.append(entropy_split)
        
    return entropy_vect



def best_split_nom(categories,predictor,df,total_private,total_commercial,total):
    best_split_vect=[]
    best_entropy_vect=[]
    num_categories=len(categories)
    max_iterations=round(num_categories/2+1)
    for j in range(1,max_iterations):
        comb=list(combinations(categories,j))
        num_comb=len(comb)
        entropy=entropy_calc(categories,comb,num_comb,predictor,df,total_private,total_commercial,total)
        best_entropy_vect.append(min(entropy))
        best_split_vect.append(comb[entropy.index(min(entropy))])
        
    best_entropy=min(best_entropy_vect)
    best_split=best_split_vect[best_entropy_vect.index(min(best_entropy_vect))]
    return best_entropy,best_split

def best_split_ord(categories,predictor,df,total_private,total_commercial,total):
    num_cat=len(categories)
    partition=[]
    comb=[]
    for k in range(0,num_cat-1):
        partition.append(categories[k])
        parts=partition.copy()
        comb.append(parts)
    num_comb=len(comb)
    entropy= entropy_calc(categories,comb,num_comb,predictor,df,total_private,total_commercial,total)
    best_entropy=min(entropy)
    best_split=comb[entropy.index(min(entropy))]
    
    return best_entropy,best_split

#a) ENTROPY OF ROOT NODE:
tot=data_train.groupby('CAR_USE')['CAR_USE'].count()
total_private=tot['Private']
total_commercial=tot['Commercial']
total=total_private+total_commercial
entropy_root_node=-(((total_private/total)*np.log2(total_private/total))+((total_commercial/total)*np.log2(total_commercial/total)))
print('The entropy at the root node is:'+str(entropy_root_node))
print()
#WE FIND THE BEST SPLIT FOR EACH PREDICTOR
#CAR TYPE
categories_car_type=['Minivan','Panel Truck','Pickup','SUV','Sports Car','Van']
predictor_car_type='CAR_TYPE'
best_entropy_car_type,best_split_car_type=best_split_nom(categories_car_type,predictor_car_type,data_train[[predictor_car_type,'CAR_USE']],total_private,total_commercial,total)
print('The best split is:'+str(best_split_car_type))
print('The entropy for the best split is:'+str(best_entropy_car_type))  
print()
#OCCUPATION
categories_occupation=['Blue Collar','Clerical','Doctor','Home Maker','Lawyer','Manager','Professional','Student','Unknown']
predictor_occupation='OCCUPATION'
best_entropy_occupation,best_split_occupation=best_split_nom(categories_occupation,predictor_occupation,data_train[[predictor_occupation,'CAR_USE']],total_private,total_commercial,total)
print('The best split for occupation is:'+str(best_split_occupation))
print('The entropy for the best split is:'+str(best_entropy_occupation))      
print()
#EDUCATION
categories_education=['Below High School','High School','Bachelors','Masters','Doctors']
predictor_education='EDUCATION'
best_entropy_education,best_split_education=best_split_ord(categories_education,predictor_education,data_train[[predictor_education,'CAR_USE']],total_private,total_commercial,total)
print('The best split for education is:'+str(best_split_education))
print('The entropy for the best split is:'+str(best_entropy_education))  
print()
#SELECT THE PREDICTOR WITH THE LOWEST ENTROPY:
print('The predictor of the first split is OCCUPATION')
print()
#WE ASING THE OBSERVATIONS
categories_best_split1=categories_occupation.copy()
cat_split = list(filter(lambda i: i not in best_split_occupation, categories_best_split1))
branch1=data_train[data_train['OCCUPATION'].isin(best_split_occupation)]
branch2=data_train[data_train['OCCUPATION'].isin(cat_split)]

#WE FIND THE NEXT BEST SPLIT FOR EACH PREDICTOR IN BRANCH 1
#CAR TYPE
tot11=branch1.groupby('CAR_USE')['CAR_USE'].count()
total_private1=tot11['Private']
total_commercial1=tot11['Commercial']
total1=total_private1+total_commercial1
#CAR TYPE
best_entropy_car_type1,best_split_car_type1=best_split_nom(categories_car_type,predictor_car_type,branch1[[predictor_car_type,'CAR_USE']],total_private1,total_commercial1,total1)
print('The best split of car type is:'+str(best_split_car_type1))
print('The entropy for the best split is:'+str(best_entropy_car_type1))  
print()
#OCCUPATION
best_entropy_occupation1,best_split_occupation1=best_split_nom(best_split_occupation,predictor_occupation,branch1[[predictor_occupation,'CAR_USE']],total_private1,total_commercial1,total1)
print('The best split for occupation is:'+str(best_split_occupation1))
print('The entropy for the best split is:'+str(best_entropy_occupation1))      
print()
#EDUCATION
best_entropy_education1,best_split_education1=best_split_ord(categories_education,predictor_education,branch1[[predictor_education,'CAR_USE']],total_private1,total_commercial1,total1)
print('The best split for education is:'+str(best_split_education1))
print('The entropy for the best split is:'+str(best_entropy_education1)) 
print()
print('The predictor selected for branch 1 is EDUCATION')
#WE ASIGN OBSERVATIONS
categories_best_split2=categories_education.copy()
cat_split2 = list(filter(lambda i: i not in best_split_education1, categories_best_split2))
branch11=branch1[branch1['EDUCATION'].isin(best_split_education1)]
branch12=branch1[branch1['EDUCATION'].isin(cat_split2)]

#WE FIND THE BEST SPLIT FOR EACH PREDICTOR IN BRANCH 2
tot12=branch2.groupby('CAR_USE')['CAR_USE'].count()
total_private2=tot12['Private']
total_commercial2=tot12['Commercial']
total2=total_private2+total_commercial2
#cAR_TYPE
best_entropy_car_type2,best_split_car_type2=best_split_nom(categories_car_type,predictor_car_type,branch2[[predictor_car_type,'CAR_USE']],total_private2,total_commercial2,total2)
print('The best split of car type is:'+str(best_split_car_type2))
print('The entropy for the best split is:'+str(best_entropy_car_type2))  
print()
#OCCUPATION
best_entropy_occupation2,best_split_occupation2=best_split_nom(cat_split,predictor_occupation,branch2[[predictor_occupation,'CAR_USE']],total_private2,total_commercial2,total2)
print('The best split for occupation is:'+str(best_split_occupation2))
print('The entropy for the best split is:'+str(best_entropy_occupation2))   
print()
#EDUCATION
best_entropy_education2,best_split_education2=best_split_ord(categories_education,predictor_education,branch2[[predictor_education,'CAR_USE']],total_private2,total_commercial2,total2)
print('The best split for education is:'+str(best_split_education2))
print('The entropy for the best split is:'+str(best_entropy_education2))   
print()
print('The preductor selected for the split in branch 2 is CAR TYPE')

#WE ASING OBSERVATIONS
categories_best_split3=categories_car_type.copy()
cat_split3 = list(filter(lambda i: i not in best_split_car_type2, categories_best_split3))
branch21=branch2[branch2['CAR_TYPE'].isin(best_split_car_type2)]
branch22=branch2[branch2['CAR_TYPE'].isin(cat_split3)]

#c) WE COUNT THE OBSERVATIONS IN EACH LEAF
total_branch11=branch11.groupby('CAR_USE')['CAR_USE'].count()
private_branch11=total_branch11['Private']
commercial_branch11=total_branch11['Commercial']

total_branch12=branch12.groupby('CAR_USE')['CAR_USE'].count()
private_branch12=total_branch12['Private']
commercial_branch12=total_branch12['Commercial']

total_branch21=branch21.groupby('CAR_USE')['CAR_USE'].count()
private_branch21=total_branch21['Private']
commercial_branch21=total_branch21['Commercial']

total_branch22=branch22.groupby('CAR_USE')['CAR_USE'].count()
private_branch22=total_branch22['Private']
commercial_branch22=total_branch22['Commercial']
print('------------------------------------------------------------------------')
print()
#Kolmogorov-smirnov
# f) Kolmogorov-Smirnov statistic and the event probability cutoff value in training

# Predicted event probabilities of the training data.
leaf1_prob=commercial_branch11/(commercial_branch11+private_branch11)
print('The probability of commercial in leaf 1 is:'+str(leaf1_prob))
print()
leaf2_prob=commercial_branch12/(commercial_branch12+private_branch12)
print('The probability of commercial in leaf 2 is:'+str(leaf2_prob))
print()
leaf3_prob=commercial_branch21/(commercial_branch21+private_branch21)
print('The probability of commercial in leaf 3 is:'+str(leaf3_prob))
print()
leaf4_prob=commercial_branch22/(commercial_branch22+private_branch22)
print('The probability of commercial in leaf 4 is:'+str(leaf4_prob))
print()
#We asign observations of the test set to each of the leaves:

train_leaf1=branch11
train_leaf2=branch12
train_leaf3=branch21
train_leaf4=branch22

#We classify each of the observations in each leaf 
train_leaf1['y_true'] = np.where(train_leaf1['CAR_USE']=='Commercial', 1, 0)
train_leaf2['y_true'] = np.where(train_leaf2['CAR_USE']=='Commercial', 1, 0)
train_leaf4['y_true'] = np.where(train_leaf4['CAR_USE']=='Commercial', 1, 0)
train_leaf3['y_true'] = np.where(train_leaf3['CAR_USE']=='Commercial', 1, 0)

train_leaf1['y_pred']=0
train_leaf2['y_pred']=1
train_leaf3['y_pred']=0
train_leaf4['y_pred']=1

train_leaf1['y_pred_prob']=leaf1_prob
train_leaf2['y_pred_prob']=leaf2_prob
train_leaf3['y_pred_prob']=leaf3_prob
train_leaf4['y_pred_prob']=leaf4_prob
y_true_train = [train_leaf1['y_true'], train_leaf2['y_true'], train_leaf3['y_true'],train_leaf4['y_true']]
y_true_train=pd.concat(y_true_train)
y_pred_train= [train_leaf1['y_pred'], train_leaf2['y_pred'], train_leaf3['y_pred'],train_leaf4['y_pred']]
y_pred_train=pd.concat(y_pred_train)
y_pred_prob_train = [train_leaf1['y_pred_prob'], train_leaf2['y_pred_prob'], train_leaf3['y_pred_prob'],train_leaf4['y_pred_prob']]
y_pred_prob_train=pd.concat(y_pred_prob_train)
y_real_train = [train_leaf1['CAR_USE'], train_leaf2['CAR_USE'], train_leaf3['CAR_USE'],train_leaf4['CAR_USE']]
y_real_train=pd.concat(y_real_train)

fpr, tpr, thresholds = metrics.roc_curve(y_true_train, y_pred_prob_train, pos_label = 1)
print('------------------------------------------------------------------------')
# Kolmogorov Smirnov graph
print('f) Kolmogorov Smirnov')
# Draw the Kolmogorov Smirnov curve
cutoff = np.where(thresholds > 1.0, np.nan, thresholds)
plt.plot(cutoff, tpr, marker = 'o',
         label = 'True Positive',
         color = 'blue', linestyle = 'solid')
plt.plot(cutoff, fpr, marker = 'o',
         label = 'False Positive',
         color = 'red', linestyle = 'solid')
plt.grid(True)
plt.xlabel("Probability Threshold")
plt.ylabel("Positive Rate")
plt.legend(loc = 'upper right', shadow = True)
plt.show()

dist=[]
for p in range(0,5):
    distance=tpr[p]-fpr[p]
    dist.append(distance)
print('The maximun distance is at the point',dist.index(max(dist)))
print()
KS_stat = tpr[2] - fpr[2]
print('The KS statistic is:', KS_stat)

thr_train = cutoff[2]
print('The cutoff value is:', thr_train)

print('-----------------------------------------------------------------------')
print()


print()
#QUESTION 3

#We asign observations of the test set to each of the leaves:
test_one=data_test[data_test['OCCUPATION'].isin(best_split_occupation)]
test_leaf1=test_one[test_one['EDUCATION'].isin(best_split_education1)]

test_two=data_test[data_test['OCCUPATION'].isin(best_split_occupation)]
test_leaf2=test_two[test_two['EDUCATION'].isin(cat_split2)]

test_three=data_test[data_test['OCCUPATION'].isin(cat_split)]
test_leaf3=test_three[test_three['CAR_TYPE'].isin(best_split_car_type2)]

test_four=data_test[data_test['OCCUPATION'].isin(cat_split)]
test_leaf4=test_four[test_four['CAR_TYPE'].isin(cat_split3)]

#We classify each of the observations in each leaf 
test_leaf1['y_true'] = np.where(test_leaf1['CAR_USE']=='Commercial', 1, 0)
test_leaf2['y_true'] = np.where(test_leaf2['CAR_USE']=='Commercial', 1, 0)
test_leaf4['y_true'] = np.where(test_leaf4['CAR_USE']=='Commercial', 1, 0)
test_leaf3['y_true'] = np.where(test_leaf3['CAR_USE']=='Commercial', 1, 0)

test_leaf1['y_pred']=0
test_leaf2['y_pred']=1
test_leaf3['y_pred']=0
test_leaf4['y_pred']=1

test_leaf1['y_pred_prob']=leaf1_prob
test_leaf2['y_pred_prob']=leaf2_prob
test_leaf3['y_pred_prob']=leaf3_prob
test_leaf4['y_pred_prob']=leaf4_prob



y_true = [test_leaf1['y_true'], test_leaf2['y_true'], test_leaf3['y_true'],test_leaf4['y_true']]
y_true=pd.concat(y_true)
y_pred= [test_leaf1['y_pred'], test_leaf2['y_pred'], test_leaf3['y_pred'],test_leaf4['y_pred']]
y_pred=pd.concat(y_pred)
y_pred_prob = [test_leaf1['y_pred_prob'], test_leaf2['y_pred_prob'], test_leaf3['y_pred_prob'],test_leaf4['y_pred_prob']]
y_pred_prob=pd.concat(y_pred_prob)
y_real = [test_leaf1['CAR_USE'], test_leaf2['CAR_USE'], test_leaf3['CAR_USE'],test_leaf4['CAR_USE']]
y_real=pd.concat(y_real)
accuracy=accuracy_score(y_true, y_pred)
print('The misclassification rate is:',1-accuracy)
#b
print('-----------------------------------------------------------------------')
print()
print('Since the cutoff value is ',thr_train,'the predicted categories will be the same')
print('The accuracy thus will be the same as before')

print('------------------------------------------------------------------------')
#c Average root averaged squared error
ASE = metrics.mean_squared_error(y_true,y_pred)
RASE = math.sqrt(ASE)
print()
print("The Root Average Squared Error is: ", RASE)
print('------------------------------------------------------------------------')
#d Area under the curve
AUC = metrics.roc_auc_score(y_true,y_pred_prob)
print()
print("The Area Under the Curve is: " , AUC)
print()
print('------------------------------------------------------------------------')
#e gini indix
Gini = (2 * AUC) - 1
print('The gini index is:'+str(Gini))
print('------------------------------------------------------------------------')
#g Generate the Receiver Operating Characteristic curve for the Test partition
df_gamma=pd.DataFrame(columns=['y_real', 'y_pred_prob'])
df_gamma['y_pred_prob']=y_pred_prob
df_gamma ['y_real']= y_real

# group the predicted probabilities in 2 groups.
df_gamma = df_gamma.sort_values(by=['y_real'])
df_gamma_0 = df_gamma[df_gamma['y_real']=='Private']
df_gamma_1 = df_gamma[df_gamma['y_real']=='Commercial']

# Sort the predicted probabilities in ascending order within each group 
df_gamma_0 = df_gamma_0.sort_values(by=['y_pred_prob'])
df_gamma_1 = df_gamma_1.sort_values(by=['y_pred_prob'])
df_gamma_0 = np.array(df_gamma_0['y_pred_prob'])
df_gamma_1 = np.array(df_gamma_1['y_pred_prob'])

# Table of Concordant (C), Discordant (D), and Tied (T) pairs
C = 0
D = 0
T = 0
for i in range(0,len(df_gamma_1)):
    for j in range(0,len(df_gamma_0)):
        if df_gamma_1[i] > df_gamma_0[j]: C=C+1
        if df_gamma_1[i] == df_gamma_0[j]: T=T+1
        if df_gamma_1[i] < df_gamma_0[j]: D=D+1
Pairs = C+D+T

Gamma = (C-D)/(C+D)
print()
print('f) The Goodman-Kruskal Gamma statistic in the Test partition is:', Gamma)
print()
print('------------------------------------------------------------------------')
# Generate the coordinates for the ROC curve
OneMinusSpecificity, Sensitivity, thresholds = metrics.roc_curve(y_real ,y_pred_prob, pos_label = 'Commercial')

# Add two dummy coordinates
OneMinusSpecificity = np.append([0], OneMinusSpecificity)
Sensitivity = np.append([0], Sensitivity)

OneMinusSpecificity = np.append(OneMinusSpecificity, [1])
Sensitivity = np.append(Sensitivity, [1])

# Draw the ROC curve
plt.figure(figsize=(6,6))
plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
ax = plt.gca()
ax.set_aspect('equal')
plt.savefig("ROC_curve.png")
plt.show()

