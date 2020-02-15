# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 14:19:59 2020

@author: marta
"""
import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)

#QUESTION 1
# We first order the data by values of x
normal_sample_data = pd.read_csv('NormalSample.csv');
number_samples = normal_sample_data.shape[0]
data = normal_sample_data.sort_values(by= 'x')
index = range(0,number_samples)
data['Index_new'] = index
data_ordered = data.set_index('Index_new')

Q = data_ordered.quantile([0.25,0.5,0.75])['x']
Q1 = Q.values[0]
Q3 = Q.values[2]
IQR=Q3-Q1
print(Q)
#The width of the bin is 
h=2*IQR*(number_samples)**(-1/3)
print('The bin-width is:'+str(h))

#The minimun and maximun values of x are:
min= 26.3
max= 35.4

#Values of a and b
a=26
b=36
#We calculate the mid points
def midPoints(h,min,max):
    mid_points=[];
    mid=min+(h/2);
    mid_points=np.append(mid_points,mid);
    while mid<max:
        mid=mid+h;
        if mid<max:
            mid_points=np.append(mid_points,mid);
    
    return mid_points
x=data_ordered['x']
#We calculate the density
def denEstimate(midPoints,x,N,h):
    density = [];
    x_values = x.values
    k = 1/(N*h)
   
    for i in midPoints:
        w = 0
        for j in x_values:
            u = (j-i)/h
            if u < 0.5 and u > -0.5:
                w = w+1;
            
        est = w * k
        density.append(est)
        
    return (density)

h=[0.25,0.5,1,2]
for hi in h:
    midpoints=midPoints(hi,a,b)
    density=denEstimate(midpoints,x,number_samples,hi)
    
    fig, ax = plt.subplots()
    ax.bar(x=midpoints,height=density,width=hi)
    fig.savefig('hist_h_{}.png'.format(hi))
    plt.show()
    
#QUESTION 2

lower_whisker=Q1-1.5*IQR
upper_whisker=Q3+1.5*IQR
print('lower_whisker='+ str(lower_whisker))
print('upper_whisker='+str(upper_whisker))

#We separate the data in groups of 1 and 0
one_data=data_ordered.where(data_ordered['group']==1)
one_data=one_data.dropna()
zero_data=data_ordered.where(data_ordered['group']==0)
zero_data=zero_data.dropna()
Quant_one = one_data.quantile([0.25,0.5,0.75])['x']
Quant_zero = zero_data.quantile([0.25,0.5,0.75])['x']
min_one=29.1
max_one=35.4
min_zero=26.3
max_zero=32.2
IQR_one=Quant_one.values[2]-Quant_one.values[0]
IQR_zero=Quant_zero.values[2]-Quant_zero.values[0]

lower_whisker_one=(Quant_one.values[0]-1.5*IQR_one)
upper_whisker_one=(Quant_one.values[2]+1.5*IQR_one)
print('lower_whisker_one='+ str(lower_whisker_one))
print('upper_whisker_one='+str(upper_whisker_one))   

lower_whisker_zero=(Quant_zero.values[0]-1.5*IQR_zero)
upper_whisker_zero=(Quant_zero.values[2]+1.5*IQR_zero)
print('lower_whisker_zero='+ str(lower_whisker_zero))
print('upper_whisker_zero='+str(upper_whisker_zero))   


#We draw the boxplot for the whole data 
boxplot = data_ordered.boxplot(column=['x'])

#We draw the boxplot for the whole x , and the two groups
data_ordered['x_one'] = one_data['x'];
data_ordered['x_zero']=zero_data['x'];

boxplot = data_ordered.boxplot(column = ['x' , 'x_one' , 'x_zero']  )

boxplot.figure.savefig('boxplot_1.png')

#We want to see if there are any outliers 
#Outliers for the whole x
data_ordered = data.set_index('Index_new')
outliers_x_down=data_ordered.where((data_ordered['x']<27.4))
outliers_x_up=data_ordered.where(data_ordered['x']>35.7)
outliers_x_down=outliers_x_down.dropna()
outliers_x_up=outliers_x_up.dropna()
print('Outliers in total data set='+str(outliers_x_down))
print('Outliers in total data set='+str(outliers_x_up))

#Outliers in the group 0
outliers_zero_down=zero_data.where((zero_data['x']<27.59))
outliers_zero_up=zero_data.where(zero_data['x']>32.4)
outliers_zero_down=outliers_zero_down.dropna()
outliers_zero_up=outliers_zero_up.dropna()
print('Outliers in zero group ='+str(outliers_zero_down))
print('Outliers in zero group='+str(outliers_zero_up))

#Outliers in group 1
outliers_one_down=one_data.where((one_data['x']<29.45))
outliers_one_up=one_data.where(one_data['x']>34.65)
outliers_one_down=outliers_one_down.dropna()
outliers_one_up=outliers_one_up.dropna()
print('Outliers in one group ='+str(outliers_one_down))
print('Outliers in one group='+str(outliers_one_up))


#QUESTION 3 

#We import the data 
fraud_data = pd.read_csv('Fraud.csv')

#detection of fraud
fraud=(fraud_data.loc[fraud_data['FRAUD'] == 1])
fraud_num=fraud['FRAUD'].count()
 
fraud_perc=(fraud_num/fraud_data['FRAUD'].count())*100
print('The fraud percentage is:'+str(fraud_perc))

#Boxplot for each interval variable
#Boxplot of Total Spend
Total_spend = fraud_data.boxplot(column = ['TOTAL_SPEND'], by='FRAUD',vert=False  )
Total_spend.set_title("Total Spend")
plt.suptitle('')
plt.xlabel("")
plt.show()
Total_spend.figure.savefig('Total_spend.png')
doc_visits=fraud_data.boxplot( column=['DOCTOR_VISITS'],by= 'FRAUD', vert=False)
doc_visits.set_title("Doctor Visits")
plt.suptitle('')
plt.xlabel("")
plt.show()
doc_visits.figure.savefig('doc_visits.png')
num_claims=fraud_data.boxplot(column=['NUM_CLAIMS'], by='FRAUD', vert=False)
num_claims.set_title("Number of Claims")
plt.suptitle('')
plt.xlabel("")
plt.show()
num_claims.figure.savefig('num_claims.png')
mem_dur=fraud_data.boxplot(column=['MEMBER_DURATION'], by= 'FRAUD', vert=False)
mem_dur.set_title("Member Duration")
plt.suptitle('')
plt.xlabel("")
plt.show()
mem_dur.figure.savefig('mem_dur.png')
optm_presc=fraud_data.boxplot(column=['OPTOM_PRESC'], by= 'FRAUD', vert= False)
optm_presc.set_title("Optom Presc")
plt.suptitle('')
plt.xlabel("")
plt.show()
optm_presc.figure.savefig('optm_presc.png')
num_members=fraud_data.boxplot(column=['NUM_MEMBERS'], by= 'FRAUD', vert= False)
num_members.set_title("Number of Members")
plt.suptitle('')
plt.xlabel("")
plt.show()
num_members.figure.savefig('num_members.png')

#Orthonormalize interval variables 
x=np.matrix(fraud_data.drop(columns=["CASE_ID","FRAUD"]))
print("Number of Dimensions = ", x.ndim)
y=np.matrix(pd.DataFrame((fraud_data['FRAUD'])))
#We get the eigenvalues 
xtx = x.transpose() * x
# Eigenvalue decomposition
evals, evecs = LA.eigh(xtx)
print("Eigenvalues of x = \n", evals)
print("Eigenvectors of x = \n",evecs)

# Here is the transformation matrix
transf = evecs * LA.inv(np.sqrt(np.diagflat(evals)))
print("Transformation Matrix = \n", transf)

# Here is the transformed X
transf_x = x * transf
print("The Transformed x = \n", transf_x)

# Check columns of transformed X
xtx = transf_x.transpose() * transf_x
print("Expect an Identity Matrix = \n", xtx)
print("Actual X = \n", transf_x)
y=np.ravel(y)
#Use nearneighbours using k=5
nbrs = NearestNeighbors(n_neighbors = 5 , algorithm = 'brute' , metric = 'euclidean').fit(transf_x)
distances, indices = nbrs.kneighbors(transf_x)
print(distances)
print(indices)
neigh = KNeighborsClassifier(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
neigh.fit(transf_x,y)
print(neigh.score(transf_x,y))

#For the observation which has these input variable values: TOTAL_SPEND = 7500, DOCTOR_VISITS = 15, 
#NUM_CLAIMS = 3, MEMBER_DURATION = 127, OPTOM_PRESC = 2, and NUM_MEMBERS = 2, find its five neighbors

obs = [7500 , 15 , 3 , 127 , 2 , 2];

obs = obs * transf
print(obs)
myneigh =nbrs.kneighbors(obs , return_distance = False)

print("my neighbors are: \n" , myneigh)
id= myneigh[0]+1
neigh_one=fraud_data.loc[fraud_data['CASE_ID'] == id[0]]
print("First neighbour:" , neigh_one)
neigh_two=fraud_data.loc[fraud_data['CASE_ID'] == id[1]]
print("Second neighbour:" , neigh_two)
neigh_three=fraud_data.loc[fraud_data['CASE_ID'] == id[2]]
print("Thrird neighbour:" , neigh_three)
neigh_four=fraud_data.loc[fraud_data['CASE_ID'] == id[3]]
print("Fourth neighbour:" , neigh_four)
neigh_five=fraud_data.loc[fraud_data['CASE_ID'] == id[4]]
print("Five neighbour:" , neigh_five)

frames = [neigh_one, neigh_two, neigh_three,neigh_four,neigh_five]

result = pd.concat(frames)
#Predicted probability 
y_pred = neigh.predict(obs)
print('The prediction will be:'+str(y_pred))
