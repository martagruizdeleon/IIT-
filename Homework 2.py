# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:00:33 2020

@author: marta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from kmodes.kmodes import KModes
import sklearn.cluster as cluster
import sklearn.neighbors as neighbors
import math
from numpy import linalg as LA

#QUESTION 1
data= pd.read_csv("Groceries.csv");
#a)
df = data.groupby('Customer')['Item'].nunique()


hist=plt.hist(df)
plt.xlabel('Number of different items')
plt.ylabel('Number of clients')
plt.savefig('hist_numitems.png')
plt.show

p_25 = np.percentile(df,25)
print('The 25th percentile is :'+str(p_25))  
p_50 = np.percentile(df, 50)
print('The 50th percentile or median is :'+str(p_50))  
p_75 = np.percentile(df, 75)
print('The 75th percentile is :'+str(p_75)) 

#b)
Nx=75;
N=df.size;
support=(Nx/N);
print('The support is:'+str(support*100))
ListItem = data.groupby(['Customer'])['Item'].apply(list).values.tolist() 
te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
trainData = pd.DataFrame(te_ary, columns=te.columns_) # Item List -> Item Indicator

frequent_itemsets = apriori(trainData, min_support = support ,use_colnames = True)
length = frequent_itemsets.support.size
print('The number of frequent itemsets is: ' , length )
print('Itemset with most items: ' , frequent_itemsets.iloc[length-1]['itemsets'])

#c)

assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)
print('The number of association rules is: ', assoc_rules['confidence'].count())

#d)
support_rule = assoc_rules['support'].to_numpy()
confidence = assoc_rules['confidence'].to_numpy()
lift = assoc_rules['lift'].to_numpy()
print('The size of the marker is:',lift.shape[0])

plt.scatter(confidence,support_rule, alpha=0.7 , s=lift)
plt.xlabel('Confidence')
plt.ylabel('Support')
plt.title('Support vs Confidence')
plt.savefig('support_confidence.png')

#e)
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.6)
print(assoc_rules)

#QUESTION 2
df_2=pd.read_csv("cars.csv")
#a)
type_freq = df_2.groupby('Type')['Type'].count()
print( 'The frequency of Types is:',type_freq)
#b)
DriveTrain_freq = df_2.groupby('DriveTrain')['DriveTrain'].count()
print( 'The frequencies of DriveTrain are:',DriveTrain_freq)

#c)
Origin_freq = df_2.groupby('Origin')['Origin'].count()
print( 'The frequencies of Origin is:',Origin_freq)
d_Asia_Europe=(1/Origin_freq['Asia'])+(1/Origin_freq['Europe'])
print('The distance between Asia and Europe is:'+str(d_Asia_Europe))

#d)

df_na = df_2.fillna("Missing")
cyl_freq = df_na.groupby('Cylinders')['Cylinders'].count()
print( 'The frequencies of Cylinder is:',cyl_freq)
d_5_missing=(1/cyl_freq[5])+(1/cyl_freq['Missing'])
print('The distance between Cylinder 5 and Missing is:'+str(d_5_missing))

#e)
df_na = df_2.fillna(0)
df1 = df_na[['Type','DriveTrain','Origin','Cylinders']]
df1["Cylinders"] = df1["Cylinders"].astype('category')
km = KModes(n_clusters=3, init='Huang')
clusters = km.fit_predict(df1)

unique, counts = np.unique(clusters, return_counts=True)
print('The number of observations in each cluster is:',counts)

# Print the cluster centroids
print(km.cluster_centroids_)
#f)
clusters=pd.DataFrame(clusters)
data = pd.concat([df1, clusters],axis=1)
data.columns = ['Type', 'DriveTrain','Origin','Cylinders','Cluster']
origin_freq = data.groupby('Cluster').Origin.value_counts()




#QUESTION 3
data_3= pd.read_csv("FourCircle.csv");
rdm_state=60616;
plt.scatter(data_3.x,data_3.y)
plt.title('Scatter plot')
plt.savefig('scatter.png')

#b)
x = np.array(data_3['x'])
y = np.array(data_3['y'])
X = np.column_stack((x,y))
kmeans = cluster.KMeans(n_clusters=4, random_state=rdm_state).fit(X)
print("Cluster Assignment:", kmeans.labels_)
print("Cluster Centroid 0:", kmeans.cluster_centers_[0])
print("Cluster Centroid 1:", kmeans.cluster_centers_[1])
print("Cluster Centroid 2:", kmeans.cluster_centers_[2])
print("Cluster Centroid 3:", kmeans.cluster_centers_[3])

plt.scatter(x, y, c=kmeans.labels_.astype(float))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot before K-mean clustering')
plt.show()
plt.savefig('scatter_KMeans.png')


#c)

n_neigh = 10
kNNSpec =neighbors.NearestNeighbors(n_neighbors = n_neigh, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(X)
d3, i3 = nbrs.kneighbors(X)
print('Distance to the nearest neighbors: '+str(d3))
print('Which are the nearest neihbors: '+str(i3))

# Retrieve the distances among the observations
distObject =neighbors.DistanceMetric.get_metric('euclidean')
distances = distObject.pairwise(X)

nObs = 1440

# Create the Adjacency matrix
Adjacency = np.zeros((nObs, nObs))
for i in range(nObs):
    for j in i3[i]:
        Adjacency[i,j] = math.exp(- (distances[i][j])**2 )

# Make the Adjacency matrix symmetric
Adjacency = 0.5 * (Adjacency + Adjacency.transpose())

# Create the Degree matrix
Degree = np.zeros((nObs, nObs))
for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i,j]
    Degree[i,i] = sum

# Create the Laplacian matrix        
Lmatrix = Degree - Adjacency

# Obtain the eigenvalues and the eigenvectors of the Laplacian matrix
evals, evecs = LA.eigh(Lmatrix)

# Series plot of the smallest five eigenvalues to determine the number of clusters
sequence = np.arange(1,6,1) 
plt.plot(sequence, evals[0:5,], marker = "o")
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.xticks(sequence)
plt.grid("both")
plt.show()
print('The eigenvalues in scientific notation are:',evals[0:5,])

# Series plot of the smallest twenty eigenvalues to determine the number of neighbors
sequence = np.arange(1,21,1) 
plt.plot(sequence, evals[0:20,], marker = "o")
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.grid("both")
plt.xticks(sequence)
plt.show()


#e)
Z = evecs[:,[0,3]]
print('The eigenvectors are:',Z)
# Final KMeans solution 
kmeans_spectral = cluster.KMeans(n_clusters=4, random_state=rdm_state).fit(Z)
data_3['SpectralCluster'] = kmeans_spectral.labels_

plt.scatter(data_3['x'], data_3['y'], c = data_3['SpectralCluster'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Spectral Cluster with {neighbors} neighbors'.format(neighbors = n_neigh))
plt.savefig('Spectral_{neighbors}.png'.format(neighbors = n_neigh))
plt.grid(True)
plt.show()
