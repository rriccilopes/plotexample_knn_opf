# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 13:31:13 2022

@author: rriccilopes
"""

# Import libs
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# Define colors per class
colors = {True:'green', False:'red'}

# Load data. Could be using a function to load a CSV file.
# Eg: pd.read_csv('data.csv')

data = pd.DataFrame({'Age': [48, 55, 50, 53, 47],
                     'BMI': [23, 28, 27.5, 25, 25],
                     'Recovered': [True, False, True, False, True]
                     })

# Plot the data points
plt.figure(figsize=(8, 6))
plt.scatter(data.Age, data.BMI, color=data.Recovered.map(colors), s=200)
plt.xlabel('Age')
plt.ylabel('BMI')

# Compute euclidean distance between points
def compute_distance(x1, x2, y1, y2):
    return ((x2-x1)**2 + (y2-y1)**2)**0.5

# Task 1: For each point, compute distances to all others
# For each point
## For each other point
### Calculate distance

# Convert to array
data_array = data.values

# Create distance matrix to store distances (distance from 4 samples to 4 samples)
distance_matrix = np.zeros((5, 5))

# Plot the data points
plt.figure(figsize=(8, 6))
plt.scatter(data.Age, data.BMI, color=data.Recovered.map(colors), s=200)
plt.xlabel('Age')
plt.ylabel('BMI')

# For each sample
for index1, sample1 in enumerate(data_array):
    sample1_age, sample1_bmi = sample1[0], sample1[1]
    
    # Compute distance with each sample
    for index2, sample2 in enumerate(data_array):
        sample2_age, sample2_bmi = sample2[0], sample2[1]
        
        # Compute distance and store
        distance = compute_distance(sample1_age, sample2_age,
                                    sample1_bmi, sample2_bmi)
        distance = np.round(distance, 1)
        distance_matrix[index1, index2] = distance

        # Only plot lines and distance if not plotted yet
        if index1 < index2:
            plt.plot((sample1_age, sample2_age), (sample1_bmi, sample2_bmi), '--', color='black')
            plt.annotate(distance,
                         xy=((sample1_age+sample2_age)/2, (sample1_bmi + sample2_bmi)/2),
                         xycoords='data',
   xytext=(0.5, 3.5), textcoords='offset points')
print(distance_matrix)
plt.show()

#%%
# Plot the data points
plt.figure(figsize=(8, 6))
plt.scatter(data.Age, data.BMI, color=data.Recovered.map(colors), s=200)
plt.xlabel('Age')
plt.ylabel('BMI')

# For each sample
for index1, sample1 in enumerate(data_array):
    sample1_age, sample1_bmi = sample1[0], sample1[1]
    
    # Compute distance with each sample
    for index2, sample2 in enumerate(data_array):
        sample2_age, sample2_bmi = sample2[0], sample2[1]
        
        # Compute distance and store
        distance = compute_distance(sample1_age, sample2_age,
                                    sample1_bmi, sample2_bmi)
        distance = np.round(distance, 1)
        distance_matrix[index1, index2] = distance

        # Only plot lines and distance if not plotted yet
        if index1 < index2:
            plt.plot((sample1_age, sample2_age), (sample1_bmi, sample2_bmi), '--', color='black')
            plt.annotate(distance,
                         xy=((sample1_age+sample2_age)/2, (sample1_bmi + sample2_bmi)/2),
                         xycoords='data',
   xytext=(0.5, 3.5), textcoords='offset points')

# Add new data to predict using KNN
plt.scatter(54, 24.5, color='blue', s=200)
plt.show()

s2 = pd.Series([54, 24.5, False], index=['Age', 'BMI', 'Recovered'])
data = pd.concat([data, s2.to_frame().T], ignore_index=True)
data_array = data.values
# %% Knn predict new sample

# Plot the data points
plt.figure(figsize=(8, 6))
plt.scatter(data.Age, data.BMI, color=data.Recovered.map(colors), s=200)
plt.xlabel('Age')
plt.ylabel('BMI')

distance_matrix = np.zeros((data.shape[0], data.shape[0]))

# For each sample
for index1, sample1 in enumerate(data_array):
    sample1_age, sample1_bmi = sample1[0], sample1[1]
    
    # Compute distance with each sample
    for index2, sample2 in enumerate(data_array):
        sample2_age, sample2_bmi = sample2[0], sample2[1]
        
        # Compute distance and store
        distance = compute_distance(sample1_age, sample2_age,
                                    sample1_bmi, sample2_bmi)
        distance = np.round(distance, 1)
        distance_matrix[index1, index2] = distance

        # Plot the lines for the new sample
        if index1 == 5 and index1 != index2:
            plt.plot((sample1_age, sample2_age), (sample1_bmi, sample2_bmi), '--', color='black')
            plt.annotate(distance,
                         xy=((sample1_age+sample2_age)/2, (sample1_bmi + sample2_bmi)/2),
                         xycoords='data',
   xytext=(0.5, 3.5), textcoords='offset points')
print(distance_matrix)
plt.show()
#%% OPF
data = pd.DataFrame({'Age': [48, 55, 50, 53, 47],
                     'BMI': [23, 28, 27.5, 25, 25],
                     'Recovered': [True, False, True, False, True]
                     })
# Convert to array
data_array = data.values

# Create distance matrix to store distances
distance_matrix = np.zeros((data.shape[0], data.shape[0]))

# For each sample
for index1, sample1 in enumerate(data_array):
    sample1_age, sample1_bmi = sample1[0], sample1[1]
    
    # Compute distance with each sample
    for index2, sample2 in enumerate(data_array):
        sample2_age, sample2_bmi = sample2[0], sample2[1]
        
        # Compute distance and store
        distance = compute_distance(sample1_age, sample2_age,
                                    sample1_bmi, sample2_bmi)
        distance = np.round(distance, 1)
        distance_matrix[index1, index2] = distance

#%% Prim's Algorithm in Python

INF = 9999999
# number of vertices in graph
N = 5
#creating graph by adjacency matrix method

selected_node = [0, 0, 0, 0, 0]

no_edge = 0

selected_node[0] = True

# printing for edge and weight
print("Edge : Weight\n")
while (no_edge < N - 1):
    
    minimum = INF
    a = 0
    b = 0
    for m in range(N):
        if selected_node[m]:
            for n in range(N):
                if ((not selected_node[n]) and distance_matrix[m, n]):  
                    # not in selected and there is an edge
                    if minimum > distance_matrix[m, n]:
                        minimum = distance_matrix[m, n]
                        a = m
                        b = n
    print(str(a) + "-" + str(b) + ":" + str(distance_matrix[a, b]))
    selected_node[b] = True
    no_edge += 1
    
# %% OPF with prototypes
# Plot the data points
plt.figure(figsize=(8, 6))
plt.scatter(data.Age, data.BMI, color=data.Recovered.map(colors), s=200)
plt.xlabel('Age')
plt.ylabel('BMI')

# For each sample
for index1, sample1 in enumerate(data_array):
    sample1_age, sample1_bmi = sample1[0], sample1[1]
    
    # Compute distance with each sample
    for index2, sample2 in enumerate(data_array):
        sample2_age, sample2_bmi = sample2[0], sample2[1]
        
        # Compute distance and store
        distance = compute_distance(sample1_age, sample2_age,
                                    sample1_bmi, sample2_bmi)
        distance = np.round(distance, 1)
        distance_matrix[index1, index2] = distance

        # Plot MST
        if ((index1 == 0 and index2 == 4) or 
           (index1 == 4 and index2 == 2) or
           (index1 == 2 and index2 == 3) or
           (index1 == 3 and index2 == 1)): 
            plt.plot((sample1_age, sample2_age), (sample1_bmi, sample2_bmi), '--', color='black')
            plt.annotate(distance,
                         xy=(sample1_age, sample1_bmi),
                         xycoords='data',
   xytext=(0.5, 3.5), textcoords='offset points')
print(distance_matrix)
plt.show()


#%%
plt.figure(figsize=(8, 6))
plt.scatter(data.Age, data.BMI, color=data.Recovered.map(colors), s=200)
plt.xlabel('Age')
plt.ylabel('BMI')
# For each sample
for index1, sample1 in enumerate(data_array):
    sample1_age, sample1_bmi = sample1[0], sample1[1]
    
    # Compute distance with each sample
    for index2, sample2 in enumerate(data_array):
        sample2_age, sample2_bmi = sample2[0], sample2[1]
        
        # Compute distance and store
        distance = compute_distance(sample1_age, sample2_age,
                                    sample1_bmi, sample2_bmi)
        distance = np.round(distance, 1)
        distance_matrix[index1, index2] = distance

        # Plot MST
        if ((index1 == 0 and index2 == 4) or 
           (index1 == 4 and index2 == 2) or
          #(index1 == 2 and index2 == 3) or
           (index1 == 3 and index2 == 1)): 
            plt.plot((sample1_age, sample2_age), (sample1_bmi, sample2_bmi), '--', color='black')
            if index1 == 0 or index1==1 or index1==4:
                plt.annotate(distance,
                             xy=(sample1_age, sample1_bmi),
                             xycoords='data',
                             xytext=(0.5, 3.5), textcoords='offset points')
            else:
                plt.annotate(distance,
                             xy=(sample2_age, sample2_bmi),
                             xycoords='data',
                             xytext=(0.5, 3.5), textcoords='offset points')

print(distance_matrix)
plt.show()
# %% OPF predict sample

plt.figure(figsize=(8, 6))
plt.scatter(data.Age, data.BMI, color=data.Recovered.map(colors), s=200)
plt.xlabel('Age')
plt.ylabel('BMI')
# For each sample
for index1, sample1 in enumerate(data_array):
    sample1_age, sample1_bmi = sample1[0], sample1[1]
    
    # Compute distance with each sample
    for index2, sample2 in enumerate(data_array):
        sample2_age, sample2_bmi = sample2[0], sample2[1]
        
        # Compute distance and store
        distance = compute_distance(sample1_age, sample2_age,
                                    sample1_bmi, sample2_bmi)
        distance = np.round(distance, 1)
        distance_matrix[index1, index2] = distance

        # Plot MST
        if ((index1 == 0 and index2 == 4) or 
           (index1 == 4 and index2 == 2) or
          #(index1 == 2 and index2 == 3) or
           (index1 == 3 and index2 == 1)): 
            #plt.plot((sample1_age, sample2_age), (sample1_bmi, sample2_bmi), '--', color='black')
            if index1 == 0 or index1==1 or index1==4:
                plt.annotate(distance,
                             xy=(sample1_age, sample1_bmi),
                             xycoords='data',
                             xytext=(0.5, 3.5), textcoords='offset points')
            else:
                plt.annotate(distance,
                             xy=(sample2_age, sample2_bmi),
                             xycoords='data',
                             xytext=(0.5, 3.5), textcoords='offset points')


test_age, test_bmi = 54, 24.5
for index1, sample1 in enumerate(data_array):
    sample1_age, sample1_bmi = sample1[0], sample1[1]
    distance = compute_distance(sample1_age, test_age,
                                    sample1_bmi, test_bmi)
    distance = np.round(distance, 1)
    plt.plot((sample1_age, test_age), (sample1_bmi, test_bmi), '--', color='black')
    plt.annotate(distance,
                xy=((sample1_age+test_age)/2, (sample1_bmi + test_bmi)/2),
                xycoords='data',
                xytext=(0.5, 3.5), textcoords='offset points')
plt.scatter(test_age, test_bmi, color='blue', s=200)

plt.show()

#%%
plt.figure(figsize=(8, 6))
plt.scatter(data.Age, data.BMI, color=data.Recovered.map(colors), s=200)
plt.xlabel('Age')
plt.ylabel('BMI')
# For each sample
for index1, sample1 in enumerate(data_array):
    sample1_age, sample1_bmi = sample1[0], sample1[1]
    
    # Compute distance with each sample
    for index2, sample2 in enumerate(data_array):
        sample2_age, sample2_bmi = sample2[0], sample2[1]

test_age, test_bmi = 54, 24.5
sample1_age, sample1_bmi = data_array[3, 0], data_array[3, 1]
distance = compute_distance(sample1_age, test_age,
                                sample1_bmi, test_bmi)
distance = np.round(distance, 1)
plt.plot((sample1_age, test_age), (sample1_bmi, test_bmi), '--', color='black')
plt.annotate(distance,
            xy=((sample1_age+test_age)/2, (sample1_bmi + test_bmi)/2),
            xycoords='data',
            xytext=(0.5, 3.5), textcoords='offset points')
plt.scatter(test_age, test_bmi, color='red', s=200)

plt.show()
