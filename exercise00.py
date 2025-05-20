from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', None)
# fetch dataset
steel_plates_faults = fetch_ucirepo(id=198)

# data (as pandas dataframes)
X = steel_plates_faults.data.features
Y = steel_plates_faults.data.targets
print(X)
# metadata
print(steel_plates_faults.metadata)

# variable information
print(steel_plates_faults.variables)


x = np.array(X)
y = np.array(Y)
#print(x)
print(x.dtype)
print(x.shape)

def function1 (array):
    #array = np.atleast_2d(array)
    try:
        (r, c) = array.shape
    except ValueError:
        r = array.shape[0]
        c = 1
    try:
        min = array[0,0]
    except IndexError:
        min = array[0]
    try:
        max = array[0,0]
    except IndexError:
        max = array[0]

    if array.ndim == 1:
        r = array.shape[0]
        tempar = np.zeros((r,1))
        tempar[:,0] = array[:]
        array = tempar


    for i in range(r):
        if array[i, 0] < min:
            min = array[i,0]
        if array[i, 0] > max:
            max = array[i, 0]
    mean = array[:,0].mean()

    return min, max, mean


def function2 (array, column_names):
    (r, c) = array.shape
    L = []
    for i in range(c):
        columnname = column_names[i]
        L.append((columnname, function1(array[:,i])))
    return L

print(function2(x, X.columns))

#binary columns: TypeOfSteel_A300, TypeOfSteel_A400, Outside_Global_Index

List_of_binary_columns = ['TypeOfSteel_A300', 'TypeOfSteel_A400', 'Outside_Global_Index']

List_of_numbers = []
for column in List_of_binary_columns:
    vector = X[column]
    r = vector.shape[0]
    anzahl = 0
    for i in range(r):
        if vector[i] == 1:
            anzahl = anzahl + 1
    List_of_numbers.append([column, anzahl])
#List_of_numbers = np.array(List_of_numbers)
print(List_of_numbers)

labels = [row[0] for row in List_of_numbers]
#values = List_of_numbers[:, 1].astype(int)
values = [row[1] for row in List_of_numbers]

print('lists of lists')
print(labels)
print(values)


plt.figure(figsize = (10,10))
plt.bar(labels, values)
#plt.ylim(0,1200)
plt.xticks(fontsize = 10, rotation = 45)
plt.xlabel('Binary data')
plt.tight_layout()
plt.show()

plt.figure(figsize = (12, 8))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.hist(X.iloc[:,i])

plt.show()

print(Y)

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c = Y['Z_Scratch'])
plt.show()
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c = Y['Other_Faults'])
plt.title(f"{X.columns[0]} vs {X.columns[1]} colored by ...")
plt.show()





