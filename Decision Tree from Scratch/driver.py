from DecisionTree import *
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import datasets
import random

#Author: Taniya Riar
#The link was not available. Added the iris dataset from sklearn datasets library.
iris = datasets.load_iris()

#Storing the class names or the target names ['setosa' -> 0.0,'versicolor' -> 1.0,'virginica' -> 2.0] 
class_names = list(iris['target_names'])
new_dict = {0.0 : 'setosa', 1.0 : 'versicolor', 2.0 : 'virginica'}

df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class'])
header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']

#Adding Class names to the dataframe
df['Class'] = df['Class'].map(new_dict)

#The link not working
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])

lst = df.values.tolist()

t = build_tree(lst, header)
print_tree(t)

print("********** Leaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t)
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))
    
trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()


t = build_tree(train, header)
print("*************Tree before pruning*******")
print_tree(t)
acc = computeAccuracy(test, t)
print("Accuracy on test = " + str(acc))

## TODO: You have to decide on a pruning strategy
#Using Strategy 2 : Pruning using Random Strategy by selecting 'n' Non leaf nodes 
n = 3 #radomly choosing a number n 

#Next few steps to randomly choose n innernodes from the list of inner nodes and getting their ids
prune_nodes = []
for p_node  in random.sample(innerNodes, n):
    if p_node.id !=0:
        prune_nodes.append(p_node.id)
        
print("*************Pruned Nodes [list of ids]*******")
print(prune_nodes)

t_pruned = prune_tree(t, prune_nodes)

print("*************Tree after pruning*******")
print_tree(t_pruned)
    
acc = computeAccuracy(test, t)
print("Accuracy on test = " + str(acc))