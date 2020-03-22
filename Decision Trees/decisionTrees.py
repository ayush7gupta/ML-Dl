import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

data = pd.read_csv('data.csv')

print(data.shape)

print(data.head())

print(data.isnull().sum())

output = data['Survived']
input = data.drop(['Survived'], axis= 1)

print(input.shape)

input_train, input_valid, output_train, output_valid = train_test_split(input, output, random_state=101, stratify= output, test_size= 0.25)

print(output_train.value_counts(normalize=True))

print(output_valid.value_counts(normalize=True))

model_dt = DecisionTreeClassifier(random_state=101)

model_dt.fit(input_train, output_train)

print("the score on tarain is:")
print(model_dt.score(input_train, output_train))

print("The score on valid is:")
print(model_dt.score(input_valid, output_valid))

output_predict = model_dt.predict(input_valid)
print(output_predict)

output_predict_prob = model_dt.predict_proba(input_valid)
print(output_predict_prob)

accuracy_score = accuracy_score(output_valid, output_predict)

print(accuracy_score)

train_accuracy =[]
valid_accuracy =[]

for x in range(1,10):
    model_dt = DecisionTreeClassifier(random_state=101, max_depth= x)
    model_dt.fit(input_train, output_train)
    train_accuracy.append(model_dt.score(input_train, output_train))
    valid_accuracy.append(model_dt.score(input_valid, output_valid))

print("printing train and valid accuracy")
print(train_accuracy)
print(valid_accuracy)

frame = pd.DataFrame({'max_depth': range(1,10), 'train_accuracy' : train_accuracy, 'valid_accuracy': valid_accuracy})
print(frame.head(5))

plt.figure(figsize=(12,6))
plt.plot(frame['max_depth'], frame['train_accuracy'], marker ='o')
plt.plot(frame['max_depth'], frame['valid_accuracy'], marker ='o')
plt.xlabel('Depth of tree')
plt.ylabel('Performance')
plt.legend
#plt.show()

model_dt = DecisionTreeClassifier(max_depth=8, max_leaf_nodes=25, random_state =101)
model_dt.fit(input_train, output_train)

decision_tree = tree.export_graphviz(model_dt, out_file='tree.dot', feature_names= input_train.columns, max_depth =8, filled= True)

#use below command to convert to png
#dot -Tpng tree.dot -o tree.png