import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print("shape of training data :", train_data.shape)
print("shape of test data:", test_data.shape)

train_input = train_data.drop(columns=['Survived'],axis=1)
train_output = train_data['Survived']

test_input = test_data.drop(columns=['Survived'],axis=1)
test_output = test_data['Survived']

#using gaussian normal naive bayes from sklearn
model = GaussianNB()

model.fit(train_input, train_output)

#predicted values
train_prediction = model.predict(train_input)

#checking accuracy
train_accuracy = accuracy_score(train_output, train_prediction)


print(train_output)
print(train_input)
print(train_accuracy)

test_prediction = model.predict(test_input)

test_accuracy = accuracy_score(test_output, test_prediction)
print(test_accuracy)
