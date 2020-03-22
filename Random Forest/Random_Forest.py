import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_input = train_data.drop(columns=['Survived'], axis =1)
train_output = train_data['Survived']

test_input = test_data.drop(columns=['Survived'], axis =1)
tetst_output = test_data['Survived']

random_forest_model = RandomForestClassifier()
random_forest_model.fit(train_input, train_output)

print("no. of trees used are:{}", random_forest_model.n_estimators)

train_predict = random_forest_model.predict(train_input)
train_acc = accuracy_score(train_output, train_predict)

print("The accuracy on train is {}", train_acc)

test_predict = random_forest_model.predict(test_input)
test_acc = accuracy_score(tetst_output, test_predict)

print("The accuracy on predict is {}", test_acc)