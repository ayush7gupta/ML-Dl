import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(train_data.shape)

train_input = train_data.drop(columns=['Item_Outlet_Sales'], axis=1)
test_input = test_data.drop(columns=['Item_Outlet_Sales'], axis=1)

train_output = train_data['Item_Outlet_Sales']
test_output = test_data['Item_Outlet_Sales']

print(train_input.shape)

lr_model = LinearRegression()

lr_model.fit(train_input, train_output)
train_prediction = lr_model.predict(train_input)

#print(train_output.shape)
#print(train_prediction.shape)
rmse_train = mean_squared_error(train_output, train_prediction)**(0.5)

print("rsme on train dataset is {}", rmse_train)


test_predict = lr_model.predict(test_input)
rsme_test = mean_squared_error(test_output, test_predict)**0.5

print("rsme of test dataset is: {}", rsme_test)

pca_model = PCA( n_components= 12)

train_pca_input = pca_model.fit_transform(train_input)
test_pca_input = pca_model.fit_transform(test_input)

lr_2_model = LinearRegression()

print(train_pca_input.shape)
lr_2_model.fit(train_pca_input, train_output)
train_pca_predict = lr_2_model.predict(train_pca_input)

rsme_pca_train = mean_squared_error(train_output, train_pca_predict)**(0.5)

print("rmse with pca for train is {}", rsme_pca_train)

test_pca_predict = lr_2_model.predict(test_pca_input)

rmse_pca_test = mean_squared_error(test_output, test_pca_predict)**(0.5)

print("test rmse witch pca is {}", rmse_pca_test)
