import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels import tools
from scipy import stats
from sklearn import preprocessing

#many of the comments can be uncommented. most of them are related to statical analysis.
#

dataset = pd.read_csv('data/house1.csv')

#we are going to predict the price to taking it in a different vector
Y = dataset[['price']]

# dropping the colums that wont be used for prediction
X = dataset.drop(['price', 'id', 'date'],  axis=1)

# the below command will give the info about the matrice X
#X.info()

#get first 5 entries for X
#print(X.head())

# gives all the info like mean, count, std dev, min and max
#print(X.describe())

#dropping id and date from dataset to visualize it
dataset = dataset.drop(['id', 'date'], axis=1)

# finding the corellation matrix for the whole data minus id and date
#The three methods that can be used are
#pearson : standard correlation coefficient
#kendall : Kendall Tau correlation coefficient
#spearman : Spearman rank correlation
corr = (dataset.corr(method='pearson'))

#writing the data frame to a file
corr.to_csv("corr", encoding='utf-8', index=False)


#plotting the heatmap for the corellation
plt.subplots(figsize=(10,10))
sns.heatmap(dataset.corr())
#plt.show()

#below function will add a column of 1
X_new = tools.add_constant(X)
# a new column with name constant is added in the beginning
#print(X_new)

#linear regression using the OLS(ordinary least square) error
#first argument is the output, second one is data
result = sm.OLS(endog = Y,exog =  X_new).fit()

#print(result.summary())
#from below print it is visible that data having high co relation with output will generally have a higher beta value(param)
#print(result.params)

#we have picked sqft area as it was having the highest co-relation

new_x = X[['sqft_living']]
new_y = Y
#print(new_x)
#
# plt.figure(figsize=(10, 10))
# plt.scatter(new_x, new_y, marker='x', color ='red')
# plt.xlabel('square feet')
# plt.ylabel('price')
# plt.show()


#manual Regression, using the gradient descent to update the beta
#using the squared error
# prediction given by :
# y = mx + c
# where m = theta1
#c= theta0

#converting to numpy arrays for easier calculations
#print(new_x)
#print(new_y)

x = new_x.values.reshape(-1,1)
y = new_y.values.reshape(-1,1)
#x = preprocessing.normalize(x)
#y = preprocessing.normalize(y)
np.savetxt("x", x)
np.savetxt("y", y)

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom

x = scale(x, -1, 1)
y= scale(y, -1, 1)
#print(x.shape)
#print(y.shape)

#x= np.ones((21613,1))
#y= np.ones((21613,1))*3

#adding a colums of 1
x = np.concatenate((np.ones(len(new_x)).reshape(-1,1), x), axis=1)
np.savetxt(r'test.csv',x,delimiter=',', fmt=('%f, %f'))
np.savetxt(r'output.csv',y,delimiter=',', fmt=('%f'))
#print(x)

#loss function is:
#loss = 1/2m sigma((yhat- y)^2)
def computeCost(x, y, theta, i):
    m = len(y)
    #print(x.shape)
    #print(theta.shape)
    cost = np.dot(x, theta)
    total_cost =0
    total_cost = np.sum(np.square(cost - y))/(2*m)
    #print(cost)
    #print(i,total_cost)
    return total_cost

#alpha is the learning rate
#x is input
# y is output
#theta is the learning parametres
# iterations means number os updates to be done
#keeping an array which will have all the costs
j = []
def GradientDescent(x, y, theta, alpha, iteration):
    print("Started gd")
    for i in range(0,iteration):
        total_cost = computeCost(x, y, theta,i)
        j.append(total_cost)
        m = len(y)
        h_x = np.dot(x, theta)
        if(total_cost < 0.1):
            break
        theta[1] = theta[1] + alpha/m *(np.sum( np.dot(x[:,1].T, y-h_x)))
        theta[0] = theta[0] + alpha/m *(np.sum(y - h_x))
        #print(np.sum(y-h_x),theta[0], theta[1])
        print(np.sum(np.dot(x[:,1].T, y)), theta[0], theta[1])
    return theta, j



theta = np.zeros((2,1))
iteration = 1000000
alpha = 0.001


theta, cost = GradientDescent(x, y, theta, alpha, iteration)
print('Theta found by Gradient Descent: intercept = {} and slope {}'.format(theta[0], theta[1]))
print(j)


#just for sake of learning the below code is added
'''
#Regression using the libraries
sns.set(color_codes=True)
print("hello")
data = dataset[['price','sqft_living']]
print(data.shape)
slope, intercept, r_value, p_value, std_err = stats.linregress(data['sqft_living'],data['price'])
f = plt.figure(figsize=(10,6))

ax = sns.regplot (y='price', x='sqft_living',data=data,
                 scatter_kws={"color": "g"},
                line_kws={'color': 'b', 'label':"y={0:.1f}x+{1:.1f}".format(slope,intercept)})
#ax.legend()
plt.show()
print(std_err)
print(slope, intercept)
'''