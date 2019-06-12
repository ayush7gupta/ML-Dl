import numpy as np
from numpy import array
from numpy import genfromtxt
from csv import reader
import random

my_data = genfromtxt('sonarData.csv', delimiter=',')
data = np.delete(my_data,60, 1 )
result = my_data[:,60]
#print(my_data)
#print(my_data.shape)
#print(result[0])


#numpy not working as last column is not a number. Will have to use something else

def getcsv_to_list(filename):
    #generating an empty list
    data= list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            data.append(row)
    #here data is converted to an 2D list
    #print(data)
    return data

def convertStringtoFloat(dataset, column):
    for row in dataset:
        #print(row[column])
        row[column] = float(row[column].strip())
        #print(row[column])

def convertStringtoInt(dataset, column):
    i=0
    for row in dataset:
        #print(row[column])
        #print(i)
        if row[column] == 'M':
            row[column] = 1
        else:
            row[column] = 0
        #print(row[column])
        #i= i+1

def trainPerceptron(data_train, data_test, result_train, result_test, learning_rate = 0.15, epoch=1 ):
    weight = np.random.rand(60)
    #print(weight)
    bias = random.random()
    #Here we are training the network
    for i in range (epoch):
        number = random.randint(0, 168)
        #dot product is not possible in 1d arrays in matlab and numpy, so have to reshape it to a 2-d array
        weight = weight.reshape(-1,1)
        #print(type((weight)))
        #print(weight.shape)
        #print(data_train[number].shape)
        test_entity = data_train[number]
        #one array has to be transpose of other for successfull dot product, notice a subtle differences in 1 and -1
        test_entity = test_entity.reshape(1,-1)
        #print(type(test_entity))
        #print(test_entity.shape)
        val = test_entity.dot(weight)
        print(val)
        print(result_train[number])

        #now we will check for how many are we getting the negative result. If we get negative result we will update our weights
        if(val < 0 and result_train[number] == 1):
            #print((test_entity.T).shape)
            weight = weight + learning_rate * (test_entity.T)
            bias = bias + learning_rate
        if(val>0 and result_train[number] == 0):
            #print((test_entity.T).shape)
            #print(weight.T)
            weight = weight - learning_rate * (test_entity.T)
            #print(weight.T)
            bias = bias - learning_rate

    #now we will check the accuracy
    count = 0
    for i in range(len(result_test)):
        test_entity = data_test[i]
        test_entity = test_entity.reshape(1, -1)
        val = test_entity.dot(weight)
        if(val >0 and result_test[i] == 1):
            count = count + 1
        if(val < 0 and result_test[i] == 0):
            count = count + 1
    print(count)
    print((count/len(result_test))*100)


data = getcsv_to_list('SonarData.csv')
#converting all the data point to numbers from string, one column at a time
for x in range( len(data[0]) -1):
    convertStringtoFloat(data, x)
convertStringtoInt(data, len(data[0])-1)

#shuffling the list to create a better data set
#print(data[0])
random.shuffle(data)
#print(data[0][60])

#converting list to numpy array
arraynp = array(data)

#if(arraynp[0][60] == 1):
    #print(arraynp[0][60])


#splitting the created numpy array into training set and test set(The number chosen are random) there is an unsaid rule of 80-20 split
#print(arraynp.shape)
train = arraynp[:169]
test = arraynp[169:]

#print(train.shape)
#print(test.shape)

data_train = train[:,:-1]
result_train = train[:,-1]

#print(data_train.shape)
#print(result_train.shape)

data_test = test[:, :-1]
result_test = test[:,-1]

#print(data_test.shape)
#print(result_test.shape)
# This is our baap function which is controlling the whole perceptron
trainPerceptron(data_train, data_test, result_train, result_test, 1, 500000 )




