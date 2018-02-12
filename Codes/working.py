import numpy as np
#numpy library use for machine learning 
import urllib2
#urlib library is used for plotting graph
import numpy 
import matplotlib.pyplot as plt
#the garph is been plotted by plt  
from sklearn import datasets,linear_model
#linear algorithim is been imported 
from sklearn.cross_validation import train_test_split
url="https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
#fetching the data sets from the url
raw_data=urllib2.urlopen(url)
dataset=np.loadtxt(raw_data,delimiter=",")
print(dataset.shape)
x=dataset[:,0:8]
y=dataset[:,0:8]
lin_reg= linear_model.LinearRegression()
print x,y
lin_reg.fit(x,y)
predicted_y=lin_reg.predict(x)
#linear regression alogorithm is applied
mse=np.random.normal(0.0,0.5,size=(120,10))**2 
mse=mse/np.sum(mse,axis=1)[:,None]

plt.pcolor(mse)

maxvi = np.argsort(mse,axis=1)#sort the dataset by the probability that they belong to a cluster 
ii = np.argsort(maxvi[:,-1])

plt.pcolor(mse[ii,:])
plt.xlabel('indian diabetes')
plt.ylabel('prdicted diabetes')
print mse
print predicted_y

plt.figure()
plt.scatter(x,y,color='red',label='actual points')
plt.plot(x,predicted_y,color='blue',label='fitted line')
plt.xlabel('No. of diabetes patient in year 2013-14')
plt.ylabel('No. of diabetes patients in the year 2014-2015')
plt.legend(loc='upper right')
plt.show()
