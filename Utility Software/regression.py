import numpy as np
import numpy 
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model
from sklearn.cross_validation import train_test_split

diabetes = datasets.load_diabetes()
x=diabetes.data[:,0]
X=x.reshape((x.shape[0],1))
y=diabetes.target
xtrain,xtest,ytrain,ytest = train_test_split(X,y,random_state=0)

lin_reg= linear_model.LinearRegression()
print (xtrain.shape, ytrain.shape)
lin_reg.fit(xtrain, ytrain)
predicted_y = lin_reg.predict(xtest)
mse = np.mean((predicted_y - ytest)**2)
print (mse)

plt.figure()
plt.scatter(xtest,ytest,color = 'red',label = 'actual points')
plt.plot(xtest,predicted_y,color = 'blue',label = ' fitted line')
plt.legend(loc = 'upper right')
plt.show()



