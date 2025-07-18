#Imperting Python Dependencies
import pandas as pd
from sklearn.linear_model import LogisticRegression


iris_data = pd.read_csv('iris.csv')
iris_data.head()

#Printing IRIS CSV Header 
# print(iris_data.head())

#Spliting data into two factors
X = iris_data.drop(columns=['Id', 'Species'])
Y = iris_data['Species']
# print(X.head())

#Create a ML Model
model = LogisticRegression()

#Train Model
model.fit(X.values,Y)

#predict the trained model
predictions = model.predict([[4.6,3.5,1.5,6.5]])

#print Predictions
print(predictions)