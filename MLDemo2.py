#Imperting Python Dependencies
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


iris_data = pd.read_csv('iris.csv')
iris_data.head()


#Spliting data into two factors
X = iris_data.drop(columns=['Id', 'Species'])
Y = iris_data['Species']
# print(X.head())

#Split data into training and testing set
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Standard the feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Create a ML Model
model = LogisticRegression()

#Train Model
model.fit(X_train_scaled, Y_train)

#Evaluate the model on the training set
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy : ", accuracy)

#Sample data for predictions
new_data = np.array([[5.1, 3.5, 1.4,  0.2],
                     [6.3, 2.9, 5.6,  1.8],
                     [4.9, 3.0, 1.4,  0.1]])

#Standardize new data
new_data_scaled = scaler.transform(new_data)

#make predicitons
predictions = model.predict(new_data_scaled)

print('predictions : ',predictions)