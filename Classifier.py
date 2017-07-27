import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# read in data
df_train = pd.read_csv("classification.csv")

# split into features and labels
X = df_train.iloc[:,:-1]
y = df_train.iloc[:,-1]

# split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# fit model to training data
#gbc = MLPClassifier(hidden_layer_sizes=[100],solver='lbfgs',activation='relu',random_state=0,alpha=.015).fit(X_train,y_train) #.7894

# predict labels test set
y_pred = gbc.predict(X_test)

# check score
print('Accuracy training data: {:.2f}'.format( gbc.score(X_train,y_train)))
print('Accuracy testing data: {:.2f}'.format( gbc.score(X_test,y_test)))
print(y_test.values)
print(y_pred)
