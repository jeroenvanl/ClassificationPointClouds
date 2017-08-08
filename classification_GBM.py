import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree,svm
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.externals import joblib
import pickle
# read in data
df_train = pd.read_csv("classification.csv")


# normalize normals 
df_train['normalX']=df_train['normalX'].abs()
df_train['normalY']=df_train['normalY'].abs()
df_train['normalZ']=df_train['normalZ'].abs()

# split into features and labels
X = df_train.iloc[:,:-1]
y = df_train.iloc[:,-1]

normalize=True
if(normalize):
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)
    df_normalized = pd.DataFrame(np_scaled)
else:
    df_normalized = X


# split into training and testing
X_train, X_test, y_train, y_test = train_test_split(df_normalized, y, random_state=0)


# fit model to training data
#gbc = MLPClassifier(hidden_layer_sizes=(20,20),random_state=0,alpha=.1,max_iter=10000).fit(X_train,y_train) #.88
#gbc = KNeighborsClassifier(n_neighbors = 9).fit(X_train,y_train) #.89
#gbc = RandomForestClassifier(n_estimators=1000).fit(X_train,y_train)
gbc = GradientBoostingClassifier(random_state=0,learning_rate=.150).fit(X_train,y_train) #.96


# predict labels test set
y_pred = gbc.predict(X_test)

# check score
print('Accuracy training data: {:.2f}'.format( gbc.score(X_train,y_train)))
print('Accuracy testing data: {:.2f}'.format( gbc.score(X_test,y_test)))
print(y_test.values)
print(y_pred)




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class_names=['floor','wall','ceiling','rest']
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')



#Plot feature imporances
#objects = X_train.columns.values
#y_pos = np.arange(len(objects))
#performance = gbc.feature_importances_
#performance, objects = (list(t) for t in zip(*sorted(zip(performance, objects))))
#plt.bar(y_pos, performance, align='center', alpha=0.5)
#plt.xticks(y_pos, objects,rotation='vertical')
#plt.ylabel('Usage')
#plt.title('Programming language usage')
#plt.show()