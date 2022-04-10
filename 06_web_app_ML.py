# Import libraries
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Heading
st.write("""
# Explore different ML models on datasets
And lets see which works best in it.
""")
# Data set initialization
dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris','Wine','Breast Cancer')
)
classifier_name = st.sidebar.selectbox(
    'Select Classifier',
    ('KNN','SVM','Random Forest')
)

# Import Datasets

def get_dataset(dataset_name):
    data=None
    if dataset_name =='iris':
        data = datasets.load_iris()
    elif dataset_name =='wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x,y
# Now going to call function and x,y variable ky equal rakh lengy
X,y = get_dataset(dataset_name)
# Now we will check the shape of our dataset
st.write("Shape of dataset:", X.shape)
st.write("Number of classes:", len(np.unique(y)))

# Now by adding different parameters of 3 classifiers

def add_parameter_ui(classifier_name):
    params = dict()  # create an empty dictionary
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C',0.01,10.0)
        params['C']=C # degree of correct classification
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K',1,15)
        params['K']= K # the number of K nearest neighbor value
    else:
        max_depth = st.sidebar.slider('max_depth', 2,15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators',1,100)
        params['n_estimators']=n_estimators
    return params
# Now calling function and returning the value

params = add_parameter_ui(classifier_name)

# Now we will implement classifier

def get_classifier(classifier_name,params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
        max_depth=params['max_depth'], random_state=1234)
    return clf
clf = get_classifier(classifier_name,params)
# Now splitting data into train test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1234)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

# Now accuracy score function

acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier =  {classifier_name}')
st.write(f'Accuracy =', acc)

# Now plotting Dataset
# making two dimensions plot or draw using all the features
pca = PCA(2)
X_projected = pca.fit_transform(X)
# slicing data into 0 and 1
x1 = X_projected[:,0]
x2 = X_projected[:,1]
fig = plt.figure()
plt.scatter(x1,x2, c=y, alpha=0.8,cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
st.pyplot(fig)


