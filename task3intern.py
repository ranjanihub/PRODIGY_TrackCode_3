#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset from local directory
data = pd.read_csv(r"C:\Users\TRINITY ELE\Downloads\bank+marketing\bank-additional\bank-additional\bank-additional.csv", sep=";")

# Extract target variable 'y' before preprocessing
y = data['y']
X = data.drop(columns=['y'])

# Encoding categorical variables
le = LabelEncoder()
X_encoded = X.apply(le.fit_transform)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Building the decision tree classifier
clf = DecisionTreeClassifier()

# Training the decision tree classifier
clf.fit(X_train, y_train)

# Predicting on the testing set
y_pred = clf.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualizing the decision tree
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=X_encoded.columns.tolist())
plt.show()


# In[ ]:




