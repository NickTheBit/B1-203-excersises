# ###################################
# Group ID : 203
# Members : Malthe Boelskift, Louis Ildal, Guillermo Gutierrez Bea, Nikolaos Gkloumpos.
# Date : 18/10/2023
# Lecture: 12
# Dependencies: numpy, matplotlib, sklearn
# Python version: 3.11.4
# Functionality: Use trees with varying structure to discriminate against poor people :)
# ###################################

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Load California housing data
california_housing = fetch_california_housing()
data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
target = pd.DataFrame(california_housing.target, columns=["Price"])

# Create binary classes for classification
mean_price = target["Price"].mean()
target["Class"] = (target["Price"] > mean_price).astype(int)

# Split the data into training and test sets for classification
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    data, target["Class"], test_size=0.2, random_state=42
)

# Train decision tree classifier
def train_classifier(max_depth):
    classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    classifier.fit(X_train_cls, y_train_cls)
    return classifier

# Evaluate classification performance
def evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    train_predictions = classifier.predict(X_train)
    test_predictions = classifier.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    return train_accuracy, test_accuracy

# Classification experiment with different tree depths
for depth in [None, 3, 5, 10]:
    classifier = train_classifier(depth)
    train_acc, test_acc = evaluate_classifier(classifier, X_train_cls, y_train_cls, X_test_cls, y_test_cls)
    print(f"Tree Depth: {depth}, Training Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

# Split the data into training and test sets for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    data, target["Price"], test_size=0.2, random_state=42
)

# Train decision tree regressor
def train_regressor(max_depth):
    regressor = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    regressor.fit(X_train_reg, y_train_reg)
    return regressor

# Evaluate regression performance
def evaluate_regressor(regressor, X_train, y_train, X_test, y_test):
    train_predictions = regressor.predict(X_train)
    test_predictions = regressor.predict(X_test)

    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)

    return train_mse, test_mse

# Regression experiment with different tree depths
for depth in [None, 3, 5, 10]:
    regressor = train_regressor(depth)
    train_mse, test_mse = evaluate_regressor(regressor, X_train_reg, y_train_reg, X_test_reg, y_test_reg)
    print(f"Tree Depth: {depth}, Training MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
 
    