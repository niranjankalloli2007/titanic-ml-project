# ============================
# IMPORT LIBRARIES
# ============================
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# ============================
# LOAD DATA
# ============================
train_df = pd.read_csv("Titanic_train.csv")
test_df = pd.read_csv("Titanic_test.csv")

# ============================
# DATA PREPROCESSING
# ============================
test_ids = test_df['PassengerId']

train_df = train_df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
test_df = test_df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

for df in [train_df, test_df]:
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

# ============================
# FEATURE ENGINEERING
# ============================
for df in [train_df, test_df]:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 1
    df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0

# ============================
# ENCODING
# ============================
train_df = pd.get_dummies(train_df, drop_first=True)
test_df = pd.get_dummies(test_df, drop_first=True)

train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

# ============================
# SPLIT DATA
# ============================
X = train_df.drop('Survived', axis=1)
Y = train_df['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ============================
# SCALING
# ============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================
# MODEL
# ============================
model = LogisticRegression(max_iter=2000)

# ============================
# CROSS VALIDATION
# ============================
cv_scores = cross_val_score(model, X, Y, cv=5)

print("Cross Validation:", cv_scores.mean())

# ============================
# TRAIN MODEL
# ============================
model.fit(X_train, Y_train)

# ============================
# EVALUATION
# ============================
Y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))

print("\nFinal Result:")
print("Train Accuracy:", model.score(X_train, Y_train))
print("Test Accuracy:", model.score(X_test, Y_test))