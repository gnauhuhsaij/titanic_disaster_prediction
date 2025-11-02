import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
print("DATA PREPROCESSING")
print("__________________________________________________________________________________________________")
print("Loading data")
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data', 'train.csv')
df = pd.read_csv(data_path)

print("Drop entries with null values in the 'Embarked' column speficially")
df.dropna(subset = "Embarked", inplace = True)
print("Transforming catogrical variables to labels")
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
df["Pclass"] = df["Pclass"].map({1: 0, 2: 1, 3: 2})
print("Filling missing values for age column with column median, grouped by the variable 'Embarked'")
df["Age"] = df.groupby("Embarked")["Age"].apply(lambda x: x.fillna(x.median())).values
print("Extracting Titles from Names and save them as a new column")
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
rare = df["Title"].value_counts()[df["Title"].value_counts()<6].index
df["Title"] = df["Title"].replace(rare, "Rare")
print("Parse tickets into the preceding ticket label and the following ticket number and save them as two separate columns")
df["Ticket_clean"] = df["Ticket"].str.replace(r"[./]", "", regex=True).str.strip()
df["Ticket_label"] = df["Ticket_clean"].str.extract(r"^([A-Za-z]+)")
df["Ticket_label"] = df["Ticket_label"].fillna("NOLABEL")
df["Ticket_number"] = df["Ticket_clean"].str.extract(r"(\d+)")
def trim_ticket_num(x):
    if pd.isna(x):
        return np.nan
    if len(x) <= 4:
        return x
    return x[:-4]  # drop last 4 digits
df["Ticket_number_trimmed"] = df["Ticket_number"].apply(trim_ticket_num)
df["Ticket_number_trimmed"] = pd.to_numeric(df["Ticket_number_trimmed"], errors="coerce")
print("Filling missing values for cabin first letter column with 'H'")
df["Cabin"] = df["Cabin"].fillna("H")
df["Cabin"] = df["Cabin"].apply(lambda x: x[0])
df["Cabin"] = df["Cabin"].map({"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "T": 8})
df = df[df["Cabin"] != 8]
print("Dropping unnecessary columns - Ticket, Name, and PassengerId, and filtering out missing values")
df.drop(columns = ["Ticket", "Name", "PassengerId", "Ticket_clean", "Ticket_number"], inplace = True)
df.dropna(inplace = True)


print("Transforming categorical columns into dummies columns.")
print("Categorical columns: Sex, Cabin, Title, Ticket Label")
train = pd.get_dummies(df, columns=["Sex", "Cabin", "Title", "Ticket_label"], drop_first=True)
print("Splitting data into predictors and target")
X = train.drop("Survived", axis=1)
y = train["Survived"]

print("Scaling numeric predictors using StandardScaler")
print("Numeric variables: Age, Fare, Sibsp, Parch, Pclass, Embarked, Ticket Number")
numeric_features = ["Age", "Fare", "SibSp", "Parch", "Pclass", "Embarked", "Ticket_number_trimmed"]
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])


print("""
MODEL TRAINING""")
print("__________________________________________________________________________________________________")
print(
    """Find the best logistic regression model parameters by finetuning 
     - Inverse of regularization strength 'C'
     - Algorithm to use in the optimization problem 'Solver'
     - Classification Weight 'class_weight'
     Fixed the number of iterations to 50000 and the penalty term to l2.""")
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'saga'],
    'class_weight': [None, 'balanced'],
    'max_iter': [50000]
}

grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy', return_train_score=True)
grid.fit(X, y)
print("Best params:", grid.best_params_)

print("Training logistic regression model")
model = LogisticRegression(**grid.best_params_, random_state=42)
print("Fit the best model on the training dataset using the best parameters")
model.fit(X, y)

print("""
MODEL EVALUATION""")
print("__________________________________________________________________________________________________")
print("Loading test data")
data_path = os.path.join(script_dir, '..', 'data', 'test.csv')
test = pd.read_csv(data_path)
print("Preprocessing test data")
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})
test["Embarked"] = test["Embarked"].map({"S": 0, "C": 1, "Q": 2})
test["Pclass"] = test["Pclass"].map({1: 0, 2: 1, 3: 2})
test["Age"] = test.groupby("Embarked")["Age"].apply(lambda x: x.fillna(x.median())).values
test["Fare"] = test["Fare"].fillna(test["Fare"].mean())

test["Title"] = test["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
test["Title"] = test["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
test["Title"] = test["Title"].replace(rare, "Rare")

test["Ticket_clean"] = test["Ticket"].str.replace(r"[./]", "", regex=True).str.strip()
test["Ticket_label"] = test["Ticket_clean"].str.extract(r"^([A-Za-z]+)")
test["Ticket_label"] = test["Ticket_label"].fillna("NOLABEL")

test["Ticket_number"] = test["Ticket_clean"].str.extract(r"(\d+)")
test["Ticket_number_trimmed"] = test["Ticket_number"].apply(trim_ticket_num)
test["Ticket_number_trimmed"] = pd.to_numeric(test["Ticket_number_trimmed"], errors="coerce")

test["Cabin"] = test["Cabin"].fillna("H")
test["Cabin"] = test["Cabin"].apply(lambda x: x[0])
test["Cabin"] = test["Cabin"].map({"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "T": 8})
test.drop(columns = ["Ticket", "Name", "PassengerId", "Ticket_clean", "Ticket_number"], inplace = True)
test = pd.get_dummies(test, columns=["Sex", "Cabin", "Title", "Ticket_label"], drop_first=True)
test[numeric_features] = scaler.fit_transform(test[numeric_features])
test = test.reindex(columns=X.columns, fill_value=0)

print("Get the predictions for the training set and the test set")
train_pred = model.predict(X)
test_pred = model.predict(test)

print("Saved the test set predictions to data/submission.csv")
test = pd.read_csv(data_path)
output_df = test.copy()
output_df["Survived"] = test_pred
output_df = output_df[["PassengerId", "Survived"]]
data_path = os.path.join(script_dir, '..', 'data', 'submission.csv')
output_df.to_csv(data_path, index = False)

print(f"The Training Set Accuracy is {(y == train_pred).sum()/len(y)}")
print(f"The Test Set Accuracy is 0.761 (Retrieved by submitting the submissions.csv to link https://www.kaggle.com/competitions/titanic/overview)")
