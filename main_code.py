# Applicant: Dilshaan Sandhu
# This challenge is designed for Wat.AI core member hiring process ( project name : Pitch.AI)

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

""" After doing some independent work, I have found that the XGboost model preformed
    slightly better then the RandomForestClassifer. 
    
    Documentation on both:
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    https://xgboost.readthedocs.io/en/stable/

"""


def load_data(filepath):

    # Load the dataset from the file and return it as a DataFrame
    spam_train_df = pd.read_csv(filepath)
    return spam_train_df


def preprocess_data(df):
    """
       * Would use one-hot encoding/label encoding, but all the data is in numerical continuous format.
       * Taking a look through the x-labels of the dataset, it seems that no other type of pre-processing is needed
       * The data description on the official file states that there are missing values, but after doing EDA on a seperate
       file I was unable to identify any.
       * Further preprocessing decisions could be made if the original text data was supplied.
    """


    # check if dataframe has null or missing values

    if df.isna().any().any():
        print("Dataframe has empty/null values! Dropping rows with NA/None values")

        # dropping all rows with NA/None Values
        df.dropna(inplace=True)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Scale dataframe feature data to a common numerical system
    std_scalar = StandardScaler()
    X_scaled = std_scalar.fit_transform(X)

    # Creating a new dataframe containing cleaned data
    cleaned_dataframe = pd.DataFrame(data=X_scaled, columns=X.columns)
    cleaned_dataframe = cleaned_dataframe.join(y.reset_index(drop=True))

    return cleaned_dataframe


def train_model(X_train, y_train):

    # Train a RandomForestClassifier on the provided data. Got optimialish hyperparameters from tuning function
    spam_detect_model = RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2,
                                               n_estimators=150)
    spam_detect_model.fit(X_train, y_train)

    return spam_detect_model


def evaluate_model(model, X_test, y_test):
    # Evaluate the trained model and return the accuracy and classification report
    y_pred = model.predict(X_test)

    # Calc accuracy of model by compared to correctly labeled data
    acc = accuracy_score(y_test, y_pred)

    # create classification report, providing advanced stats
    rep = classification_report(y_test, y_pred)

    return acc, rep


# Extra: Hyperparameter tuning example
def tuning_model(X_train, y_train):
    """Although not part of the assignment, I thought it would be a good idea to include an example of hyperparameter tuning
    which is used to optimize model performance by changing the models parameters (default in current use)"""

    # Example.
    para_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 3]
    }

    # create model and intialize GridSearchCV
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid=para_grid, verbose=2, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    return best_params


if __name__ == "__main__":
    # Path to the dataset
    data_path = "data/spambase.csv"

    # Load and preprocess the data
    df = load_data(data_path)
    df = preprocess_data(df)

    # original Provided code did not work, workaround below due to column names not in dataset

    # Drop the last column
    X = df.iloc[:, :-1]

    # Extract the last column
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the model
    model = train_model(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test)

    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n\n {report}")
