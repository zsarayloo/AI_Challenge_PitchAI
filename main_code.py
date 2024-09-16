# This challenge is designed for Wat.AI core member hiring process ( project name : Pitch.AI)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(filepath):
    # Load the dataset from the file and return it as a DataFrame
    # TODO: Implement this function
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    # Perform preprocessing like scaling features
    # TODO: Implement this function
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled

def train_model(X_train, y_train):
    # Train a RandomForestClassifier on the provided data
    # TODO: Implement this function
    param_grid = {
        'n_estimators': np.arange(100, 501, 10),
        'max_depth': np.arange(1, 25)
    }
    rfc = RandomForestClassifier()
    random_search = RandomizedSearchCV(rfc, 
                                       param_distributions=param_grid, 
                                       n_iter=10, 
                                       cv=5, 
                                       random_state=42)
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    # Evaluate the trained model and return the accuracy and classification report
    # TODO: Implement this function
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

if __name__ == "__main__":

    # Path to the dataset
    data_path = "data/spambase.csv"

    # Load the data
    df = load_data(data_path)

    # Split data into features and target
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
    X = X.to_numpy()
    y = y.to_numpy()
    y = y.ravel()

    # Preprocess the data
    X = preprocess_data(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Train and evaluate the model
    model = train_model(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test)

    print(model)
    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n{report}")


