# This challenge is designed for Wat.AI core member hiring process ( project name : Pitch.AI)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(filepath):
    # Load the dataset from the file and return it as a DataFrame

    df = pd.read_csv(filepath)
    return df


def preprocess_data(df):
    # Perform preprocessing like scaling features
    #drop the features with coorelation less than <0.1
    dict1 = dict(df.corr()['1'])
    list_features = []
    for key, values in dict1.items():
        if abs(values) < 0.1:
            list_features.append(key)
    df = df.drop(list_features, axis=1)
    return df


def train_model(X_train, y_train):
    # Train a RandomForestClassifier on the provided data
    rfcModel = RandomForestClassifier()
    rfcModel.fit(X_train, y_train)
    return rfcModel

def evaluate_model(model, X_test, y_test):
    # Evaluate the trained model and return the accuracy and classification report
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, report

if __name__ == "__main__":
    # Path to the dataset
    data_path = "data/spambase.csv"

    # Load and preprocess the data
    df = load_data(data_path)
    df = preprocess_data(df)

    # Split data into features and target
    y = df['1']

    X = df.drop('1', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the model
    model = train_model(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test)

    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n{report}")


