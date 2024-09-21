# This challenge is designed for Wat.AI core member hiring process ( project name : Pitch.AI)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(filepath):
    # Load the dataset from the file and return it as a DataFrame
    # TODO: Implement this function
    column_names = [f'feat_{i}' for i in range(57)] + ['target']
    df = pd.read_csv(filepath, header=None, names=column_names)
    return df

def preprocess_data(df):
    # Perform preprocessing like scaling features
    # TODO: Implement this function
    scaler = StandardScaler()
    X = df.drop("target", axis=1)
    y = df["target"]

    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled["target"] = y.values

    return df_scaled

def train_model(X_train, y_train):
    # Train a RandomForestClassifier on the provided data
    # TODO: Implement this function
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # Evaluate the trained model and return the accuracy and classification report
    # TODO: Implement this function
    y_predict = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred=y_predict)
    report = classification_report(y_test, y_pred=y_predict)
    return accuracy, report

if __name__ == "__main__":
    # Path to the dataset
    data_path = "data/spambase.csv"

    # Load and preprocess the data
    df = load_data(data_path)
    df = preprocess_data(df)

    # Split data into features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the model
    model = train_model(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test)

    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n{report}")


