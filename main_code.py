# This challenge is designed for Wat.AI core member hiring process ( project name : Pitch.AI)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(filepath):
    # Load the dataset from the file and return it as a DataFrame
    # TODO: Implement this function
    df = pd.read_csv(filepath, header=None)
    df.rename(columns={57:'target'}, inplace=True)
    print(df.head())
    return df

def preprocess_data(df):
    # Perform preprocessing like scaling features
    # TODO: Implement this function
    sc = StandardScaler()
    scaled_features = sc.fit_transform(df.iloc[:,:-1])
    scaled_features = pd.DataFrame(scaled_features, columns=df.columns[:-1])
    scaled_features['target'] = df['target'].values
    return scaled_features

def train_model(X_train, y_train):
    # Train a RandomForestClassifier on the provided data
    # TODO: Implement this function
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    return rf

def evaluate_model(model, X_test, y_test):
    # Evaluate the trained model and return the accuracy and classification report
    # TODO: Implement this function
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    return acc, clf_report

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


