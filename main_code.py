# This challenge is designed for Wat.AI core member hiring process ( project name : Pitch.AI)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(filepath):
    # Load the dataset from the file and return it as a DataFrame
    # TODO: Implement this function

    #loaded via pandas
    df = pd.read_csv(filepath, header=None)
    return df

def preprocess_data(df):
    # Perform preprocessing like scaling features
    # TODO: Implement this function
     # looking at the data, the last column (57) is the target ; else
    scaler = StandardScaler()
    
    # Separate features and column 57
    features = df.iloc[:, :-1]
    target = df.iloc[:, -1]
    
    # Scale 
    scaled_features = scaler.fit_transform(features)
    
    # Convert scaled features back into a DF
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    scaled_df["target"] = target.values
    
    return scaled_df

def train_model(X_train, y_train):
    # Train a RandomForestClassifier on the provided data
    # TODO: Implement this function
    # RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    
    # Training the model 
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    # Evaluate the trained model and return the accuracy and classification report
    # TODO: Implement this function
    # Predict the labels for the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # classification report
    report = classification_report(y_test, y_pred)
    
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


