import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(filepath):
    """
    Load the dataset from the file and return it as a DataFrame
    """
    try:
        # Load data, assuming it's comma-separated
        df = pd.read_csv(filepath, header=None)
        # Adding column names based on UCI Spambase dataset structure
        df.columns = [f"feature_{i}" for i in range(df.shape[1]-1)] + ["target"]
        return df
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def preprocess_data(df):
    """
    Perform preprocessing like scaling features
    """
    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]
    
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Return the preprocessed features and target
    return pd.DataFrame(X_scaled, columns=X.columns), y

def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier on the provided data
    """
    # Initialize and train the random forest classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and return the accuracy and classification report
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

if __name__ == "__main__":
    # Path to the dataset (replace with the correct file path)
    data_path = "./data/spambase.data"
    
    # Load and preprocess the data
    df = load_data(data_path)
    
    if df is not None:
        X, y = preprocess_data(df)

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train and evaluate the model
        model = train_model(X_train, y_train)
        accuracy, report = evaluate_model(model, X_test, y_test)

        print(f"Model Accuracy: {accuracy:.2f}")
        print(f"Classification Report:\n{report}")
