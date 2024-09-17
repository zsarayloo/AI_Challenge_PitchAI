# This challenge is designed for Wat.AI core member hiring process ( project name : Pitch.AI)
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(filepath):
    # Load the dataset from the file and return it as a DataFrame
    return pd.read_csv(filepath, header=None)

def preprocess_data(df):
    # Perform preprocessing like scaling features
    features = df.iloc[:, :-1]  # All columns except the last one
    target = df.iloc[:, -1]     # The last column is the target
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Convert to DataFrame for consistency
    df_scaled = pd.DataFrame(scaled_features)
    df_scaled["target"] = target.values  # Add the target column back
    
    return df_scaled

def train_model(X_train, y_train):
    # Define the parameter grid for hyperparameter tuning using Grid Search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Initialize the RandomForestClassifier
    clf = RandomForestClassifier(random_state=42)

    # Use GridSearchCV to search for the best parameters
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, 
                               cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

    # Fit GridSearchCV on training data
    grid_search.fit(X_train, y_train)
    
    # Return only the best model found by Grid Search
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    # Evaluate the trained model and return the accuracy and classification report
    y_pred = model.predict(X_test)
    
    # Calculate accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return accuracy, report

if __name__ == "__main__":
    # Path to the dataset
    data_path = "data/spambase.csv"

    # Load and preprocess the data
    df = load_data(data_path)
    df = preprocess_data(df)

    # Split data into features and target
    X = df.drop("target", axis=1)  # Drop the target column for features
    y = df["target"]  # Target is the last column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model with Grid Search for hyperparameter tuning
    model, best_params = train_model(X_train, y_train)  # Unpack the model and best params
    print(f"Best Parameters found: {best_params}")

    # Evaluate the best model
    accuracy, report = evaluate_model(model, X_test, y_test)  # Pass only the model to evaluate_model

    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n{report}")
