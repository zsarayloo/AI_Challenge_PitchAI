# This challenge is designed for Wat.AI core member hiring process ( project name : Pitch.AI)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Loads the dataset 
def load_data(filepath):

    df = pd.read_csv(filepath, header = None)
    return df

# "Cleans" the data in the dataset by setting it all to the same scale and 
# checking to see if there are outliers and null values
def preprocess_data(df):

    if df.isnull().values.any():
        df = df.dropna()

    # Changed .drop to .iloc since there were no headers in the dataset
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    scaler = StandardScaler() 
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns = X.columns)

    df_preprocessed = pd.concat([X_scaled_df, y.reset_index(drop = True)], axis = 1)
    
    return df_preprocessed


# Trains the model using the RandomForestClassifier algorithm
# Calls .fit function to train model 
def train_model(X_train, y_train):
   
   model = RandomForestClassifier(random_state = 42)
   model.fit(X_train, y_train)

   return model

# Evaluates the model and produces the accuracy as well as the report
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    
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
    # Changed the .drop for the .iloc because there are no headers in the dataset
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the model
    model = train_model(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test)

    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n{report}")
