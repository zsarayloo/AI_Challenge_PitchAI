# This challenge is designed for Wat.AI core member hiring process ( project name : Pitch.AI)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
def load_data(filepath):
    
    #creating column names lists since the csv file does not have column names
    word_freq_columns = [f"word_freq_{i}" for i in range(1, 49)] # 48 word frequency attributes 
    char_freq_columns = [f"char_freq_{i}" for i in range(1, 7)] # 6 char frequency attributes 

    capital_columns = [ "capital_run_length_average", 
                       "capital_run_length_longest", 
                       "capital_run_length_total" ] # 3 capital-related attributes 
    
    target_column = ["target"] # The last column (spam or not) 

    # Combine all the column names 
    cols_names = word_freq_columns + char_freq_columns + capital_columns + target_column

    # Load the dataset from the file and return it as a DataFrame
    df_data = pd.read_csv(filepath, header=None, names=cols_names).dropna(axis=1, how="all")
    
    return df_data

def preprocess_data(df):
    # Perform preprocessing like scaling features
    x = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    scaler.set_output(transform='pandas')  #returns it in the form of a dataframe instead of a array

    df_scaled = scaler.fit_transform(x)
    df_scaled["target"] = y.values

    return df_scaled


def train_model(X_train, y_train):
    # Train a RandomForestClassifier on the provided data
    model = RandomForestClassifier(n_estimators=75,max_depth=26,min_samples_leaf=1,random_state=42)
    model.fit(X_train, y_train)
    return model



def evaluate_model(model, X_test, y_test):
    # Evaluate the trained model and return the accuracy and classification report
    y_predict = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_predict)
    report = classification_report(y_test, y_predict)

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

    print(f"Model Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}") 





