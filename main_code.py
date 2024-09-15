# This challenge is designed for Wat.AI core member hiring process ( project name : Pitch.AI)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(filepath):
    # Load the dataset from the file and return it as a DataFrame

    # Defining column names since spambase.csv doesn't have them (from spambase.DOCUMENTATION and spambase.names)
    # This is not needed
    column_names = [
        # 48 wor dfrequency columns
        'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 'word_freq_over', 
        'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
        'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free', 'word_freq_business', 'word_freq_email', 
        'word_freq_you', 'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money',
        'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs',
        'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology',
        'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting', 
        'word_freq_original', 'word_freq_project','word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference',

        # 6 character frequency columns
        'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#',

        # 3 capital run length columns
        'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total',

        # Target column
        'target'
    ]

    return pd.read_csv(filepath, header=None, names=column_names)

def preprocess_data(df):
    # Perform preprocessing like scaling features

    # Separate features and labels
    X = df.iloc[:, :-1] # All columns except last one (features)
    y = df.iloc[:, -1] # Last column (labels)

    # Scale features using StandardScaler
    scaler = StandardScaler()
    df_preprocess = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    df_preprocess['target'] = y

    return df_preprocess
    

def train_model(X_train, y_train):
    # Train a RandomForestClassifier on the provided data
    
    # Initialize RandomForestClassifier
    model = RandomForestClassifier(random_state=42, n_estimators=100) # Default 100, can change later if needed
    model.fit(X_train, y_train) # Training based on data

    return model

def evaluate_model(model, X_test, y_test):
    # Evaluate the trained model and return the accuracy and classification report
    
    # Making predictions on test data
    y_pred = model.predict(X_test)

    # Calculating accuracy and generating classification report
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
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the model
    model = train_model(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test)

    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n{report}")


