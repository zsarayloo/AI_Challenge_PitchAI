# This challenge is designed for Wat.AI core member hiring process ( project name : Pitch.AI)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

def load_data(filepath):
    # The csv file does not contain column names, so we need to specify them manually

    columns = [ # The first 57 labels are the features, the last label is the target
                # For the features we have taken the label names from https://archive.ics.uci.edu/dataset/94/spambase
                # We add the target label at the end
        "word_freq_make",
        "word_freq_address",
        "word_freq_all",
        "word_freq_3d",
        "word_freq_our",
        "word_freq_over",
        "word_freq_remove",
        "word_freq_internet",
        "word_freq_order",
        "word_freq_mail",
        "word_freq_receive",
        "word_freq_will",
        "word_freq_people",
        "word_freq_report",
        "word_freq_addresses",
        "word_freq_free",
        "word_freq_business",
        "word_freq_email",
        "word_freq_you",
        "word_freq_credit",
        "word_freq_your",
        "word_freq_font",
        "word_freq_000",
        "word_freq_money",
        "word_freq_hp",
        "word_freq_hpl",
        "word_freq_george",
        "word_freq_650",
        "word_freq_lab",
        "word_freq_labs",
        "word_freq_telnet",
        "word_freq_857",
        "word_freq_data",
        "word_freq_415",
        "word_freq_85",
        "word_freq_technology",
        "word_freq_1999",
        "word_freq_parts",
        "word_freq_pm",
        "word_freq_direct",
        "word_freq_cs",
        "word_freq_meeting",
        "word_freq_original",
        "word_freq_project",
        "word_freq_re",
        "word_freq_edu",
        "word_freq_table",
        "word_freq_conference",
        "char_freq_;",
        "char_freq_\(",
        "char_freq_\[",
        "char_freq_!",
        "char_freq_$",
        "char_freq_#",
        "capital_run_length_average",
        "capital_run_length_longest",
        "capital_run_length_total",
        "target"
    ]
    
    # Load the dataset from the file and return it as a DataFrame
    df = pd.read_csv(filepath, header=None, names=columns)

    return df

def preprocess_data(df):
    # The dataset does not contain any missing values
    # The dataset does not contain any categorical columns (Except the target column)
    # The dataset does not contain any columns that need to be dropped
    # --> We need to scale only the feature columns

    # Separate features and target
    fts = df.drop("target", axis=1)  # All columns except target
    
    # Perform scaling only on the feature columns
    fts_scaled = StandardScaler().fit_transform(fts)

    fts_scaled_df = pd.DataFrame(fts_scaled, columns=fts.columns, index=df.index)
    df_scaled = pd.concat([fts_scaled_df, df["target"]], axis=1)
    
    return df_scaled

def train_model(X_train, y_train):
    # Define a grid of hyperparameters
    param_grid = { # In order to find the best hyperparameters, we need to define a grid of possible values
        # All possible values for the hyperparameters will be fitted and the best combination will be chosen
        # The possible values are chosen based on common values used in practice
        'n_estimators': [10, 100], # Number of trees in the forest
        'max_depth': [20, 25], # Maximum depth of the tree
        'min_samples_split': [3, 5], # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 3], # Minimum number of samples required to be at a leaf node
        'bootstrap': [True, False] # Whether bootstrap samples are used when building trees
    } # We won't use too many values to keep the computation time low

    # Use GridSearchCV for hyperparameter tuning
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Return the best model
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    # Evaluate the trained model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    rp = classification_report(y_test, y_pred)
    return accuracy, rp

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
