# This challenge is designed for Wat.AI core member hiring process ( project name : Pitch.AI)
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    # Load the dataset from the file and return it as a DataFrame
    # TODO: Implement this function
    columns = ["make", "address", "all", "3d", "our", "over", "remove", "internet",
                "order", "mail", "receive", "will", "people", "report", "addresses",
                "free", "business", "email", "you", "credit", "your", "font", "000",
                "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
                "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
                "meeting", "original", "project", "re", "edu", "table", "conference", 
                ";", "(", "[", "!", "$", "#", "capital_run_length_average", "capital_run_length_longest"
                "capital_run_length_total", "target"]
    df = pd.read_csv(filepath, header=None, names=columns)
    #df.rename(columns={57:'target'}, inplace=True)
    print(df.head())
    print(df.info())
    print(df.isnull().sum())
    print(df.describe())
    return df

def visualizations(df):
    #correlation
    corr = df.corr()
    
    plt.figure(dpi=130)
    sns.heatmap(df.corr(), annot=True, fmt= '.2f')
    plt.show()

def preprocess_data(df):
    # Perform preprocessing like scaling features
    # TODO: Implement this function
    
    # To avoid the risk of leaking information about the test set into your model, 
    # it's better to fit the scaler using only the training data, then apply it to both the training and test sets. 
    # If you fit the scaler on the entire dataset before splitting, information from the test set influences the transformation
    # of the training data, which can affect downstream processes.
    # For instance, knowing the distribution of the complete dataset might impact how you handle outliers or configure model
    # parameters. Even though the actual data isn’t revealed, distributional insights are. 
    # Consequently, your test set performance wouldn’t accurately reflect how the model performs on unseen data.
    
    sc = StandardScaler()
    scaled_features = sc.fit_transform(df.iloc[:,:-1])
    scaled_features = pd.DataFrame(scaled_features, columns=df.columns[:-1])
    scaled_features['target'] = df['target'].values
    return scaled_features

def cross_val_scores(X_train, y_train):
    # Lets do Cross validation in order to check whether model is performing consistantly and not over fitting
    clf_cv = RandomForestClassifier()
    kf = KFold(n_splits=10, shuffle=True)
    cv_results = cross_val_score(clf_cv, X_train, y_train, cv=kf, scoring="accuracy")
    return cv_results

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

def find_hyperparams(clf, X_val, y_val):
    # Set the parameters by cross-validation
    param_grid = [{'n_estimators': [x for x in range(50, 100)],
                    'criterion': ['gini', 'entropy'],
                    'max_features': ['sqrt', 'log2', None ]}]
    grid = GridSearchCV(clf, param_grid)
    grid.fit(X_val, y_val)
    print('done fitting')
    return grid.best_estimator_

if __name__ == "__main__":
    # Path to the dataset
    data_path = "data/spambase.csv"

    # Load data
    df = load_data(data_path)
    # visualizations 
    visualizations(df)
    # preprocess the data
    df = preprocess_data(df)

    # Split data into features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # cross validation 
    results = cross_val_scores(X_train, y_train)
    print("\nCross validation score is: \n", results)
    # CV results indecate model is performing well and can be used on test set

    # Train and evaluate the model
    model = train_model(X_train, y_train)
    # As results are considerable no need to do hyperparamter tuning
    accuracy, report = evaluate_model(model, X_test, y_test)

    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n{report}")


