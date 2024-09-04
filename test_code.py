import unittest
from main_code import load_data, preprocess_data, train_model, evaluate_model

class TestSpambaseChallenge(unittest.TestCase):
    def setUp(self):
        self.data_path = "data/spambase.csv"
        self.df = load_data(self.data_path)
        self.df = preprocess_data(self.df)
        self.X = self.df.drop("target", axis=1)
        self.y = self.df["target"]

    def test_load_data(self):
        self.assertIsNotNone(self.df)
        self.assertGreater(len(self.df), 0)  # Check that data is loaded

    def test_preprocess_data(self):
        self.assertFalse(self.df.isnull().values.any())  # Check for missing values

    def test_train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        self.assertIsNotNone(model)
        accuracy, _ = evaluate_model(model, X_test, y_test)
        self.assertGreater(accuracy, 0.8)  # Expecting at least 80% accuracy

if __name__ == "__main__":
    unittest.main()

