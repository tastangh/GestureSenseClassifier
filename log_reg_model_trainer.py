from sklearn.linear_model import LogisticRegression

class ModelTrainer:
    def __init__(self, max_iter=1000, random_state=42, class_weight="balanced"):
        self.model = LogisticRegression(max_iter=max_iter, random_state=random_state, class_weight=class_weight)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
