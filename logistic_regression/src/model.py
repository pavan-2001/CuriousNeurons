import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = None
        self.cost_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def _compute_cost(self, Y_true, Y_pred):
        eps = 1e-15
        Y_pred = np.clip(Y_pred, eps, 1-eps)
        m = len(Y_true)
        cost = - (1 / m) * np.sum(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))
        return cost
    
    def _compute_gradient(self, X, Y_true, Y_pred):
        m = X.shape[0]

        dz = Y_pred - Y_true
        dW = (1 / m) * np.dot(X.T, dz)
        db = (1 / m) * np.sum(dz)
        return dW, db
    
    def fit(self, X, Y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0

        for epoch in range(self.epochs):
            Y_pred = self._forward(X)
            cost = self._compute_cost(Y, Y_pred)
            dW, db = self._compute_gradient(X, Y, Y_pred)

            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db

            self.cost_history.append(cost)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.4f}")

        return self

    def predict_proba(self, X):
        return self._forward(X)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)