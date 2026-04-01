import numpy as np


class MLP1Hidden:
    """MLP com 1 camada oculta, função logística e backpropagation."""

    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
        n_outputs: int,
        learning_rate: float = 0.5,
        beta: float = 0.5,
        epsilon: float = 1e-3,
        max_epochs: int = 5000,
        random_state: int | None = None,
    ):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.max_epochs = max_epochs

        self.rng = np.random.default_rng(random_state)

        # Pesos inicializados no intervalo [0, 1], como pedido no enunciado.
        self.W1 = self.rng.uniform(0.0, 1.0, size=(n_inputs + 1, n_hidden))
        self.W2 = self.rng.uniform(0.0, 1.0, size=(n_hidden + 1, n_outputs))

        self.history_mse = []

    def _sigmoid(self, u: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.beta * u))

    def _sigmoid_derivative_from_output(self, g_u: np.ndarray) -> np.ndarray:
        return self.beta * g_u * (1.0 - g_u)

    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        bias = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack([X, bias])

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Xb = self._add_bias(X)

        u_hidden = Xb @ self.W1
        y_hidden = self._sigmoid(u_hidden)
        y_hidden_b = self._add_bias(y_hidden)

        u_out = y_hidden_b @ self.W2
        y_out = self._sigmoid(u_out)

        return Xb, y_hidden, y_hidden_b, y_out

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Treina com backpropagation online (amostra por amostra)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = X.shape[0]

        for epoch in range(1, self.max_epochs + 1):
            squared_error_sum = 0.0

            for i in range(n_samples):
                xi = X[i : i + 1]
                ti = y[i : i + 1]

                Xb, y_hidden, y_hidden_b, y_out = self.forward(xi)

                error_out = ti - y_out
                squared_error_sum += float(np.sum(error_out**2))

                delta_out = error_out * self._sigmoid_derivative_from_output(y_out)

                # Remove coluna de bias para retropropagar apenas pelos neurônios ocultos.
                W2_no_bias = self.W2[:-1, :]
                delta_hidden = (delta_out @ W2_no_bias.T) * self._sigmoid_derivative_from_output(y_hidden)

                grad_W2 = y_hidden_b.T @ delta_out
                grad_W1 = Xb.T @ delta_hidden

                self.W2 += self.learning_rate * grad_W2
                self.W1 += self.learning_rate * grad_W1

            mse = squared_error_sum / n_samples
            self.history_mse.append(mse)

            if mse <= self.epsilon:
                return {"epochs": epoch, "mse": mse}

        return {"epochs": self.max_epochs, "mse": self.history_mse[-1]}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        _, _, _, y_out = self.forward(X)
        return y_out

    def predict_binary(self, X: np.ndarray) -> np.ndarray:
        y_prob = self.predict_proba(X)
        return (y_prob >= 0.5).astype(int)
