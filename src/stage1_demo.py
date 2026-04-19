import numpy as np

from mlp import MLP1Hidden


# Exemplo didático com problema lógico AND (classificação binária).
X = np.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
)

y = np.array([0, 0, 0, 1])

model = MLP1Hidden(
    n_inputs=2,
    n_hidden=4,
    n_outputs=1,
    learning_rate=0.5,
    beta=0.5,
    epsilon=1e-3,
    max_epochs=10000,
    random_state=42,
)

result = model.fit(X, y)

proba = model.predict_proba(X)
pred = model.predict_binary(X).ravel()

print("Treino final:")
print(f"  epocas: {result['epochs']}")
print(f"  mse: {result['mse']:.6f}")
print("\nPredicoes (AND):")
for xi, pi, yi in zip(X, proba.ravel(), pred):
    print(f"  x={xi.tolist()} -> prob={pi:.4f}, classe={int(yi)}")
