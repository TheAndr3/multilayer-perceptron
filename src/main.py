from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mlp import MLP1Hidden


@dataclass
class CVSummary:
    n_hidden: int
    momentum: float
    mse_mean: float
    mse_std: float
    epochs_mean: float
    epochs_std: float
    acc_mean: float
    acc_std: float


def one_hot_encode(labels: list[str]) -> np.ndarray:
    classes = sorted(set(labels))
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_idx = np.array([class_to_idx[label] for label in labels], dtype=int)
    return np.eye(len(classes), dtype=float)[y_idx]


def load_iris(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    labels: list[str] = []

    with file_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(",")
            features.append([float(x) for x in parts[:4]])
            labels.append(parts[4])

    X = np.asarray(features, dtype=float)
    y = one_hot_encode(labels)
    return X, y


def load_circulo(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(file_path, dtype=float)
    X = data[:, :2]
    y = data[:, 2:3]
    return X, y


def split_treino_validacao_teste(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = X.shape[0]
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_samples)
    n_test = int(round(n_samples * test_size))
    n_test = max(1, min(n_samples - 1, n_test))

    test_idx = indices[:n_test]
    tv_idx = indices[n_test:]
    return X[tv_idx], X[test_idx], y[tv_idx], y[test_idx]


def kfold_indices(
    n_samples: int,
    n_splits: int = 10,
    shuffle: bool = True,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("n_splits deve ser >= 2")
    if n_splits > n_samples:
        raise ValueError("n_splits nao pode ser maior que n_samples")

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    folds: list[np.ndarray] = []
    current = 0
    for fold_size in fold_sizes:
        folds.append(indices[current : current + fold_size])
        current += fold_size

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        splits.append((train_idx, val_idx))
    return splits


def accuracy_exact_match(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    equal_rows = np.all(y_pred == y_true, axis=1)
    return float(np.mean(equal_rows) * 100.0)


def avaliar_topologia_cv(
    X_tv: np.ndarray,
    y_tv: np.ndarray,
    n_hidden: int,
    momentum: float,
    random_state: int = 42,
) -> CVSummary:
    splits = kfold_indices(
        n_samples=X_tv.shape[0],
        n_splits=10,
        shuffle=True,
        random_state=random_state,
    )

    mse_values: list[float] = []
    epochs_values: list[float] = []
    acc_values: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X_tv[train_idx], X_tv[val_idx]
        y_train, y_val = y_tv[train_idx], y_tv[val_idx]

        model_seed = 1000 + n_hidden * 100 + fold_idx
        model = MLP1Hidden(
            n_inputs=X_tv.shape[1],
            n_hidden=n_hidden,
            n_outputs=y_tv.shape[1],
            learning_rate=0.5,
            momentum=momentum,
            beta=0.5,
            epsilon=1e-3,
            max_epochs=5000,
            random_state=model_seed,
        )

        result = model.fit(X_train, y_train)
        y_pred = model.predict_binary(X_val)
        acc = accuracy_exact_match(y_pred, y_val)

        mse_values.append(float(result["mse"]))
        epochs_values.append(float(result["epochs"]))
        acc_values.append(acc)

    return CVSummary(
        n_hidden=n_hidden,
        momentum=momentum,
        mse_mean=float(np.mean(mse_values)),
        mse_std=float(np.std(mse_values)),
        epochs_mean=float(np.mean(epochs_values)),
        epochs_std=float(np.std(epochs_values)),
        acc_mean=float(np.mean(acc_values)),
        acc_std=float(np.std(acc_values)),
    )


def selecionar_melhor_topologia(
    X_tv: np.ndarray,
    y_tv: np.ndarray,
    lista_topologias: list[int],
    momentum: float,
    random_state: int = 42,
) -> tuple[CVSummary, list[CVSummary]]:
    resultados = [
        avaliar_topologia_cv(
            X_tv=X_tv,
            y_tv=y_tv,
            n_hidden=n_hidden,
            momentum=momentum,
            random_state=random_state,
        )
        for n_hidden in lista_topologias
    ]

    melhor = max(resultados, key=lambda r: (r.acc_mean, -r.mse_mean, -r.epochs_mean))
    return melhor, resultados


def treinar_e_testar_final(
    X_tv: np.ndarray,
    y_tv: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_hidden: int,
    momentum: float,
) -> float:
    final_seed = 50000 + n_hidden
    model = MLP1Hidden(
        n_inputs=X_tv.shape[1],
        n_hidden=n_hidden,
        n_outputs=y_tv.shape[1],
        learning_rate=0.5,
        momentum=momentum,
        beta=0.5,
        epsilon=1e-3,
        max_epochs=5000,
        random_state=final_seed,
    )
    model.fit(X_tv, y_tv)
    y_pred = model.predict_binary(X_test)
    return accuracy_exact_match(y_pred, y_test)


def print_cv_table(title: str, summaries: list[CVSummary]) -> None:
    print(f"\n{title}")
    print("hidden | mse(mean+-std) | epochs(mean+-std) | acc%(mean+-std)")
    print("-" * 66)
    for s in summaries:
        print(
            f"{s.n_hidden:>6} | "
            f"{s.mse_mean:>8.5f} +- {s.mse_std:<8.5f} | "
            f"{s.epochs_mean:>8.2f} +- {s.epochs_std:<8.2f} | "
            f"{s.acc_mean:>8.2f} +- {s.acc_std:<8.2f}"
        )


def processar_dataset(
    nome: str,
    X: np.ndarray,
    y: np.ndarray,
    topologias: list[int],
    random_state: int = 42,
) -> None:
    print(f"\n=== Dataset: {nome} ===")

    X_tv, X_test, y_tv, y_test = split_treino_validacao_teste(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    melhor_padrao, resultados_padrao = selecionar_melhor_topologia(
        X_tv,
        y_tv,
        lista_topologias=topologias,
        momentum=0.0,
        random_state=random_state,
    )
    melhor_momentum, resultados_momentum = selecionar_melhor_topologia(
        X_tv,
        y_tv,
        lista_topologias=topologias,
        momentum=0.9,
        random_state=random_state,
    )

    print_cv_table("Selecao por CV - Backprop Padrao", resultados_padrao)
    print_cv_table("Selecao por CV - Backprop com Momentum", resultados_momentum)

    acc_teste_padrao = treinar_e_testar_final(
        X_tv,
        y_tv,
        X_test,
        y_test,
        n_hidden=melhor_padrao.n_hidden,
        momentum=0.0,
    )
    acc_teste_momentum = treinar_e_testar_final(
        X_tv,
        y_tv,
        X_test,
        y_test,
        n_hidden=melhor_momentum.n_hidden,
        momentum=0.9,
    )

    print("\nMelhores topologias:")
    print(
        f"- Padrao: hidden={melhor_padrao.n_hidden}, "
        f"acc_cv={melhor_padrao.acc_mean:.2f}%"
    )
    print(
        f"- Momentum: hidden={melhor_momentum.n_hidden}, "
        f"acc_cv={melhor_momentum.acc_mean:.2f}%"
    )

    print("\nTeste puro (20% hold-out):")
    print(f"- Acuracia teste - Padrao: {acc_teste_padrao:.2f}%")
    print(f"- Acuracia teste - Momentum: {acc_teste_momentum:.2f}%")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    iris_path = repo_root / "iris.data"
    circulo_path = repo_root / "circulo.txt"

    topologias = [2, 4, 6, 8, 10]

    X_iris, y_iris = load_iris(iris_path)
    X_circulo, y_circulo = load_circulo(circulo_path)

    processar_dataset("Iris", X_iris, y_iris, topologias)
    processar_dataset("Circulo", X_circulo, y_circulo, topologias)


if __name__ == "__main__":
    main()
