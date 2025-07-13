import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    # Load dataset
    dataset = hydra.utils.instantiate(cfg.dataset)
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.dataset.test_size, random_state=cfg.dataset.random_state
    )

    # Initialize model
    model = hydra.utils.instantiate(cfg.model)

    # Train model
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model: {cfg.model._target_}")
    print(f"Dataset: {cfg.dataset.name}")
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train()