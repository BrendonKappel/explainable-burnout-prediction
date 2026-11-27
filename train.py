import pickle

import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocess import MODEL_PATH, PROJECT_DIR, build_preprocessor, prepare_inputs


DATA_PATH = PROJECT_DIR / "data" / "denning_dataset.csv"
TARGET_MAPPING = {0: 0, 1: 0, 2: 0, 3: 1}


def build_pipeline():
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=1.0,
    )
    return Pipeline([("prepare", build_preprocessor()), ("model", model)])


def main():
    raw = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    features = prepare_inputs(raw)
    target = raw["Burnout?"].map(TARGET_MAPPING)
    if target.isna().any():
        raise ValueError("Burnout? contém ausências ou valores fora de 0, 1, 2 e 3.")

    pipeline = build_pipeline().fit(features, target.astype("Int64"))
    MODEL_PATH.write_bytes(pickle.dumps(pipeline, protocol=pickle.HIGHEST_PROTOCOL))
    print(f"Modelo treinado com {len(features)} registros. Arquivo salvo em {MODEL_PATH}.")


if __name__ == "__main__":
    main()
