import argparse
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap

from preprocess import CATEGORICAL_COLUMNS, MODEL_PATH, PROJECT_DIR, prepare_inputs
from train import main as train_model


OUTPUTS_DIR = PROJECT_DIR / "outputs"


def save_results(source, pipeline, features):
    prepare = pipeline.named_steps["prepare"]
    model = pipeline.named_steps["model"]
    encoder = prepare.named_transformers_["categorical"].named_steps["encoder"]
    for column, categories in zip(CATEGORICAL_COLUMNS, encoder.categories_, strict=True):
        values = features[column]
        unknown = values.notna() & ~values.isin(categories)
        if unknown.any():
            print(f"Aviso: {unknown.sum()} registro(s) com categoria desconhecida em {column}.")

    prepared = prepare.transform(features)
    feature_names = [name.split("__", 1)[-1] for name in prepare.get_feature_names_out()]
    explanation_inputs = pd.DataFrame(prepared, columns=feature_names, index=features.index)
    explainer = shap.TreeExplainer(
        model, model_output="raw", feature_perturbation="tree_path_dependent",
    )
    explanations = explainer(explanation_inputs)

    records = [f"registro_{i:03d}" for i in range(1, len(features) + 1)]
    graphs = [f"explanations/{record}.png" for record in records]
    predictions = pd.DataFrame({
        "arquivo_entrada": source.name,
        "registro": records,
        "classe_prevista": model.predict(prepared),
        "probabilidade_classe_1": model.predict_proba(prepared)[:, 1],
        "grafico": graphs,
    })
    name = f"{source.stem}_{datetime.now():%Y-%m-%d_%H%M%S}"
    destination = OUTPUTS_DIR / name
    suffix = 2
    while True:
        try:
            destination.mkdir(parents=True)
            break
        except FileExistsError:
            destination = OUTPUTS_DIR / f"{name}_{suffix}"
            suffix += 1

    (destination / "explanations").mkdir()
    print(f"Salvando {len(graphs)} gráficos SHAP...", flush=True)
    for i, graph in enumerate(graphs):
        shap.plots.waterfall(explanations[i], max_display=10, show=False)
        plt.savefig(destination / graph, dpi=150, bbox_inches="tight")
        plt.close()
    predictions.to_csv(destination / "predictions.csv", index=False, encoding="utf-8-sig")
    return destination


def main():
    parser = argparse.ArgumentParser(description="Gera previsões e gráficos SHAP para um CSV.")
    parser.add_argument("csv", type=Path, help="Arquivo CSV com os dados de entrada.")
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.exit(2, f"{parser.prog}: error: CSV não encontrado: {args.csv}\n")

    try:
        features = prepare_inputs(pd.read_csv(args.csv, encoding="utf-8-sig"))
        for record, missing in enumerate(features.isna().to_numpy(), start=1):
            if missing.any():
                columns = ", ".join(features.columns[missing])
                raise ValueError(f"Registro {record} com campos ausentes: {columns}")
    except (ValueError, TypeError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")

    if not MODEL_PATH.is_file():
        train_model()
    with MODEL_PATH.open("rb") as file:
        pipeline = pickle.load(file)
    plt.switch_backend("Agg")
    print(f"Gerando previsões para {len(features)} registros...", flush=True)
    output = save_results(args.csv, pipeline, features)
    print(f"{len(features)} registros processados. Resultados em {output}.")


if __name__ == "__main__":
    main()
