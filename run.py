import argparse

from handlers import (
    LIGHTWEIGHT_DATASETS,
    AVAILABLE_DATASETS,
    DatasetHandler,
    ModelTrainer,
    Evaluation,
    info
)
from ipm import get_ipm
from maia import UDRandomForestClassifier as UD
from sklearn.ensemble import RandomForestClassifier as RF

def main(dataset_path, output_prefix, light):

    dataset_handler = DatasetHandler(dataset_path)
    datasets = LIGHTWEIGHT_DATASETS if light else AVAILABLE_DATASETS
    evaluator = Evaluation(output_prefix, datasets=datasets)


    for dataset_name, experiment in datasets:
        info(f"Processing dataset: {dataset_name}")

        X_train, y_train = dataset_handler.load_data(experiment, "train")
        max_samples = 0.1 if experiment == "acspubcov" else None
        trainer = ModelTrainer(unbiased_model=RF, biased_model=UD, n_jobs=-1, max_samples=max_samples)

        # Train base model
        base_model = trainer.train_base_model(X_train, y_train)
        info("Base model trained")

        # Load test data
        X_id, y_id = dataset_handler.load_data(experiment, "id_test")
        X_ood, y_ood = dataset_handler.load_data(experiment, "ood_test")

        # Evaluate base model
        y_hat_id = base_model.predict(X_id)
        y_hat_ood = base_model.predict(X_ood)

        metrics_id = evaluator.compute_metrics(y_id, y_hat_id)
        metrics_ood = evaluator.compute_metrics(y_ood, y_hat_ood)

        for metric, value in metrics_id.items():
            evaluator.update_results(experiment, f"{metric}_id", value)

        for metric, value in metrics_ood.items():
            evaluator.update_results(experiment, f"{metric}_ood", value)

        bias = get_ipm(base_model, X_ood).clip(1e-9)
        strategies = [bias]

        del base_model


        # Train adapted model
        for i, strategy in enumerate(strategies, start=1):
            adapted_model = trainer.train_adapted_model(X_train, y_train, feature_bias=strategy)
            y_hat_ood_adapted = adapted_model.predict(X_ood)
            del adapted_model

            metrics_adapted = evaluator.compute_metrics(y_ood, y_hat_ood_adapted)
            for metric, value in metrics_adapted.items():
                evaluator.update_results(experiment, f"{metric}_s{i}", value)
            
            info(f"Strategy {i} finished")

        info(f"Finished dataset: {dataset_name}")

    # Generate LaTeX reports
    strategy_names = ["IPM"]
    latex_report = evaluator.generate_latex_report(strategy_names)

    with open(f"{output_prefix}.tex", "w") as f:
        f.write(latex_report)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-path",
        type=str,
        default="../tableshift/tmp",
        help="Path to the downloaded datasets"
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default="ipm_in_rfbs",
        help="Prefix used for output files"
    )

    parser.add_argument(
        "--light",
        action="store_true",
        help="Use only smaller datasets"
    )

    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_prefix = args.output_prefix
    light = args.light

    main(dataset_path, output_prefix, light)
