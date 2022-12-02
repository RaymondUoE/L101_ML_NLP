import argparse

from evaluation.model_pref import model_perf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        type=str,
        help="Task Name: (uncertainty_nli / uncertainty_abdnli).",
        required=True
    )

    parser.add_argument(
        "--data_file",
        type=str,
        help='The path to the data file.', required=True)

    parser.add_argument(
        "--prediction_file",
        type=str,
        default=None,
        help='The path to the prediction file.', required=True)

    args = parser.parse_args()

    dataset_name = ""
    if "chaosNLI_snli.jsonl" in args.data_file:
        dataset_name = "ChaosNLI-SNLI"
    elif "chaosNLI_mnli_m.jsonl" in args.data_file:
        dataset_name = "ChaosNLI-MNLI"
    elif "chaosNLI_alphanli.jsonl" in args.data_file:
        dataset_name = "ChaosNLI-alphaNLI"

    task_name = args.task_name
    data_file = args.data_file
    model_prediction_file = args.prediction_file
    model_perf(dataset_name, task_name, data_file, model_prediction_file)