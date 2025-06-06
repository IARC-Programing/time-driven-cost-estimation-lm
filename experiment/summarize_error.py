import pandas as pd


dataset_names = ["1-munchkin", "2-chinchilla",
                 "3-vicheanmas", "4-scottishfold"]
academic_dataset_names = ["Dataset 1", "Dataset 3", "Dataset 2", "Dataset 4"]


learning_rate = [0.005, 0.01, 0.05, 0.1, 0.5]
overall_accuracy_df = pd.DataFrame()

model_learning_rates = ["1e-07", "1e-08"]
iteration = 100
modifiers = [
    "",
    "_remove_outlier",
    "_augmented",
    "_remove_outlier_augmented",
    "_early_stopping",
    "_remove_outlier_early_stopping",
    "_augmented_early_stopping",
    "_remove_outlier_augmented_early_stopping",
]
displayed_modifiers = [
    "Original",
    "Remove Outlier",
    "Augmentation",
    "Remove Outlier + Augmentation",
    "Early Stopping",
    "Remove Outlier + Early Stopping",
    "Augmentation + Early Stopping",
    "Remove Outlier + Augmentation + Early Stopping",
]


def find_overall_accuracy(round_number=3,
                          model_version=1):
    overall_accuracy_df = pd.DataFrame()
    for dataset_name in dataset_names:
        for round_no in range(round_number):
            for modifer in modifiers:
                accuracy_list = []
                # Find Best Accuracy
                # For Each Model Learning Rate
                for model_learning_rate in model_learning_rates:
                    directory_name = f"result/{dataset_name}/{dataset_name}_{model_version}_{model_learning_rate}{modifer}"
                    # For Each Learning Rate
                    for lr in learning_rate:
                        datafile = f"{directory_name}/results/{lr}_round_error.csv"
                        result = pd.read_csv(datafile)
                        result_with_this_round = result[result["round"]
                                                        == round_no]
                        result_with_this_round = result_with_this_round.iloc[0]
                        # Find the last iteration which is early stopping
                        epoch_error_file = (
                            f"{directory_name}/round{round_no+1}/{iteration}-{lr}.csv"
                        )
                        epoch_error = pd.read_csv(epoch_error_file)
                        # Find Last Record
                        epoch_error = epoch_error.iloc[-1]
                        # Find last record epoch
                        last_epoch = epoch_error["epoch"]
                        accuracy_payload = {
                            "type": "training",
                            "learning_rate": lr,
                            "model_learning_rate": model_learning_rate,
                            "rmspe": result_with_this_round["error_percent"],
                            "last_epoch": last_epoch,
                        }
                        accuracy_list.append(accuracy_payload)
                        accuracy_payload = {
                            "type": "validate",
                            "learning_rate": lr,
                            "model_learning_rate": model_learning_rate,
                            "rmspe": result_with_this_round["validate_error_percent"],
                            "last_epoch": last_epoch,
                        }
                        accuracy_list.append(accuracy_payload)

                accuracy_list = pd.DataFrame(accuracy_list)
                best_training = (
                    accuracy_list[accuracy_list["type"] == "training"]
                    .sort_values(by="rmspe")
                    .iloc[0]
                )

                best_validate = (
                    accuracy_list[accuracy_list["type"] == "validate"]
                    .sort_values(by="rmspe")
                    .iloc[0]
                )

                # Find Data Variation For Training
                if "augmented" in modifer:
                    train_data_variation = pd.read_csv(
                        f"{directory_name}/train_data_variation_after_augmented_{round_no}.csv"
                    )
                else:
                    train_data_variation = pd.read_csv(
                        f"{directory_name}/train_data_variation_{round_no}.csv"
                    )

                train_cost_data = train_data_variation[
                    train_data_variation["data"] == "Total Cost"
                ]
                train_cost_data = train_cost_data.iloc[0]
                payload = {
                    "dataset": dataset_name,
                    "round_no": round_no,
                    "modifer": modifer,
                    "type": "training",
                    "cv": train_cost_data["variation"],
                    "iqr": train_cost_data["iqr"],
                    "min": train_cost_data["min"],
                    "max": train_cost_data["max"],
                    "mean": train_cost_data["mean"],
                    "minimum_error": best_training["rmspe"],
                    "best_learning_rate": best_training["learning_rate"],
                    "best_model_learning_rate": best_training["model_learning_rate"],
                    "last_epoch": best_training["last_epoch"],
                }
                overall_accuracy_df = pd.concat(
                    [overall_accuracy_df, pd.DataFrame(payload, index=[0])],
                    ignore_index=True,
                )
                # Find Data Variation For Validate
                validate_data_variation = pd.read_csv(
                    f"{directory_name}/validate_data_variation_{round_no}.csv"
                )
                validate_cost_data = validate_data_variation[
                    validate_data_variation["data"] == "Total Cost"
                ]
                validate_cost_data = validate_cost_data.iloc[0]
                payload = {
                    "dataset": dataset_name,
                    "round_no": round_no,
                    "modifer": modifer,
                    "type": "validate",
                    "cv": validate_cost_data["variation"],
                    "iqr": validate_cost_data["iqr"],
                    "min": validate_cost_data["min"],
                    "max": validate_cost_data["max"],
                    "mean": validate_cost_data["mean"],
                    "minimum_error": best_validate["rmspe"],
                    "best_learning_rate": best_training["learning_rate"],
                    "best_model_learning_rate": best_training["model_learning_rate"],
                    "last_epoch": best_training["last_epoch"],
                }

                overall_accuracy_df = pd.concat(
                    [overall_accuracy_df, pd.DataFrame(payload, index=[0])],
                    ignore_index=True,
                )
    # Post Processing
    overall_accuracy_df["remove_outlier"] = overall_accuracy_df["modifer"].apply(
        lambda x: 1 if "_remove_outlier" in x else 0
    )
    overall_accuracy_df["augmented"] = overall_accuracy_df["modifer"].apply(
        lambda x: 1 if "_augmented" in x else 0
    )
    overall_accuracy_df["early_stopping"] = overall_accuracy_df["modifer"].apply(
        lambda x: 1 if "_early_stopping" in x else 0
    )

    overall_accuracy_df.to_csv("result/overall_accuracy.csv", index=False)
