import os
import pandas as pd


def create_directory(primary_directory_name):
    result_dir = f"{primary_directory_name}/results"
    try:
        os.mkdir(result_dir)
    except FileExistsError:
        print("Folder is Exist")

    return result_dir


def creating_error_csv(
    primary_directory_name,
    learning_rate,
    iteration_number,
    round_number,
    breakpoint_number,
):
    result_dir = create_directory(primary_directory_name)
    overall_learning_rate_err_ = []
    for lr in learning_rate:
        round_err_list = []
        round_percent_list = []
        round_validate_err_list = []
        round_validate_err_percent_list = []
        round_brakpoint_err_list = []
        round_brakpoint_err_percent_list = []
        round_brakpoint_val_err_list = []
        round_brakpoint_val_err_percent_list = []
        round_errors = []
        learning_rate_err_dict = {"learning_rate": str(lr)}
        for round_no in range(round_number):
            round_err_dict = {"learning_rate": str(lr)}
            directory_name = primary_directory_name + "/round" + str(round_no + 1)
            df = pd.read_csv(f"{directory_name}/{iteration_number}-{lr}.csv")
            round_err_list.append(df["error"].values[-1])
            round_percent_list.append(df["error_percent"].values[-1])
            round_validate_err_list.append(df["validate_error"].values[-1])
            round_validate_err_percent_list.append(
                df["validate_error_percent"].values[-1]
            )
            breakpoint_df = df[df["epoch"] <= breakpoint_number]
            round_brakpoint_err_list.append(breakpoint_df["error"].values[-1])
            round_brakpoint_err_percent_list.append(
                breakpoint_df["error_percent"].values[-1]
            )
            round_brakpoint_val_err_list.append(
                breakpoint_df["validate_error"].values[-1]
            )
            round_brakpoint_val_err_percent_list.append(
                breakpoint_df["validate_error_percent"].values[-1]
            )

            # Added to the dictionary
            round_err_dict[f"round"] = round_no
            round_err_dict[f"error"] = df["error"].values[-1]
            round_err_dict[f"error_percent"] = df["error_percent"].values[-1]
            round_err_dict[f"validate_error"] = df["validate_error"].values[-1]
            round_err_dict[f"validate_error_percent"] = df[
                "validate_error_percent"
            ].values[-1]
            round_err_dict[f"brakpoint_error"] = breakpoint_df["error"].values[-1]
            round_err_dict[f"brakpoint_error_percent"] = breakpoint_df[
                "error_percent"
            ].values[-1]
            round_err_dict[f"brakpoint_validate_error"] = breakpoint_df[
                "validate_error"
            ].values[-1]
            round_err_dict[f"brakpoint_validate_error_percent"] = breakpoint_df[
                "validate_error_percent"
            ].values[-1]
            round_errors.append(round_err_dict)

        round_err_df = pd.DataFrame(round_errors)
        round_err_df.to_csv(f"{result_dir}/{lr}_round_error.csv", index=False)

        average_error = round_err_df["error"].mean()
        average_error_percent = round_err_df["error_percent"].mean()
        average_validate_error = round_err_df["validate_error"].mean()
        average_validate_error_percent = round_err_df["validate_error_percent"].mean()
        average_brakpoint_error = round_err_df["brakpoint_error"].mean()
        average_brakpoint_error_percent = round_err_df["brakpoint_error_percent"].mean()
        average_brakpoint_validate_error = round_err_df[
            "brakpoint_validate_error"
        ].mean()
        average_brakpoint_validate_error_percent = round_err_df[
            "brakpoint_validate_error_percent"
        ].mean()
        learning_rate_err_dict["average_error"] = average_error
        learning_rate_err_dict["average_error_percent"] = average_error_percent
        learning_rate_err_dict["average_validate_error"] = average_validate_error
        learning_rate_err_dict["average_validate_error_percent"] = (
            average_validate_error_percent
        )
        learning_rate_err_dict["average_brakpoint_error"] = average_brakpoint_error
        learning_rate_err_dict["average_brakpoint_error_percent"] = (
            average_brakpoint_error_percent
        )
        learning_rate_err_dict["average_brakpoint_validate_error"] = (
            average_brakpoint_validate_error
        )
        learning_rate_err_dict["average_brakpoint_validate_error_percent"] = (
            average_brakpoint_validate_error_percent
        )

        overall_learning_rate_err_.append(learning_rate_err_dict)

    overall_learning_rate_err_df = pd.DataFrame(overall_learning_rate_err_)
    overall_learning_rate_err_df.to_csv(
        f"{result_dir}/overall_learning_rate_err.csv", index=False
    )
