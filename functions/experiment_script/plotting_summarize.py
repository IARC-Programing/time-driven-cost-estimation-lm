import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

sns.set_theme(style="whitegrid", font="Noto Sans",
              font_scale=1.5)
this_graph_palette = sns.color_palette("husl", 4)
sns.set_palette(this_graph_palette)
dataset_names = ["1-munchkin",
                 "3-vicheanmas", "2-chinchilla", "4-scottishfold"]
# cademic_dataset_names = ["Dataset 1", "Dataset 3", "Dataset 2", "Dataset 4"]
academic_dataset_names = [
    "Simple Dataset", "Complicated Dataset", "Actual Dataset", "Extended Random Dataset"]


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


def to_kebab_case(value):
    return "-".join(value.lower().split())


def plotting_learning_curve(use_academic_name=False, round_no=0,
                            model_version=1):
    dataset_index = 0
    for dataset_name in dataset_names:
        modifier_index = 0
        for modifer in modifiers:
            accuracy_list = []
            # Find Best Accuracy
            # For Each Model Learning Rate
            fig, ax = plt.subplots(2, 5, figsize=(20, 8))
            model_learning_rate_index = 0
            learning_rate_index = 0
            if use_academic_name:
                fig.suptitle(
                    f"Learning Curve of {academic_dataset_names[dataset_index]} - {displayed_modifiers[modifier_index]}",
                    fontdict={"fontsize": 20, "fontweight": "bold"},
                )

            else:
                fig.suptitle(
                    f"Learning Curve of {academic_dataset_names[dataset_index]} - {displayed_modifiers[modifier_index]}",
                    fontdict={"fontsize": 20, "fontweight": "bold"},
                )
            for model_learning_rate in model_learning_rates:
                directory_name = f"result/{dataset_name}/{dataset_name}_{model_version}_{model_learning_rate}{modifer}"
                # For Each Learning Rate
                for lr in learning_rate:
                    epoch_error_file = (
                        f"{directory_name}/round{round_no + 1}/{iteration}-{lr}.csv"
                    )
                    epoch_error = pd.read_csv(epoch_error_file)
                    ax[model_learning_rate_index][learning_rate_index].plot(
                        epoch_error["epoch"],
                        epoch_error["error_percent"],
                        label="Training",
                        linewidth=4
                    )
                    ax[model_learning_rate_index][learning_rate_index].plot(
                        epoch_error["epoch"],
                        epoch_error["validate_error_percent"],
                        label="Validation", linewidth=4
                    )
                    ax[model_learning_rate_index][learning_rate_index].set_title(
                        f"{lr}/{model_learning_rate}",
                        fontdict={"fontsize": 16},
                    )
                    # ax[model_learning_rate_index][learning_rate_index].set_xlabel(
                    #     "Epoch"
                    # )
                    # ax[model_learning_rate_index][learning_rate_index].set_ylabel(
                    #     "Error RMSPE"
                    # )
                    ax[model_learning_rate_index][learning_rate_index].set_ylim(
                        0, 100)
                    ax[model_learning_rate_index][learning_rate_index].set_xlim(
                        0, 100)
                    # ax[model_learning_rate_index][learning_rate_index].legend(
                    #     loc="lower right"
                    # )

                    learning_rate_index += 1
                model_learning_rate_index += 1
                learning_rate_index = 0
            handles, labels = ax[0][0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center",
                       ncol=2, fontsize=12)
            fig.tight_layout(pad=2.0)
            plt.tight_layout()
            os.makedirs(
                f"result/learning_curve/by-modifer/{dataset_name}", exist_ok=True
            )
            if modifer == "":
                modifer = "original"
            plt.savefig(
                f"result/learning_curve/by-modifer/{dataset_name}/{modifer}.png"
            )
            plt.close(fig)
            accuracy_list = pd.DataFrame(accuracy_list)
            modifier_index += 1

        dataset_index += 1


def plotting_learning_curve_by_rate(use_academic_name=False, round_no=0,
                                    model_version=1):
    dataset_index = 0
    for dataset_name in dataset_names:
        # Find Best Accuracy
        # For Each Model Learning Rate

        model_learning_rate_index = 0
        learning_rate_index = 0
        for model_learning_rate in model_learning_rates:
            # For Each Learning Rate
            for lr in learning_rate:
                fig, ax = plt.subplots(2, 4, figsize=(20, 8))
                for modifier_index, modifer in enumerate(modifiers):
                    directory_name = f"result/{dataset_name}/{dataset_name}_{model_version}_{model_learning_rate}{modifer}"
                    epoch_error_file = (
                        f"{directory_name}/round{round_no + 1}/{iteration}-{lr}.csv"
                    )
                    epoch_error = pd.read_csv(epoch_error_file)
                    row_index = 0 if modifier_index < 4 else 1
                    col_index = modifier_index % 4
                    ax[row_index][col_index].plot(
                        epoch_error["epoch"],
                        epoch_error["error_percent"],
                        label="Training",
                    )
                    ax[row_index][col_index].plot(
                        epoch_error["epoch"],
                        epoch_error["validate_error_percent"],
                        label="Validation",
                    )
                    if use_academic_name:
                        ax[row_index][col_index].set_title(
                            f"{academic_dataset_names[dataset_index]}\n{displayed_modifiers[modifier_index]}\n{lr}/{model_learning_rate}",
                            fontdict={"fontsize": 12},
                        )
                    else:
                        ax[row_index][col_index].set_title(
                            f"{dataset_name}\n{displayed_modifiers[modifier_index]}\n{lr}/{model_learning_rate}",
                            fontdict={"fontsize": 12},
                        )
                    ax[row_index][col_index].set_xlabel(
                        f'Epoch\n Lasted Training Error {epoch_error.iloc[-1]["error_percent"]:.2f} \n Lasted Validation Error {epoch_error.iloc[-1]["validate_error_percent"]:.2f}'
                    )
                    ax[row_index][col_index].set_ylabel("Error RMSPE")
                    ax[row_index][col_index].set_ylim(0, 100)
                    ax[row_index][col_index].set_xlim(0, 100)
                    ax[row_index][col_index].legend(loc="lower right")
                    learning_rate_index += 1
                fig.tight_layout(pad=2.0)
                os.makedirs(
                    f"result/learning_curve/by-rate/{dataset_name}", exist_ok=True
                )
                plt.savefig(
                    f"result/learning_curve/by-rate/{dataset_name}/{model_learning_rate}-{lr}.png"
                )
                plt.close(fig)
            model_learning_rate_index += 1
            learning_rate_index = 0

        dataset_index += 1


def plot_graph_grid(use_academic_name=False, use_early_stopping=False, round_no=0, model_version=1):
    dataset_index = 0
    early_stopping_title = "With Early Stopping" if use_early_stopping else ""
    for dataset_name in dataset_names:
        # Find Best Accuracy
        # For Each Model Learning Rate

        if use_early_stopping:
            filter_modifier = modifiers[4:]
            filter_display_modifier = displayed_modifiers[4:]
        else:
            filter_modifier = modifiers[:4]
            filter_display_modifier = displayed_modifiers[:4]
        model_learning_rate_index = 0
        learning_rate_index = 0
        for model_learning_rate in model_learning_rates:
            # For Each Learning Rate
            fig, ax = plt.subplots(5, 4, figsize=(20, 14))
            if use_academic_name:
                fig.suptitle(
                    f"Learning Curve of {academic_dataset_names[dataset_index]} - β ={model_learning_rate} {early_stopping_title}",
                    fontdict={"fontsize": 14, "fontweight": "bold"},
                )

            else:
                fig.suptitle(
                    f"Learning Curve of {academic_dataset_names[dataset_index]} - β ={model_learning_rate} {early_stopping_title}",
                    fontdict={"fontsize": 14, "fontweight": "bold"},
                )
            learning_rate_index = 0
            for lr in learning_rate:
                for modifier_index, modifer in enumerate(filter_modifier):
                    directory_name = f"result/{dataset_name}/{dataset_name}_{model_version}_{model_learning_rate}{modifer}"
                    epoch_error_file = (
                        f"{directory_name}/round{round_no + 1}/{iteration}-{lr}.csv"
                    )
                    epoch_error = pd.read_csv(epoch_error_file)

                    col_index = modifier_index
                    ax[learning_rate_index][col_index].plot(
                        epoch_error["epoch"],
                        epoch_error["error_percent"],
                        label="Training",
                        linewidth=4,
                    )
                    ax[learning_rate_index][col_index].plot(
                        epoch_error["epoch"],
                        epoch_error["validate_error_percent"],
                        label="Validation",
                        linewidth=4,
                    )
                    if use_academic_name:
                        ax[learning_rate_index][col_index].set_title(
                            f"{filter_display_modifier[modifier_index]}\n α = {lr}",
                            fontdict={"fontsize": 20},
                        )

                    # best_train_epoch = epoch_error[
                    #     epoch_error["error_percent"]
                    #     == epoch_error["error_percent"].min()
                    # ]
                    # best_validate_epoch = epoch_error[
                    #     epoch_error["validate_error_percent"]
                    #     == epoch_error["validate_error_percent"].min()
                    # ]
                    # try:
                    #     best_train_epoch = best_train_epoch.iloc[0]
                    #     best_train_epoch = best_train_epoch["epoch"]
                    #     best_validate_epoch = best_validate_epoch.iloc[0]
                    #     best_validate_epoch = best_validate_epoch["epoch"]
                    # except:
                    #     best_train_epoch = np.nan
                    #     best_validate_epoch = np.nan
                    # ax[learning_rate_index][col_index].set_xlabel(
                    #     f'Train Err {epoch_error.iloc[-1]["error_percent"]:.2f} , Validate Err {epoch_error.iloc[-1]["validate_error_percent"]:.2f} \n Best Epoch Train {best_train_epoch:.0f} , Validate {best_validate_epoch:.0f}'
                    # )
                    # ax[learning_rate_index][col_index].set_ylabel(
                    #     'Error RMSPE')
                    ax[learning_rate_index][col_index].set_ylim(0, 100)
                    ax[learning_rate_index][col_index].set_xlim(0, 100)
                    # ax[learning_rate_index][col_index].legend(
                    #     loc='lower right')
                learning_rate_index += 1
            handles, labels = ax[0][0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center",
                       ncol=2, fontsize=12)
            fig.tight_layout(pad=2.0)
            early_stopping_modifier = "/early-stopping" if use_early_stopping else ""
            if round_no > 0:
                early_stopping_modifier = f"/round-{round_no}/{early_stopping_modifier}"
            os.makedirs(
                f"result/learning_curve/grid-rate-modifier/{dataset_name}{early_stopping_modifier}",
                exist_ok=True,
            )
            plt.savefig(
                f"result/learning_curve/grid-rate-modifier/{dataset_name}{early_stopping_modifier}/{model_learning_rate}.png"
            )
            plt.close(fig)
            model_learning_rate_index += 1
            learning_rate_index = 0

        dataset_index += 1


# TODO: Not Implement


def plot_model_weight(use_academic_name=False, use_early_stopping=False, round_no=0, model_version=1):
    dataset_index = 0
    early_stopping_title = "With Early Stopping" if use_early_stopping else ""
    for dataset_name in dataset_names:
        # Find Best Accuracy
        # For Each Model Learning Rate

        if use_early_stopping:
            filter_modifier = modifiers[4:]
            filter_display_modifier = displayed_modifiers[4:]
        else:
            filter_modifier = modifiers[:4]
            filter_display_modifier = displayed_modifiers[:4]
        model_learning_rate_index = 0
        learning_rate_index = 0
        for model_learning_rate in model_learning_rates:
            # For Each Learning Rate
            fig, ax = plt.subplots(5, 4, figsize=(15, 14))
            if use_academic_name:
                fig.suptitle(
                    f"Weight Adjustment of {academic_dataset_names[dataset_index]} - β ={model_learning_rate} {early_stopping_title}",
                    fontdict={"fontsize": 14, "fontweight": "bold"},
                )

            else:
                fig.suptitle(
                    f"Weight Adjustment of {academic_dataset_names[dataset_index]} - β ={model_learning_rate} {early_stopping_title}",
                    fontdict={"fontsize": 14, "fontweight": "bold"},
                )
            learning_rate_index = 0
            for lr in learning_rate:
                for modifier_index, modifer in enumerate(filter_modifier):
                    directory_name = f"result/{dataset_name}/{dataset_name}_{model_version}_{model_learning_rate}{modifer}"
                    epoch_error_file = (
                        f"{directory_name}/round{round_no + 1}/{iteration}-{lr}.csv"
                    )
                    epoch_error = pd.read_csv(epoch_error_file)
                    adjustment_file = (
                        f"{directory_name}/round{round_no + 1}/sample-payload-list-{iteration}-{lr}.csv"
                    )
                    adjustment_data = pd.read_csv(adjustment_file)

                    col_index = modifier_index
                    ax[learning_rate_index][col_index].plot(
                        adjustment_data["model_weight_1"],
                        label="Material Element Weight",
                    )
                    ax[learning_rate_index][col_index].plot(
                        adjustment_data["model_weight_2"],
                        label="Labor Element Weight",
                    )
                    ax[learning_rate_index][col_index].plot(
                        adjustment_data["model_weight_3"],
                        label="Utiltiy Cost Element Weight",
                    )
                    if use_academic_name:
                        ax[learning_rate_index][col_index].set_title(
                            f"{filter_display_modifier[modifier_index]}\n α = {lr}",
                            fontdict={"fontsize": 12},
                        )

                    best_train_epoch = epoch_error[
                        epoch_error["error_percent"]
                        == epoch_error["error_percent"].min()
                    ]
                    best_validate_epoch = epoch_error[
                        epoch_error["validate_error_percent"]
                        == epoch_error["validate_error_percent"].min()
                    ]
                    try:
                        best_train_epoch = best_train_epoch.iloc[0]
                        best_train_epoch = best_train_epoch["epoch"]
                        best_validate_epoch = best_validate_epoch.iloc[0]
                        best_validate_epoch = best_validate_epoch["epoch"]
                    except Exception as e:
                        best_train_epoch = np.nan
                        best_validate_epoch = np.nan
                    ax[learning_rate_index][col_index].set_xlabel(
                        f'Train Err {epoch_error.iloc[-1]["error_percent"]:.2f} , Validate Err {epoch_error.iloc[-1]["validate_error_percent"]:.2f} \n Best Epoch Train {best_train_epoch:.0f} , Validate {best_validate_epoch:.0f}'
                    )
                    ax[learning_rate_index][col_index].set_xticks([])

                learning_rate_index += 1
            handles, labels = ax[0][0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center",
                       ncol=2, fontsize=12)
            fig.tight_layout(pad=2.0)
            early_stopping_modifier = "/early-stopping" if use_early_stopping else ""
            if round_no > 0:
                early_stopping_modifier = f"/round-{round_no}/{early_stopping_modifier}"
            os.makedirs(
                f"result/learning_curve/hyperparameter/model-weight-{dataset_name}{early_stopping_modifier}",
                exist_ok=True,
            )
            plt.savefig(
                f"result/learning_curve/hyperparameter/model-weight-{dataset_name}{early_stopping_modifier}/{model_learning_rate}.png"
            )
            plt.close(fig)
            model_learning_rate_index += 1
            learning_rate_index = 0

        dataset_index += 1


def find_amount_of_employee(sample_df):
    column_list = sample_df.columns.tolist()
    employee_column_list = [
        col for col in column_list if "employee_weight_" in col]
    number_of_employee = len(employee_column_list)
    return number_of_employee


def plot_element_weights(use_academic_name=False, use_early_stopping=False, round_no=0, model_version=1):
    dataset_index = 0
    early_stopping_title = "With Early Stopping" if use_early_stopping else ""
    for dataset_name in dataset_names:
        # Find Best Accuracy
        # For Each Model Learning Rate

        if use_early_stopping:
            filter_modifier = modifiers[4:]
            filter_display_modifier = displayed_modifiers[4:]
        else:
            filter_modifier = modifiers[:4]
            filter_display_modifier = displayed_modifiers[:4]
        model_learning_rate_index = 0
        learning_rate_index = 0
        for model_learning_rate in model_learning_rates:
            # For Each Learning Rate
            fig, ax = plt.subplots(5, 4, figsize=(18, 17))
            if use_academic_name:
                fig.suptitle(
                    f"Bias Adjustment of {academic_dataset_names[dataset_index]} - β ={model_learning_rate} {early_stopping_title}",
                    fontdict={"fontsize": 14, "fontweight": "bold"},
                )

            else:
                fig.suptitle(
                    f"Model Element Weight Adjustment of {academic_dataset_names[dataset_index]} - β ={model_learning_rate} {early_stopping_title}",
                    fontdict={"fontsize": 14, "fontweight": "bold"},
                )
            learning_rate_index = 0
            for lr in learning_rate:
                for modifier_index, modifer in enumerate(filter_modifier):
                    directory_name = f"result/{dataset_name}/{dataset_name}_{model_version}_{model_learning_rate}{modifer}"
                    epoch_error_file = (
                        f"{directory_name}/round{round_no + 1}/{iteration}-{lr}.csv"
                    )
                    epoch_error = pd.read_csv(epoch_error_file)
                    adjustment_file = (
                        f"{directory_name}/round{round_no + 1}/sample-payload-list-{iteration}-{lr}.csv"
                    )
                    adjustment_data = pd.read_csv(adjustment_file)

                    col_index = modifier_index
                    # Material Weights
                    ax[learning_rate_index][col_index].plot(
                        adjustment_data["material_weight_1"],
                        label="A Crab", color="LightPink"
                    )
                    ax[learning_rate_index][col_index].plot(
                        adjustment_data["material_weight_2"],
                        label="C Crab", color="SkyBlue"
                    )

                    # In Case of Not Exist Material Weight 4 in Actual Case
                    try:
                        ax[learning_rate_index][col_index].plot(
                            adjustment_data["material_weight_4"],
                            label="Loss Crab", color="MediumPurple"
                        )
                        ax[learning_rate_index][col_index].plot(
                            adjustment_data["material_weight_3"],
                            label="Small Crab", color="LightGreen"
                        )
                    except:
                        ax[learning_rate_index][col_index].plot(
                            adjustment_data["material_weight_3"],
                            label="Small and Loss Crab", color="MediumPurple"
                        )

                    # Employee Weights
                    number_of_employee = find_amount_of_employee(
                        adjustment_data)

                    employee_colors = ["PaleTurquoise", "Cyan",
                                       "LightCyan", "Turquoise", "DarkTurquoise"]
                    emp_index = 0
                    for emp in range(number_of_employee):
                        ax[learning_rate_index][col_index].plot(
                            adjustment_data[f"employee_weight_{emp + 1}"],
                            label=f"Employee {emp}",
                            color=employee_colors[emp % len(employee_colors)]
                        )
                        emp_index += 1

                    # Utility Cost Weights
                    ax[learning_rate_index][col_index].plot(
                        adjustment_data["capital_cost_weight_1"],
                        label="Electricity Cost",
                        color="gold"
                    )

                    ax[learning_rate_index][col_index].plot(
                        adjustment_data["capital_cost_weight_2"],
                        label="Water Supply Cost",
                        color="yellow"
                    )

                    if use_academic_name:
                        ax[learning_rate_index][col_index].set_title(
                            f"{filter_display_modifier[modifier_index]}\n α = {lr}",
                            fontdict={"fontsize": 12},
                        )

                    best_train_epoch = epoch_error[
                        epoch_error["error_percent"]
                        == epoch_error["error_percent"].min()
                    ]
                    best_validate_epoch = epoch_error[
                        epoch_error["validate_error_percent"]
                        == epoch_error["validate_error_percent"].min()
                    ]
                    try:
                        best_train_epoch = best_train_epoch.iloc[0]
                        best_train_epoch = best_train_epoch["epoch"]
                        best_validate_epoch = best_validate_epoch.iloc[0]
                        best_validate_epoch = best_validate_epoch["epoch"]
                    except Exception as e:
                        best_train_epoch = np.nan
                        best_validate_epoch = np.nan

                    # ax[learning_rate_index][col_index].set_xlabel(
                    #     f'Train Err {epoch_error.iloc[-1]["error_percent"]:.2f} , Validate Err {epoch_error.iloc[-1]["validate_error_percent"]:.2f} \n Best Epoch Train {best_train_epoch:.0f} , Validate {best_validate_epoch:.0f}'
                    # )
                    ax[learning_rate_index][col_index].set_xticks([])
                    ax[learning_rate_index][col_index].set_ylim(-2000, 10000)
                    # Ensure all values are positive before setting the y-axis to logarithmic scale
                    # if (adjustment_data.select_dtypes(include=[np.number]) > 0).all().all():
                    #     ax[learning_rate_index][col_index].set_yscale('log')
                    # else:
                    #     print(
                    #         f"Warning: Non-positive values detected in dataset {dataset_name}, modifier {modifer}, learning rate {lr}. Skipping logarithmic scale.")

                learning_rate_index += 1
            handles, labels = ax[0][0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center",
                       ncol=8, fontsize=12)
            fig.tight_layout(pad=3.0)
            early_stopping_modifier = "/early-stopping" if use_early_stopping else ""
            if round_no > 0:
                early_stopping_modifier = f"/round-{round_no}/{early_stopping_modifier}"
            os.makedirs(
                f"result/learning_curve/hyperparameter/element-weight-{dataset_name}{early_stopping_modifier}",
                exist_ok=True,
            )
            plt.savefig(
                f"result/learning_curve/hyperparameter/element-weight-{dataset_name}{early_stopping_modifier}/{model_learning_rate}.png"
            )
            plt.close(fig)
            model_learning_rate_index += 1
            learning_rate_index = 0

        dataset_index += 1


def plot_element_bias(use_academic_name=False, use_early_stopping=False, round_no=0, model_version=1):
    dataset_index = 0
    early_stopping_title = "With Early Stopping" if use_early_stopping else ""
    for dataset_name in dataset_names:
        # Find Best Accuracy
        # For Each Model Learning Rate

        if use_early_stopping:
            filter_modifier = modifiers[4:]
            filter_display_modifier = displayed_modifiers[4:]
        else:
            filter_modifier = modifiers[:4]
            filter_display_modifier = displayed_modifiers[:4]
        model_learning_rate_index = 0
        learning_rate_index = 0
        for model_learning_rate in model_learning_rates:
            # For Each Learning Rate
            fig, ax = plt.subplots(5, 4, figsize=(15, 14))
            if use_academic_name:
                fig.suptitle(
                    f"Biases Adjustment of {academic_dataset_names[dataset_index]} - β ={model_learning_rate} {early_stopping_title}",
                    fontdict={"fontsize": 14, "fontweight": "bold"},
                )

            else:
                fig.suptitle(
                    f"Biases Adjustment of {academic_dataset_names[dataset_index]} - β ={model_learning_rate} {early_stopping_title}",
                    fontdict={"fontsize": 14, "fontweight": "bold"},
                )
            learning_rate_index = 0
            for lr in learning_rate:
                for modifier_index, modifer in enumerate(filter_modifier):
                    directory_name = f"result/{dataset_name}/{dataset_name}_{model_version}_{model_learning_rate}{modifer}"
                    epoch_error_file = (
                        f"{directory_name}/round{round_no + 1}/{iteration}-{lr}.csv"
                    )
                    epoch_error = pd.read_csv(epoch_error_file)
                    adjustment_file = (
                        f"{directory_name}/round{round_no + 1}/sample-payload-list-{iteration}-{lr}.csv"
                    )
                    adjustment_data = pd.read_csv(adjustment_file)

                    col_index = modifier_index
                    ax[learning_rate_index][col_index].plot(
                        adjustment_data["material_bias_1"],
                        label="Material Element Bias",
                    )
                    ax[learning_rate_index][col_index].plot(
                        adjustment_data["employee_bias_1"],
                        label="Labor Element Bias",
                    )
                    ax[learning_rate_index][col_index].plot(
                        adjustment_data["capital_cost_bias_1"],
                        label="Utility Cost Element Bias",
                    )
                    ax[learning_rate_index][col_index].plot(
                        adjustment_data["model_bias"],
                        label="Model Bias",
                    )
                    if use_academic_name:
                        ax[learning_rate_index][col_index].set_title(
                            f"{filter_display_modifier[modifier_index]}\n α = {lr}",
                            fontdict={"fontsize": 12},
                        )

                    best_train_epoch = epoch_error[
                        epoch_error["error_percent"]
                        == epoch_error["error_percent"].min()
                    ]
                    best_validate_epoch = epoch_error[
                        epoch_error["validate_error_percent"]
                        == epoch_error["validate_error_percent"].min()
                    ]
                    try:
                        best_train_epoch = best_train_epoch.iloc[0]
                        best_train_epoch = best_train_epoch["epoch"]
                        best_validate_epoch = best_validate_epoch.iloc[0]
                        best_validate_epoch = best_validate_epoch["epoch"]
                    except Exception as e:
                        best_train_epoch = np.nan
                        best_validate_epoch = np.nan
                    ax[learning_rate_index][col_index].set_xlabel(
                        f'Train Err {epoch_error.iloc[-1]["error_percent"]:.2f} , Validate Err {epoch_error.iloc[-1]["validate_error_percent"]:.2f} \n Best Epoch Train {best_train_epoch:.0f} , Validate {best_validate_epoch:.0f}'
                    )
                    ax[learning_rate_index][col_index].set_xticks([])

                learning_rate_index += 1
            handles, labels = ax[0][0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center",
                       ncol=2, fontsize=12)
            fig.tight_layout(pad=2.0)
            early_stopping_modifier = "/early-stopping" if use_early_stopping else ""
            if round_no > 0:
                early_stopping_modifier = f"/round-{round_no}/{early_stopping_modifier}"
            os.makedirs(
                f"result/learning_curve/hyperparameter/bias-{dataset_name}{early_stopping_modifier}",
                exist_ok=True,
            )
            plt.savefig(
                f"result/learning_curve/hyperparameter/bias-{dataset_name}{early_stopping_modifier}/{model_learning_rate}.png"
            )
            plt.close(fig)
            model_learning_rate_index += 1
            learning_rate_index = 0

        dataset_index += 1


def plot_each_case(use_academic_name=False, use_early_stopping=False, round_no=0, model_version=1, selected_case=""):
    modifier_index = modifiers.index(selected_case)
    model_learning_rate_index = 0
    for model_learning_rate in model_learning_rates:
        dataset_index = 0
        fig, ax = plt.subplots(4, 5, figsize=(24, 15))
        fig.suptitle(
            f"Learning Curve of {displayed_modifiers[modifier_index]} Dataset with Model Learning Rate (β) of {model_learning_rate}",
            fontdict={"fontsize": 18, "fontweight": "bold"},
        )

        for dataset_name in dataset_names:
            learning_rate_index = 0
            # For Each Learning Rate

            learning_rate_index = 0
            for lr in (learning_rate):
                directory_name = f"result/{dataset_name}/{dataset_name}_{model_version}_{model_learning_rate}{selected_case}"
                epoch_error_file = (
                    f"{directory_name}/round{round_no + 1}/{iteration}-{lr}.csv"
                )
                epoch_error = pd.read_csv(epoch_error_file)
                ax[dataset_index][learning_rate_index].plot(
                    epoch_error["epoch"],
                    epoch_error["error_percent"],
                    label="Training",
                    linewidth=4
                )
                ax[dataset_index][learning_rate_index].plot(
                    epoch_error["epoch"],
                    epoch_error["validate_error_percent"],
                    label="Validation",
                    linewidth=4
                )
                if learning_rate_index == 0:

                    ax[dataset_index][learning_rate_index].set_title(
                        f"{academic_dataset_names[dataset_index]} \nα={lr}",
                        fontsize=20,
                    )
                else:
                    ax[dataset_index][learning_rate_index].set_title(
                        f"\nα={lr}",
                        fontsize=20,
                    )

                # best_train_epoch = epoch_error[
                #     epoch_error["error_percent"]
                #     == epoch_error["error_percent"].min()
                # ]
                # best_validate_epoch = epoch_error[
                #     epoch_error["validate_error_percent"]
                #     == epoch_error["validate_error_percent"].min()
                # ]
                # try:
                #     best_train_epoch = best_train_epoch.iloc[0]
                #     best_train_epoch = best_train_epoch["epoch"]
                #     best_validate_epoch = best_validate_epoch.iloc[0]
                #     best_validate_epoch = best_validate_epoch["epoch"]
                # except:
                #     best_train_epoch = np.nan
                #     best_validate_epoch = np.nan
                # ax[dataset_index][learning_rate_index].set_xlabel(
                #     f'Train Err {epoch_error.iloc[-1]["error_percent"]:.2f} , Val Err {epoch_error.iloc[-1]["validate_error_percent"]:.2f} \n Lowest Point Iteration on Train {best_train_epoch:.0f} , Validate {best_validate_epoch:.0f}'
                # )
                # ax[learning_rate_index][col_index].set_ylabel(
                #     'Error RMSPE')
                ax[dataset_index][learning_rate_index].set_ylim(0, 115)
                ax[dataset_index][learning_rate_index].set_xlim(0, 100)
                # ax[learning_rate_index][col_index].legend(
                #     loc='lower right')
                learning_rate_index += 1
            dataset_index += 1

        handles, labels = ax[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   ncol=2, fontsize=18)
        fig.tight_layout(pad=2.0)
        early_stopping_modifier = "/early-stopping" if use_early_stopping else ""
        if round_no > 0:
            early_stopping_modifier = f"/round-{round_no}/{early_stopping_modifier}"
        casename = to_kebab_case(displayed_modifiers[modifier_index])
        os.makedirs(
            f"result/learning_curve/by-case/{casename}{early_stopping_modifier}",
            exist_ok=True,
        )
        plt.savefig(
            f"result/learning_curve/by-case/{casename}{early_stopping_modifier}/{model_learning_rate}.png"
        )
        plt.close(fig)
        model_learning_rate_index += 1


def plot_each_case_reversed(use_academic_name=False, use_early_stopping=False, round_no=0, model_version=1, selected_case=""):
    modifier_index = modifiers.index(selected_case)
    model_learning_rate_index = 0
    for model_learning_rate in model_learning_rates:
        dataset_index = 0
        fig, ax = plt.subplots(5, 4, figsize=(24, 20))
        fig.suptitle(
            f"Learning Curve of {displayed_modifiers[modifier_index]} Dataset with Model Learning Rate (β) of {model_learning_rate}",
            fontdict={"fontsize": 20, "fontweight": "bold"},
        )

        learning_rate_index = 0
        for lr in (learning_rate):
            dataset_index = 0
            # For Each Learning Rate

            for dataset_name in dataset_names:
                directory_name = f"result/{dataset_name}/{dataset_name}_{model_version}_{model_learning_rate}{selected_case}"
                epoch_error_file = (
                    f"{directory_name}/round{round_no + 1}/{iteration}-{lr}.csv"
                )
                epoch_error = pd.read_csv(epoch_error_file)
                ax[learning_rate_index][dataset_index].plot(
                    epoch_error["epoch"],
                    epoch_error["error_percent"],
                    label="Training",
                    linewidth=4
                )
                ax[learning_rate_index][dataset_index].plot(
                    epoch_error["epoch"],
                    epoch_error["validate_error_percent"],
                    label="Validation",
                    linewidth=4
                )
                if dataset_index == 0:
                    ax[learning_rate_index][dataset_index].set_title(
                        f"α={lr}\n{academic_dataset_names[dataset_index]}",
                        fontsize=24,
                    )
                else:
                    ax[learning_rate_index][dataset_index].set_title(
                        f"\n{academic_dataset_names[dataset_index]}",
                        fontsize=24,
                    )

                # best_train_epoch = epoch_error[
                #     epoch_error["error_percent"]
                #     == epoch_error["error_percent"].min()
                # ]
                # best_validate_epoch = epoch_error[
                #     epoch_error["validate_error_percent"]
                #     == epoch_error["validate_error_percent"].min()
                # ]
                # try:
                #     best_train_epoch = best_train_epoch.iloc[0]
                #     best_train_epoch = best_train_epoch["epoch"]
                #     best_validate_epoch = best_validate_epoch.iloc[0]
                #     best_validate_epoch = best_validate_epoch["epoch"]
                # except:
                #     best_train_epoch = np.nan
                #     best_validate_epoch = np.nan
                # ax[learning_rate_index][dataset_index].set_xlabel(
                #     f'Train Err {epoch_error.iloc[-1]["error_percent"]:.2f} , Val Err {epoch_error.iloc[-1]["validate_error_percent"]:.2f} \n Lowest Point Iteration on Train {best_train_epoch:.0f} , Validate {best_validate_epoch:.0f}'
                # )
                # ax[learning_rate_index][col_index].set_ylabel(
                #     'Error RMSPE')
                ax[learning_rate_index][dataset_index].set_ylim(0, 115)
                ax[learning_rate_index][dataset_index].set_xlim(0, 100)
                # ax[learning_rate_index][col_index].legend(
                #     loc='lower right')
                dataset_index += 1
            learning_rate_index += 1

        handles, labels = ax[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   ncol=2, fontsize=18)
        fig.tight_layout(pad=2.0)
        early_stopping_modifier = "/early-stopping" if use_early_stopping else ""
        if round_no > 0:
            early_stopping_modifier = f"/round-{round_no}/{early_stopping_modifier}"
        casename = to_kebab_case(displayed_modifiers[modifier_index])
        os.makedirs(
            f"result/learning_curve/by-case/{casename}{early_stopping_modifier}",
            exist_ok=True,
        )
        plt.savefig(
            f"result/learning_curve/by-case/{casename}{early_stopping_modifier}/{model_learning_rate}-reversed.png"
        )
        plt.close(fig)
        model_learning_rate_index += 1


def plot_each_element_weights(use_academic_name=False, use_early_stopping=False, round_no=0, model_version=1, selected_case="", learning_rate=0.01):
    early_stopping_title = "With Early Stopping" if use_early_stopping else ""
    model_learning_rate_index = 0
    modifier_index = modifiers.index(selected_case)
    this_graph_palette = sns.color_palette("husl", 9)
    sns.set_palette(this_graph_palette)
    for model_learning_rate in model_learning_rates:
        # For Each Learning Rate
        dataset_index = 0
        fig, ax = plt.subplots(1, 4, figsize=(18, 5))
        if use_academic_name:
            fig.suptitle(
                f"Model Element Weight Adjustment of {displayed_modifiers[modifier_index]} {academic_dataset_names[dataset_index]} \n β ={model_learning_rate} α = {learning_rate} {early_stopping_title}",
                fontdict={"fontsize": 16, "fontweight": "bold"},
            )
        else:
            fig.suptitle(
                f"Model Element Weight Adjustment of {displayed_modifiers[modifier_index]} {academic_dataset_names[dataset_index]} \n β ={model_learning_rate}α = {learning_rate} {early_stopping_title}",
                fontdict={"fontsize": 16, "fontweight": "bold"},
            )

        for dataset_name in dataset_names:
            directory_name = f"result/{dataset_name}/{dataset_name}_{model_version}_{model_learning_rate}{selected_case}"
            epoch_error_file = (
                f"{directory_name}/round{round_no + 1}/{iteration}-{learning_rate}.csv"
            )
            epoch_error = pd.read_csv(epoch_error_file)
            adjustment_file = (
                f"{directory_name}/round{round_no + 1}/sample-payload-list-{iteration}-{learning_rate}.csv"
            )
            adjustment_data = pd.read_csv(adjustment_file)

            # Material Weights
            ax[dataset_index].plot(
                adjustment_data["material_weight_1"],
                label="A Crab"
            )
            ax[dataset_index].plot(
                adjustment_data["material_weight_2"],
                label="C Crab"
            )
            # In Case of Not Exist Material Weight 4 in Actual Case
            try:
                ax[dataset_index].plot(
                    adjustment_data["material_weight_4"],
                    label="Loss Crab"
                )
                ax[dataset_index].plot(
                    adjustment_data["material_weight_3"],
                    label="Small Crab"
                )
            except Exception as e:
                print(e)
                ax[dataset_index].plot(
                    adjustment_data["material_weight_3"],
                    label="Small and Loss Crab"
                )

            # Utility Cost Weights
            ax[dataset_index].plot(
                adjustment_data["capital_cost_weight_1"],
                label="Electricity Cost",

            )
            ax[dataset_index].plot(
                adjustment_data["capital_cost_weight_2"],
                label="Water Supply Cost",

            )

            # Employee Weights
            number_of_employee = find_amount_of_employee(
                adjustment_data)
            employee_colors = ["PaleTurquoise", "Cyan",
                               "LightCyan", "Turquoise", "DarkTurquoise"]
            emp_index = 0
            for emp in range(number_of_employee):
                ax[dataset_index].plot(
                    adjustment_data[f"employee_weight_{emp + 1}"],
                    label=f"Employee {emp}",
                    linestyle="--",
                    # color=employee_colors[emp % len(employee_colors)]
                )
                emp_index += 1
            if use_academic_name:
                ax[dataset_index].set_title(
                    f"{academic_dataset_names[dataset_index]}",
                    fontsize=18,
                )
            best_train_epoch = epoch_error[
                epoch_error["error_percent"]
                == epoch_error["error_percent"].min()
            ]
            best_validate_epoch = epoch_error[
                epoch_error["validate_error_percent"]
                == epoch_error["validate_error_percent"].min()
            ]
            try:
                best_train_epoch = best_train_epoch.iloc[0]
                best_train_epoch = best_train_epoch["epoch"]
                best_validate_epoch = best_validate_epoch.iloc[0]
                best_validate_epoch = best_validate_epoch["epoch"]
            except Exception as e:
                best_train_epoch = np.nan
                best_validate_epoch = np.nan
            # ax[dataset_index].set_xlabel(
            #     f'Train Err {epoch_error.iloc[-1]["error_percent"]:.2f} , Validate Err {epoch_error.iloc[-1]["validate_error_percent"]:.2f} \n Best Epoch Train {best_train_epoch:.0f} , Validate {best_validate_epoch:.0f}'
            # )
            ax[dataset_index].set_xticks([])
            ax[dataset_index].set_ylim(0, 10000)
            # Ensure all values are positive before setting the y-axis to logarithmic scale
            # if (adjustment_data.select_dtypes(include=[np.number]) > 0).all().all():
            #     ax[dataset_index].set_yscale('log')
            # else:
            #     print(
            #         f"Warning: Non-positive values detected in dataset {dataset_name}. Skipping logarithmic scale.")

            dataset_index += 1

        handles, labels = ax[3].get_legend_handles_labels()
        print('labels', labels)
        # Group employee labels together
        employee_labels = [
            label for label in labels if label.startswith("Employee")]
        other_labels = [
            label for label in labels if not label.startswith("Employee")]

        # Combine employee labels into a single entry
        if employee_labels:
            other_labels.append("Employees")

        labels = other_labels
        fig.legend(handles, labels, loc="lower center",
                   ncol=8, fontsize=16)
        fig.tight_layout(pad=2.0)
        early_stopping_modifier = "/early-stopping" if use_early_stopping else ""
        if round_no > 0:
            early_stopping_modifier = f"/round-{round_no}/{early_stopping_modifier}"
        casename = to_kebab_case(displayed_modifiers[modifier_index])
        os.makedirs(
            f"result/learning_curve/hyperparameter/element-weight-{learning_rate}{early_stopping_modifier}",
            exist_ok=True,
        )
        plt.savefig(
            f"result/learning_curve/hyperparameter/element-weight-{learning_rate}{early_stopping_modifier}/{casename}-{model_learning_rate}.png"
        )
        plt.close(fig)
        model_learning_rate_index += 1

        dataset_index += 1

    sns.set_palette(sns.color_palette())


def plot_each_element_weights_select(use_academic_name=False, use_early_stopping=False, round_no=0, model_version=1, selected_case="", learning_rate=0.01, is_tail=False, round=1000):
    early_stopping_title = "With Early Stopping" if use_early_stopping else ""
    model_learning_rate_index = 0
    modifier_index = modifiers.index(selected_case)
    this_graph_palette = sns.color_palette("husl", 9)
    sns.set_palette(this_graph_palette)
    prefix_modifier = is_tail and f"Last {round} Round" or f"First {round} Round"
    for model_learning_rate in model_learning_rates:
        # For Each Learning Rate
        dataset_index = 0
        fig, ax = plt.subplots(1, 4, figsize=(18, 5))
        if use_academic_name:
            fig.suptitle(
                f"Model Element Weight Adjustment of {displayed_modifiers[modifier_index]} {academic_dataset_names[dataset_index]} \n β ={model_learning_rate} α = {learning_rate} {early_stopping_title}",
                fontdict={"fontsize": 16, "fontweight": "bold"},
            )
        else:
            fig.suptitle(
                f"Model Element Weight Adjustment of {displayed_modifiers[modifier_index]} {academic_dataset_names[dataset_index]} \n β ={model_learning_rate}α = {learning_rate} {early_stopping_title}",
                fontdict={"fontsize": 16, "fontweight": "bold"},
            )

        for dataset_name in dataset_names:
            directory_name = f"result/{dataset_name}/{dataset_name}_{model_version}_{model_learning_rate}{selected_case}"
            epoch_error_file = (
                f"{directory_name}/round{round_no + 1}/{iteration}-{learning_rate}.csv"
            )
            epoch_error = pd.read_csv(epoch_error_file)
            adjustment_file = (
                f"{directory_name}/round{round_no + 1}/sample-payload-list-{iteration}-{learning_rate}.csv"
            )
            adjustment_data = pd.read_csv(adjustment_file)
            if is_tail:
                # Select the last 1000 rows of the DataFrame
                adjustment_data = adjustment_data.tail(round)
            else:
                adjustment_data = adjustment_data.head(round)
            # Material Weights
            ax[dataset_index].plot(
                adjustment_data["material_weight_1"],
                label="A Crab"
            )
            ax[dataset_index].plot(
                adjustment_data["material_weight_2"],
                label="C Crab"
            )
            # In Case of Not Exist Material Weight 4 in Actual Case
            try:
                ax[dataset_index].plot(
                    adjustment_data["material_weight_4"],
                    label="Loss Crab"
                )
                ax[dataset_index].plot(
                    adjustment_data["material_weight_3"],
                    label="Small Crab"
                )
            except Exception as e:
                print(e)
                ax[dataset_index].plot(
                    adjustment_data["material_weight_3"],
                    label="Small and Loss Crab"
                )
            # Employee Weights
            number_of_employee = find_amount_of_employee(
                adjustment_data)
            employee_colors = ["PaleTurquoise", "Cyan",
                               "LightCyan", "Turquoise", "DarkTurquoise"]
            emp_index = 0
            for emp in range(number_of_employee):
                ax[dataset_index].plot(
                    adjustment_data[f"employee_weight_{emp + 1}"],
                    label=f"Employee {emp}",
                    # color=employee_colors[emp % len(employee_colors)]
                )
                emp_index += 1
            # Utility Cost Weights
            ax[dataset_index].plot(
                adjustment_data["capital_cost_weight_1"],
                label="Electricity Cost",

            )
            ax[dataset_index].plot(
                adjustment_data["capital_cost_weight_2"],
                label="Water Supply Cost",

            )
            if use_academic_name:
                ax[dataset_index].set_title(
                    f"{academic_dataset_names[dataset_index]}",
                    fontsize=16,
                )
            best_train_epoch = epoch_error[
                epoch_error["error_percent"]
                == epoch_error["error_percent"].min()
            ]
            best_validate_epoch = epoch_error[
                epoch_error["validate_error_percent"]
                == epoch_error["validate_error_percent"].min()
            ]
            try:
                best_train_epoch = best_train_epoch.iloc[0]
                best_train_epoch = best_train_epoch["epoch"]
                best_validate_epoch = best_validate_epoch.iloc[0]
                best_validate_epoch = best_validate_epoch["epoch"]
            except Exception as e:
                best_train_epoch = np.nan
                best_validate_epoch = np.nan
            # ax[dataset_index].set_xlabel(
            #     f'Train Err {epoch_error.iloc[-1]["error_percent"]:.2f} , Validate Err {epoch_error.iloc[-1]["validate_error_percent"]:.2f} \n Best Epoch Train {best_train_epoch:.0f} , Validate {best_validate_epoch:.0f}'
            # )
            ax[dataset_index].set_xticks([])
            ax[dataset_index].set_ylim(0, 10000)
            # Ensure all values are positive before setting the y-axis to logarithmic scale
            # if (adjustment_data.select_dtypes(include=[np.number]) > 0).all().all():
            #     ax[dataset_index].set_yscale('log')
            # else:
            #     print(
            #         f"Warning: Non-positive values detected in dataset {dataset_name}. Skipping logarithmic scale.")

            dataset_index += 1

        handles, labels = ax[3].get_legend_handles_labels()
        print('labels', labels)
        # Group employee labels together
        employee_labels = [
            label for label in labels if label.startswith("Employee")]
        other_labels = [
            label for label in labels if not label.startswith("Employee")]

        # Combine employee labels into a single entry
        if employee_labels:
            other_labels.append("Employees")

        labels = other_labels
        fig.legend(handles, labels, loc="lower center",
                   ncol=8, fontsize=12)
        fig.tight_layout(pad=2.0)
        early_stopping_modifier = "/early-stopping" if use_early_stopping else ""
        if round_no > 0:
            early_stopping_modifier = f"/round-{round_no}/{early_stopping_modifier}"
        prefix_filename = is_tail and f"tail-{round}" or f"head-{round}"
        os.makedirs(
            f"result/learning_curve/hyperparameter/element-weight-{learning_rate}{early_stopping_modifier}",
            exist_ok=True,
        )
        plt.savefig(
            f"result/learning_curve/hyperparameter/element-weight-{learning_rate}{early_stopping_modifier}/{prefix_filename}-{model_learning_rate}.png"
        )
        plt.close(fig)
        model_learning_rate_index += 1

        dataset_index += 1

    sns.set_palette(sns.color_palette())


def plot_each_weights(use_academic_name=False, use_early_stopping=False, round_no=0, model_version=1, selected_case="", learning_rate=0.01):
    early_stopping_title = "With Early Stopping" if use_early_stopping else ""
    model_learning_rate_index = 0
    modifier_index = modifiers.index(selected_case)
    this_graph_palette = sns.color_palette("husl", 4)
    sns.set_palette(this_graph_palette)
    for model_learning_rate in model_learning_rates:
        # For Each Learning Rate
        dataset_index = 0
        fig, ax = plt.subplots(1, 4, figsize=(18, 5))
        if use_academic_name:
            fig.suptitle(
                f"Model Level Weight Adjustment of {displayed_modifiers[modifier_index]} {academic_dataset_names[dataset_index]} \n β ={model_learning_rate} α = {learning_rate} {early_stopping_title}",
                fontdict={"fontsize": 16, "fontweight": "bold"},
            )
        else:
            fig.suptitle(
                f"Model Level Weight Adjustment of {displayed_modifiers[modifier_index]} {academic_dataset_names[dataset_index]} \n β ={model_learning_rate}α = {learning_rate} {early_stopping_title}",
                fontdict={"fontsize": 16, "fontweight": "bold"},
            )

        for dataset_name in dataset_names:
            directory_name = f"result/{dataset_name}/{dataset_name}_{model_version}_{model_learning_rate}{selected_case}"
            epoch_error_file = (
                f"{directory_name}/round{round_no + 1}/{iteration}-{learning_rate}.csv"
            )
            epoch_error = pd.read_csv(epoch_error_file)
            adjustment_file = (
                f"{directory_name}/round{round_no + 1}/sample-payload-list-{iteration}-{learning_rate}.csv"
            )
            adjustment_data = pd.read_csv(adjustment_file)

            ax[dataset_index].plot(
                adjustment_data["model_weight_1"],
                label="Material Element Weight",
            )
            ax[dataset_index].plot(
                adjustment_data["model_weight_2"],
                label="Labor Element Weight", linewidth=3,
                # linestyle='-',
            )
            ax[dataset_index].plot(
                adjustment_data["model_weight_3"],
                label="Utiltiy Cost Element Weight",
                linewidth=3,
                linestyle=':', color=sns.palettes.hls_palette(6)[4]
            )
            if use_academic_name:
                ax[dataset_index].set_title(
                    f"{academic_dataset_names[dataset_index]}",
                    fontsize=20,
                )
            best_train_epoch = epoch_error[
                epoch_error["error_percent"]
                == epoch_error["error_percent"].min()
            ]
            best_validate_epoch = epoch_error[
                epoch_error["validate_error_percent"]
                == epoch_error["validate_error_percent"].min()
            ]
            try:
                best_train_epoch = best_train_epoch.iloc[0]
                best_train_epoch = best_train_epoch["epoch"]
                best_validate_epoch = best_validate_epoch.iloc[0]
                best_validate_epoch = best_validate_epoch["epoch"]
            except Exception as e:
                best_train_epoch = np.nan
                best_validate_epoch = np.nan
            # ax[dataset_index].set_xlabel(
            #     f'Train Err {epoch_error.iloc[-1]["error_percent"]:.2f} , Validate Err {epoch_error.iloc[-1]["validate_error_percent"]:.2f} \n Best Epoch Train {best_train_epoch:.0f} , Validate {best_validate_epoch:.0f}'
            # )
            ax[dataset_index].set_xticks([])
            ax[dataset_index].set_ylim(0, 30)
            # Ensure all values are positive before setting the y-axis to logarithmic scale
            # if (adjustment_data.select_dtypes(include=[np.number]) > 0).all().all():
            #     ax[dataset_index].set_yscale('log')
            # else:
            #     print(
            #         f"Warning: Non-positive values detected in dataset {dataset_name}. Skipping logarithmic scale.")

            dataset_index += 1

        handles, labels = ax[3].get_legend_handles_labels()
        print('labels', labels)
        # Group employee labels together
        employee_labels = [
            label for label in labels if label.startswith("Employee")]
        other_labels = [
            label for label in labels if not label.startswith("Employee")]

        # Combine employee labels into a single entry
        if employee_labels:
            other_labels.append("Employees")

        labels = other_labels
        fig.legend(handles, labels, loc="lower center",
                   ncol=8, fontsize=16)
        fig.tight_layout(pad=2.0)
        early_stopping_modifier = "/early-stopping" if use_early_stopping else ""
        if round_no > 0:
            early_stopping_modifier = f"/round-{round_no}/{early_stopping_modifier}"
        os.makedirs(
            f"result/learning_curve/hyperparameter/model-weight-{learning_rate}{early_stopping_modifier}",
            exist_ok=True,
        )
        plt.savefig(
            f"result/learning_curve/hyperparameter/model-weight-{learning_rate}{early_stopping_modifier}/{model_learning_rate}-{selected_case}.png"
        )
        plt.close(fig)
        model_learning_rate_index += 1

        dataset_index += 1

    sns.set_palette(sns.color_palette())
