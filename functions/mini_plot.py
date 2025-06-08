import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid", font_scale=1)
this_graph_palette = sns.color_palette("husl", 4)
sns.set_palette(this_graph_palette)


def plotting_learning_curve(epoch_error, element_learning_rate, model_learning_rate):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].plot(
        epoch_error["epoch"],
        epoch_error["error"],
        label="Training",
        linewidth=2
    )
    ax[0].plot(
        epoch_error["epoch"],
        epoch_error["validate_error"],
        label="Validation", linewidth=2
    )
    ax[0].set_title(
        f"Learning Curve {model_learning_rate}/{element_learning_rate} in MSE",
        fontdict={"fontsize": 12},
    )
    ax[1].plot(
        epoch_error["epoch"],
        epoch_error["error_percent"],
        label="Training",
        linewidth=2
    )
    ax[1].plot(
        epoch_error["epoch"],
        epoch_error["validate_error_percent"],
        label="Validation", linewidth=2
    )
    ax[1].set_ylim(
        0, 100)
    ax[1].set_xlim(
        0, 100)

    ax[1].set_title(
        f"Learning Curve {model_learning_rate}/{element_learning_rate} in RMSPE",
        fontdict={"fontsize": 12},
    )
    ax[0].legend(loc="lower right")
    ax[1].legend(loc="lower right")
    fig.tight_layout(pad=3.0)


def plot_model_level_weight(adjustment_data, epoch_error):
    fig, ax = plt.subplots(1, 3, figsize=(14, 3))

    ax[0].plot(
        adjustment_data["model_weight_1"],
        label="Material Element Weight",
    )
    ax[0].plot(
        adjustment_data["model_weight_2"],
        label="Labor Element Weight",
    )
    ax[0].plot(
        adjustment_data["model_weight_3"],
        label="Utiltiy Cost Element Weight",
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

    ax[0].set_xlabel(
        f'Train Err {epoch_error.iloc[-1]["error_percent"]:.2f} , Validate Err {epoch_error.iloc[-1]["validate_error_percent"]:.2f} \n Best Epoch Train {best_train_epoch:.0f} , Validate {best_validate_epoch:.0f}'
    )
    ax[0].set_xticks([])
    ax[0].legend(loc="lower right")
    ax[0].set_title(
        "All Model Level Weight",
        fontdict={"fontsize": 12},
    )

    ax[1].plot(
        adjustment_data["model_weight_1"],
        label="Material Element Weight",
    )
    ax[1].legend(loc="lower right")
    ax[1].set_title(
        "Material Element Weight",
        fontdict={"fontsize": 12},
    )

    ax[2].plot(
        adjustment_data["model_weight_2"],
        label="Labor Element Weight",
    )
    ax[2].plot(
        adjustment_data["model_weight_3"],
        label="Utiltiy Cost Element Weight",
    )
    ax[2].legend(loc="lower right")
    ax[2].set_title(
        "Labor and Utility Cost Element Weight",
        fontdict={"fontsize": 12},
    )


def plot_element_level_weight(adjustment_data, material_columns, labor_columns, utility_columns):

    fig, ax = plt.subplots(1, 3, figsize=(14, 3))
    for column in material_columns:
        ax[0].plot(
            adjustment_data[column],
            label=column,
        )

    for column in labor_columns:
        ax[1].plot(
            adjustment_data[column],
            label=column,
        )

    for column in utility_columns:
        ax[2].plot(
            adjustment_data[column],
            label=column,
        )

    ax[0].legend(loc="lower right")
    ax[2].legend(loc="lower right")

    ax[0].set_title(
        "Material Element Weight",
        fontdict={"fontsize": 12},
    )

    ax[1].set_title(
        "Labor Element Weight",
        fontdict={"fontsize": 12},
    )

    ax[2].set_title(
        "Utility Cost Element Weight",
        fontdict={"fontsize": 12},
    )

    fig.tight_layout(pad=2.0)
