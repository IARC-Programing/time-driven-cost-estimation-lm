import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font="Noto Sans",
              font_scale=1)
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
