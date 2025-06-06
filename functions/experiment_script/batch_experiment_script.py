import importlib
import sys
import getopt
import os
import rsg_breakpoint_experiment as rbe
importlib.reload(rbe)

learning_rate = [0.005, 0.01, 0.05, 0.1, 0.5]
patience_round = 50
outlier_index = 1.5


def run_batch_experiment(model_learning_rate,
                         augmentation=False,
                         remove_outlier_qtr=False,
                         round_number=1,
                         iteration=200,
                         use_model_weight=True,
                         dataset_name="simple",
                         experiment_version=2,
                         use_early_stopping=False
                         ):

    dataset_directory = f'data/{dataset_name}'
    output_group_directory = f'result/{dataset_name}'
    os.makedirs(output_group_directory, exist_ok=True)
    output_directory = f'{output_group_directory}/{dataset_name}_{experiment_version}'
    output_directory += f"_{model_learning_rate}"
    if remove_outlier_qtr:
        output_directory += "_remove_outlier"
    if augmentation:
        output_directory += "_augmented"
    if use_early_stopping:
        output_directory += "_early_stopping"

    try:
        rbe.run_experiment(data_directory=dataset_directory, epoch=iteration,
                           learning_rate_list=learning_rate, round_number=round_number,
                           folder_name=output_directory, breakpoint=iteration / 2,
                           early_stopping=use_early_stopping, patience_round=patience_round,
                           augmentation=augmentation,
                           outlier_index=outlier_index,
                           remove_outlier_qtr=remove_outlier_qtr,
                           use_model_weight=use_model_weight,
                           model_learning_rate=model_learning_rate)
    except Exception as e:
        print("Error in Experiment", e)
        pass


def run_in_many_learning_rate(dataset_name="simple", experiment_version=2, round_number=1, iteration=200):
    model_learning_rate_list = [0.0000001, 0.00000001]
    early_stopping_list = [False, True]
    for model_learning_rate in model_learning_rate_list:
        for early_stopping in early_stopping_list:
            # Normal
            print(
                f'LL: {model_learning_rate} / Normal / Eearly Stopping: {early_stopping}')
            run_batch_experiment(model_learning_rate=model_learning_rate,
                                 augmentation=False,
                                 remove_outlier_qtr=False,
                                 round_number=round_number,
                                 iteration=iteration,
                                 use_model_weight=True,
                                 dataset_name=dataset_name,
                                 experiment_version=experiment_version,
                                 use_early_stopping=early_stopping
                                 )
            # Remove Outlier
            print(
                f'LL: {model_learning_rate} / Remove Outlier / Eearly Stopping: {early_stopping}')
            run_batch_experiment(model_learning_rate=model_learning_rate,
                                 augmentation=False,
                                 remove_outlier_qtr=True,
                                 round_number=round_number,
                                 iteration=iteration,
                                 use_model_weight=True,
                                 dataset_name=dataset_name,
                                 experiment_version=experiment_version,
                                 use_early_stopping=early_stopping
                                 )
            # Augmentation
            print(
                f'LL: {model_learning_rate} / Augmentation / Eearly Stopping: {early_stopping}')
            run_batch_experiment(model_learning_rate=model_learning_rate,
                                 augmentation=True,
                                 remove_outlier_qtr=False,
                                 round_number=round_number,
                                 iteration=iteration,
                                 use_model_weight=True,
                                 dataset_name=dataset_name,
                                 experiment_version=experiment_version,
                                 use_early_stopping=early_stopping
                                 )
            print(
                f'LL: {model_learning_rate} / Remove Outlier & Augmentation / Eearly Stopping: {early_stopping}')
            # Remove Outlier and Augmentation
            run_batch_experiment(model_learning_rate=model_learning_rate,
                                 augmentation=True,
                                 remove_outlier_qtr=True,
                                 round_number=round_number,
                                 iteration=iteration,
                                 use_model_weight=True,
                                 dataset_name=dataset_name,
                                 experiment_version=experiment_version,
                                 use_early_stopping=early_stopping
                                 )


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:v:r:i:", [
                                   "dataset=", "version=", 'round=', 'iteration='])
    except getopt.GetoptError as err:
        print(err)
    # Process options

    for opt, arg in opts:
        if opt in ("-d", "--dataset"):
            dataset_name = arg
        elif opt in ("-v", "--version"):
            experiment_version = int(arg)
        elif opt in ("-r", "--round"):
            round_number = int(arg)
        elif opt in ("-i", "--iteration"):
            iteration = int(arg)

    print(
        f"Dataset: {dataset_name} / Version: {experiment_version} / Round: {round_number} / Iteration: {iteration}")
    run_in_many_learning_rate(dataset_name=dataset_name,
                              experiment_version=experiment_version,
                              round_number=round_number,
                              iteration=iteration)
