import pandas as pd
import numpy as np
import importlib
import sys
import time
import os

# fmt:off
sys.path.append('../model')
sys.path.append('../functions/matrix_generator')
sys.path.append('../functions/data_extractor')

import tdce_model as tdce
import material_fc_layer as mfl
import employee_fc_layer as efl
import capital_fc_layer as cfl
import loss
import cost_matrix_class as cmc
import display_input_variation as diva
import viyacrab_augmentation as viya
import adjust_data as ajd
import result_display as rd

importlib.reload(tdce)
importlib.reload(mfl)
importlib.reload(efl)
importlib.reload(cfl)
importlib.reload(loss)
importlib.reload(tdce)
importlib.reload(cmc)
importlib.reload(diva)
importlib.reload(viya)
importlib.reload(ajd)
importlib.reload(rd)
# fmt:on


def inital_layer(
    material_cost_matrix,
    employee_cost_matrix,
    capital_cost_matrix,
):
    total_col = 0
    # Material FC Layer
    row, high, col = material_cost_matrix.shape
    material_layer_1 = mfl.MaterialFCLayer(col, 1)
    # material_layer_1.annotate(material_cost_matrix, material_amount_matrix)
    total_col += col

    # Monthy Employee FC Layer
    row, high, col = employee_cost_matrix.shape
    employee_layer_1 = efl.EmployeeFCLayer(col, 1, 8)
    total_col += col
    # monthy_employee_layer_1.annotate(monthy_employee_cost_matrix, duration_matrix)

    # Capital Cost  FC Layer
    row, high, col = capital_cost_matrix.shape
    capital_cost_layer1 = cfl.CapitalCostFCLayer(col, 1, 21)
    total_col += col
    # capital_cost_layer1.annotate(
    #     capital_cost_matrix, life_time_matrix, machine_hour_matrix, duration_matrix)

    return (
        material_layer_1,
        employee_layer_1,
        capital_cost_layer1,
    )


def create_learning(
    epoch,
    learning_rate,
    inside_learning_rate,
    material_cost_matrix,
    material_amount_matrix,
    employee_cost_matrix,
    employee_duration_matrix,
    employee_day_amount_matrix,
    capital_cost_matrix,
    day_amount_matrix,
    capital_duration_matrix,
    result_matrix,
    folder_name,
    validation_payload,
    breakpoint,
    early_stopping=False,
    patience_round=10,
    use_model_weight=False,
):
    # Initial Model
    tdce_model = tdce.TDCEModel()
    # Initial Layer
    (
        material_layer_1,
        employee_layer_1,
        capital_cost_layer1,
    ) = inital_layer(
        capital_cost_matrix=capital_cost_matrix,
        employee_cost_matrix=employee_cost_matrix,
        material_cost_matrix=material_cost_matrix,
    )

    tdce_model.inital_inside_element(
        material_layer=material_layer_1,
        capital_cost_layer=capital_cost_layer1,
        employee_layer=employee_layer_1,
    )

    tdce_model.use(loss=loss.mse, loss_prime=loss.mse_prime,
                   loss_percent=loss.rmspe)

    tdce_model.set_learning_rate(
        inside_learning_rate[0],
        inside_learning_rate[1],
        inside_learning_rate[2],
    )

    start_time = time.time()

    if early_stopping:
        tdce_model.activate_early_stopping()
        tdce_model.edit_patience_round(patience_round)

    if use_model_weight:
        tdce_model.activete_model_weight()

    tdce_model.fit_with_validation(
        epoch=epoch,
        learning_rate=learning_rate,
        material_amount_matrix=material_amount_matrix,
        material_cost_matrix=material_cost_matrix,
        employee_cost_matrix=employee_cost_matrix,
        employee_duration_matrix=employee_duration_matrix,
        employee_day_amount_matrix=employee_day_amount_matrix,
        result_matrix=result_matrix,
        capital_cost_matrix=capital_cost_matrix,
        day_amount_matrix=day_amount_matrix,
        validation_payload=validation_payload,
        capital_cost_duration_matrix=capital_duration_matrix,
    )

    end_time = time.time()
    time_usage = end_time - start_time
    print(f"Learning Rate: {learning_rate} & {inside_learning_rate}")
    print(f"Time Using {time_usage} Second")

    error_list = tdce_model.get_epoch_error()
    sample_error_list = tdce_model.get_sample_error()

    error_df = pd.DataFrame(error_list)
    sample_error_df = pd.DataFrame(sample_error_list)

    before_breakpoint_error = error_df[error_df["epoch"] < breakpoint]
    bb_minimum_error = before_breakpoint_error["error"].min()
    bb_minimum_percent_error = before_breakpoint_error["error_percent"].min()
    bb_minimum_validate_error = before_breakpoint_error["validate_error"].min()
    bb_minimum_validate_percent_error = before_breakpoint_error[
        "validate_error_percent"
    ].min()

    minimum_error = error_df["error"].min()
    minimum_percent_error = error_df["error_percent"].min()
    minimum_validate_error = error_df["validate_error"].min()
    minimum_validate_percent_error = error_df["validate_error_percent"].min()

    try:
        os.mkdir(f"{folder_name}")
    except FileExistsError:
        print("Folder is Exist")
        pass

    error_df.to_csv(f"{folder_name}/{epoch}-{inside_learning_rate[0]}.csv")
    sample_error_df.to_csv(
        f"{folder_name}/error-list-{epoch}-{inside_learning_rate[0]}.csv"
    )

    sample_payload = tdce_model.get_sample_payload()
    sample_payload_df = pd.DataFrame(sample_payload)
    sample_payload_df.to_csv(
        f"{folder_name}/sample-payload-list-{epoch}-{inside_learning_rate[0]}.csv",
        index=False,
    )

    return (
        minimum_error,
        time_usage,
        minimum_percent_error,
        minimum_validate_error,
        minimum_validate_percent_error,
        bb_minimum_error,
        bb_minimum_percent_error,
        bb_minimum_validate_error,
        bb_minimum_validate_percent_error,
    )


def run_experiment(
    data_directory,
    epoch,
    folder_name,
    round_number,
    learning_rate_list,
    model_learning_rate,
    breakpoint,
    early_stopping=False,
    patience_round=10,
    augmentation=False,
    remove_outlier_qtr=False,
    outlier_index=1.5,
    use_model_weight=False,
):
    print("Run Experiment Script is initial !")
    try:
        os.mkdir(folder_name)
    except FileExistsError:
        print("Folder is Exist")
        pass
    cost_generator = cmc.CostMatrixGenerator()
    cost_generator.change_data_directory(data_directory)
    cost_generator.load_data()
    if remove_outlier_qtr:
        cost_generator.remove_outlier_iqr(outlier_index)
        (
            new_process_df,
            new_employee_usage,
            new_material_usage,
            new_capital_cost_usage,
        ) = cost_generator.get_data()

        (new_capital_cost_usage, new_employee_usage, new_material_usage) = (
            ajd.adjust_to_match_process(
                capital_cost_usage=new_capital_cost_usage,
                employee_usage=new_employee_usage,
                material_usage=new_material_usage,
                new_process_df=new_process_df,
            )
        )

        new_variation = diva.display_input_variation(
            new_process_df,
            new_material_usage,
            new_employee_usage,
            new_capital_cost_usage,
        )
        new_variation.to_csv(f"{folder_name}/data_variation_after_outlier.csv")
        new_process_df.to_csv(f"{folder_name}/process_df_after_outlier.csv")

        try:
            print("Data Variation After Outlier Removed")
            display(new_variation)
        except Exception as e:
            print("This is not Jupyter Notebook", e)

    # Save Variation of Overall Data
    input_variation = diva.display_input_variation_by_directory(data_directory)
    input_variation.to_csv(f"{folder_name}/data_variation.csv")

    overall_result = []

    # Iteration over the set of Learning Rates
    for learn_r in learning_rate_list:
        round_err_list = []
        round_time_list = []
        round_percent_list = []
        round_validate_err_list = []
        round_validate_err_percent_list = []
        round_brakpoint_err_list = []
        round_brakpoint_err_percent_list = []
        round_brakpoint_val_err_list = []
        round_brakpoint_val_err_percent_list = []
        round_err_dict = {"learning_rate": str(learn_r)}
        for round in range(round_number):
            ##
            (
                train_process_df,
                train_employee_usage,
                train_material_usage,
                train_capital_cost,
                validate_process_df,
                validate_employee_usage,
                validate_material_usage,
                validate_capital_cost,
            ) = cost_generator.train_test_split_without_matrix(0.7)
            # Generate Cost Matrix for Validation Set
            validation_payload = cost_generator.get_validation_payload(
                validate_process_df
            )

            # Display Variation of Train Data
            train_variation = diva.display_input_variation(
                train_process_df,
                train_material_usage,
                train_employee_usage,
                train_capital_cost,
            )
            train_variation.to_csv(
                f"{folder_name}/train_data_variation_{round}.csv")
            train_process_df.to_csv(
                f"{folder_name}/train_process_df_{round}.csv")

            # Display Variation of Validation Data
            validate_variation = diva.display_input_variation(
                validate_process_df,
                validate_material_usage,
                validate_employee_usage,
                validate_capital_cost,
            )

            validate_variation.to_csv(
                f"{folder_name}/validate_data_variation_{round}.csv"
            )
            if augmentation:
                # TODO:  Increase the Generalization of the Model
                # Augmented the Imbalance Class of Training Data
                train_process_df.to_csv(
                    f"{folder_name}/train_process_df_before_augmented_{round}.csv"
                )
                train_process_df = viya.vy_training_augmentation(
                    train_process_df)
                # Display Variation of Train Data After Augmented
                train_variation = diva.display_input_variation(
                    train_process_df,
                    train_material_usage,
                    train_employee_usage,
                    train_capital_cost,
                )
                train_variation.to_csv(
                    f"{folder_name}/train_data_variation_after_augmented_{round}.csv"
                )
                train_process_df.to_csv(
                    f"{folder_name}/train_process_df_after_augmented_{round}.csv"
                )

            # Generate Matrix From Training Set
            (
                material_cost_matrix,
                material_amount_matrix,
                employee_cost_matrix,
                employee_duration_matrix,
                employee_day_amount_matrix,
                capital_cost_matrix,
                day_amount_matrix,
                capital_cost_duration_matrix,  # New On Finetune
                result_matrix,
            ) = cost_generator.generate_data_from_input(
                train_process_df,
                train_material_usage,
                train_employee_usage,
                train_capital_cost,
            )
            # Create Training
            (
                round_err,
                round_time,
                round_err_percent,
                val_err,
                val_err_percent,
                brakpoint_err,
                breakpoint_err_percent,
                breakpoint_val_err,
                breakpoint_val_err_percent,
            ) = create_learning(
                epoch=epoch,
                learning_rate=model_learning_rate,
                folder_name=f"{folder_name}/round{round + 1}",
                # Material
                material_amount_matrix=material_amount_matrix,
                material_cost_matrix=material_cost_matrix,
                # Employee
                employee_cost_matrix=employee_cost_matrix,
                employee_duration_matrix=employee_duration_matrix,
                employee_day_amount_matrix=employee_day_amount_matrix,
                # Capital Cost
                capital_cost_matrix=capital_cost_matrix,
                day_amount_matrix=day_amount_matrix,
                capital_duration_matrix=capital_cost_duration_matrix,
                # Result Matrix
                result_matrix=result_matrix,
                # Validation Payload
                validation_payload=validation_payload,
                inside_learning_rate=[learn_r, learn_r, learn_r],
                # because it start with -1
                breakpoint=breakpoint,
                early_stopping=early_stopping,
                patience_round=patience_round,
                use_model_weight=use_model_weight,
            )
            print(
                f"Learning Rate {learn_r} Round {round} / {round_number} Finish with Error {round_err} ({round_err_percent} %), time comsume {round_time} "
            )
            round_err_list.append(round_err)
            round_time_list.append(round_time)
            round_percent_list.append(round_err_percent)
            round_validate_err_list.append(val_err)
            round_validate_err_percent_list.append(val_err_percent)
            round_brakpoint_err_list.append(brakpoint_err)
            round_brakpoint_err_percent_list.append(breakpoint_err_percent)
            round_brakpoint_val_err_list.append(breakpoint_val_err)
            round_brakpoint_val_err_percent_list.append(
                breakpoint_val_err_percent)
            round_err_dict[f"round_{round}_err"] = round_err
            round_err_dict[f"round_{round}_duration"] = round_time
            round_err_dict[f"round_{round}_err_percent"] = round_err_percent
            round_err_dict[f"round_{round}_val_err"] = val_err
            round_err_dict[f"round_{round}_val_err_percent"] = val_err_percent
            round_err_dict[f"round_{round}_brakpoint_err"] = brakpoint_err
            round_err_dict[f"round_{round}_brakpoint_err_percent"] = (
                breakpoint_err_percent
            )
            round_err_dict[f"round_{round}_brakpoint_val_err"] = breakpoint_val_err
            round_err_dict[f"round_{round}_brakpoint_val_err_percent"] = (
                breakpoint_val_err_percent
            )

        average_error = np.average(round_err_list)
        average_time = np.average(round_time_list)
        average_error_percent = np.average(round_percent_list)
        average_val_err = np.average(round_validate_err_list)
        average_val_err_percent = np.average(round_validate_err_percent_list)
        breakpoint_average_error = np.average(round_brakpoint_err_list)
        breakpoint_average_error_percent = np.average(
            round_brakpoint_err_percent_list)
        breakpoint_average_val_error = np.average(round_brakpoint_val_err_list)
        breakpoint_average_val_error_percent = np.average(
            round_brakpoint_val_err_percent_list
        )
        print(
            f"Success for learning rate {learn_r} : Average error {average_error} ({average_error_percent}%) Average Time {average_time}"
        )
        round_err_dict["average_error"] = average_error
        round_err_dict["average_err_percent"] = average_error_percent
        round_err_dict["average_time"] = average_time
        round_err_dict["average_validate_error"] = average_val_err
        round_err_dict["average_validate_error_percent"] = average_val_err_percent
        round_err_dict["average_brakpoint_error"] = breakpoint_average_error
        round_err_dict["average_brakpoint_error_percent"] = (
            breakpoint_average_error_percent
        )
        round_err_dict["average_brakpoint_val_error"] = breakpoint_average_val_error
        round_err_dict["average_brakpoint_val_error_percent"] = (
            breakpoint_average_val_error_percent
        )
        overall_result.append(round_err_dict)

    overall_result_df = pd.DataFrame(overall_result)
    overall_result_df.to_csv(f"{folder_name}/overall_result.csv")

    rd.creating_error_csv(
        primary_directory_name=folder_name,
        learning_rate=learning_rate_list,
        iteration_number=epoch,
        round_number=round_number,
        breakpoint_number=breakpoint,
    )

    return overall_result_df
