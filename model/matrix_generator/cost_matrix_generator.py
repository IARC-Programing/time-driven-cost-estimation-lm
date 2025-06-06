import pandas as pd
import numpy as np


def generate_material_usage_cost_matrix(process_df, material_usage_df):
    overall_cost_matrix = pd.DataFrame()
    overall_amount_matrix = pd.DataFrame()
    process_df = process_df.reset_index()

    # Find Unique for fixed the column of dataframe and Matrix
    unique_material_id = material_usage_df["_id"].unique()

    for index in process_df.index:
        selected_process = process_df.iloc[index]

        # Find Unique Material
        # Find All Material Usage
        selected_process_material = material_usage_df[
            material_usage_df["process_id"] == selected_process["process_id"]
        ]
        cost_dict = {}
        amount_dict = {}

        # Iteration Over Material
        for idx in range(len(selected_process_material)):
            selected_data = selected_process_material.iloc[idx]
            material_id = selected_data["_id"]
            cost = selected_data["unit_cost"]
            amount = selected_data["amount"]
            cost_dict[material_id] = cost
            amount_dict[material_id] = amount

        # Iteration Over Rest of Material
        for material_id in unique_material_id:
            if material_id not in cost_dict:
                cost_dict[material_id] = 0
                amount_dict[material_id] = 0

        cost_df = pd.DataFrame(
            cost_dict, index=[selected_process["process_id"]])
        amount_df = pd.DataFrame(
            amount_dict, index=[selected_process["process_id"]])

        overall_cost_matrix = pd.concat([overall_cost_matrix, cost_df], axis=0)
        overall_amount_matrix = pd.concat(
            [overall_amount_matrix, amount_df], axis=0)

    # Reorder Column to be same normal and validation
    column_list = unique_material_id.tolist()
    ordered_columns = sorted(column_list)
    overall_cost_matrix = overall_cost_matrix[ordered_columns]
    overall_amount_matrix = overall_amount_matrix[ordered_columns]

    overall_cost_matrix.fillna(0, inplace=True)
    overall_amount_matrix.fillna(0, inplace=True)

    return (overall_cost_matrix, overall_amount_matrix)


def generate_employee_usage_cost_matrix(process_df, employee_usage_df):
    overall_cost_matrix = pd.DataFrame()
    overall_duration_matrix = pd.DataFrame()
    overall_day_amount_matrix = pd.DataFrame()

    employee_usage_df["duration"] = (
        employee_usage_df["duration"] * employee_usage_df["amount"]
    )

    unique_emp_id = employee_usage_df["employee_id"].unique()

    process_df = process_df.reset_index()

    # Iteration Over each process
    for index in process_df.index:
        selected_process = process_df.iloc[index]
        selected_process_employee = employee_usage_df[
            employee_usage_df["process_id"] == selected_process["process_id"]
        ]

        ec_cost_dict = {}
        ec_duration_dict = {}
        ec_day_amount_dict = {}

        unique_process_employee = selected_process_employee["employee_id"].unique(
        )
        unique_pe_df = pd.DataFrame(
            unique_process_employee, columns=["employee_id"])

        # Adust Employee
        # Converse many employee record to one record for each employee in one process
        # Process will can contain many of employee but 1 employee only 1 record
        for idx in range(len(unique_pe_df)):
            selected_data = unique_pe_df.iloc[idx]
            all_record = selected_process_employee[
                selected_process_employee["employee_id"] == selected_data["employee_id"]
            ]
            first_record = all_record.iloc[0]
            # Cost also the same pick from the first record
            cost = first_record["cost"]
            day_amount = first_record["day_amount"]
            # Duration is the sum of all duration of employee in the process
            duration = all_record["duration"].sum()
            # Update the record of unique_pe_df add cost and duration
            unique_pe_df.loc[idx, "cost"] = cost
            unique_pe_df.loc[idx, "duration"] = duration
            unique_pe_df.loc[idx, "day_amount"] = day_amount
            unique_pe_df.loc[idx, "process_id"] = first_record["process_id"]

        for idx in range(len(unique_pe_df)):
            selected_data = unique_pe_df.iloc[idx]
            cost = selected_data["cost"]
            employee_id = selected_data["employee_id"]
            duration = selected_data["duration"]
            day_amount = selected_data["day_amount"]

            ec_cost_dict[employee_id] = cost
            ec_duration_dict[employee_id] = duration
            ec_day_amount_dict[employee_id] = day_amount

        # Iteration Over Rest of Employee

        for employee_id in unique_emp_id:
            if employee_id not in ec_cost_dict:
                ec_cost_dict[employee_id] = 0
                ec_duration_dict[employee_id] = 0
                ec_day_amount_dict[employee_id] = 1

        employee_cost_df = pd.DataFrame(
            ec_cost_dict, index=[selected_process["process_id"]]
        )

        employee_duration_df = pd.DataFrame(
            ec_duration_dict, index=[selected_process["process_id"]]
        )

        employee_dayamount_df = pd.DataFrame(
            ec_day_amount_dict, index=[selected_process["process_id"]]
        )

        overall_cost_matrix = pd.concat(
            [overall_cost_matrix, employee_cost_df], axis=0
        )

        overall_duration_matrix = pd.concat(
            [overall_duration_matrix, employee_duration_df], axis=0
        )

        overall_day_amount_matrix = pd.concat(
            [overall_day_amount_matrix, employee_dayamount_df], axis=0
        )

    # Reorder Column to be same normal and validation

    # Reorder for Emploee
    column_list = unique_emp_id.tolist()
    ordered_columns = sorted(column_list)
    overall_cost_matrix = overall_cost_matrix[ordered_columns]
    overall_duration_matrix = overall_duration_matrix[ordered_columns]
    overall_day_amount_matrix = overall_day_amount_matrix[ordered_columns]

    overall_cost_matrix.fillna(0, inplace=True)
    overall_duration_matrix.fillna(0, inplace=True)
    overall_day_amount_matrix.fillna(1, inplace=True)

    return (
        overall_cost_matrix,
        overall_duration_matrix,
        overall_day_amount_matrix,
    )


def generate_capital_cost_matrix(process_df, capital_cost_df):
    cost_df = pd.DataFrame()
    day_amount_df = pd.DataFrame()
    duration_df = pd.DataFrame()
    process_df = process_df.reset_index()

    # Find Unique id to fixed the column of dataframe and Matrix
    unique_capital_cost_id = capital_cost_df["_id"].unique()

    # Iteration Over each process
    for index in process_df.index:
        selected_process = process_df.iloc[index]
        selected_process_capitalcost = capital_cost_df[
            capital_cost_df["process_id"] == selected_process["process_id"]
        ]

        cost_dict = {}
        day_amount_dict = {}
        duration_dict = {}
        uniq_process_cc = selected_process_capitalcost["_id"].unique()

        uniq_pcc_df = pd.DataFrame(
            uniq_process_cc, columns=["capital_cost_id"])

        # Adust Capital Cost
        # Converse many capital Cost record to one record for each capital Cost in one process
        # Process will can contain many of capital cost but 1 capital cost only 1 record
        for idx in range(len(uniq_pcc_df)):
            selected_data = uniq_pcc_df.iloc[idx]
            all_record = selected_process_capitalcost[
                selected_process_capitalcost["_id"] == selected_data["capital_cost_id"]
            ]
            first_record = all_record.iloc[0]
            # Cost also the same pick from the first record
            cost = first_record["cost"]
            # Duration is the sum of all duration of employee in the process
            duration = all_record["duration"].sum()
            # Update the record of unique_pe_df add cost and duration
            uniq_pcc_df.loc[idx, "cost"] = cost
            uniq_pcc_df.loc[idx, "duration"] = duration
            uniq_pcc_df.loc[idx, "day_amount"] = first_record["day_amount"]
            uniq_pcc_df.loc[idx, "process_id"] = first_record["process_id"]

        # Iteration Only for 1 cc per record data
        for idx in range(len(uniq_pcc_df)):
            selected_data = uniq_pcc_df.iloc[idx]
            cost = selected_data["cost"]
            duration = selected_data["duration"]
            captial_cost_id = selected_data["capital_cost_id"]
            cost_dict[captial_cost_id] = cost
            day_amount_dict[captial_cost_id] = selected_data["day_amount"]
            duration_dict[captial_cost_id] = duration

        # Do for the rest of capital cost object
        for cc_id in unique_capital_cost_id:
            if cc_id not in cost_dict:
                cost_dict[cc_id] = 0
                day_amount_dict[cc_id] = 1
                duration_dict[cc_id] = 0

        # Dataframe Generation
        sub_cost_df = pd.DataFrame(
            cost_dict, index=[selected_process["process_id"]])
        sub_duration_df = pd.DataFrame(
            duration_dict, index=[selected_process["process_id"]]
        )
        sub_dayamount_df = pd.DataFrame(
            day_amount_dict, index=[selected_process["process_id"]]
        )

        # Generate Matrix
        cost_df = pd.concat([cost_df, sub_cost_df], axis=0)
        day_amount_df = pd.concat([day_amount_df, sub_dayamount_df], axis=0)
        duration_df = pd.concat([duration_df, sub_duration_df], axis=0)

    # Reorder the Column to be same on Training and validation

    column_list = unique_capital_cost_id.tolist()
    sorted_column = sorted(column_list)

    day_amount_df = day_amount_df[sorted_column]
    cost_df = cost_df[sorted_column]
    duration_df = duration_df[sorted_column]

    day_amount_df.fillna(1, inplace=True)
    cost_df.fillna(0, inplace=True)
    duration_df.fillna(1, inplace=True)

    return (cost_df, day_amount_df, duration_df)


def generate_price_matrix(process_df, use_3d=False, use_unit_cost=True):
    price_arr = []
    process_df = process_df.reset_index()
    if use_unit_cost:
        for index in process_df.index:
            selected_data = process_df.iloc[index]
            cost = selected_data["cost"]
            if use_3d:
                price_arr.append([[cost]])
            else:
                price_arr.append([cost])
    else:
        for index in process_df.index:
            selected_data = process_df.iloc[index]
            if use_3d:
                price_arr.append([[selected_data["cost"]]])
            else:
                price_arr.append([selected_data["cost"]])
    price_arr = np.array(price_arr)
    return price_arr


def generate_duration_matrix(process_df, use_3d=False):
    duration_array = []
    process_df = process_df.reset_index()
    for index in process_df.index:
        selected_data = process_df.iloc[index]
        duration = selected_data["duration"]
        if use_3d:
            duration_array.append([[duration]])
        else:
            duration_array.append([duration])

    duration_array = np.array(duration_array)
    return duration_array
