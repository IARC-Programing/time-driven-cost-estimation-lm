import pandas as pd
import numpy as np
import importlib
import sys

# fmt:off
sys.path.append('../matrix_generator')

import cost_matrix_generator as cmg
importlib.reload(cmg)
# fmt:on

# Import Data
process_df = pd.read_csv('example/generated_process_data.csv')
employee_usage = pd.read_csv("example/generated_employee_usage.csv")
material_usage = pd.read_csv("example/generated_material_usage.csv")
capital_cost_usage = pd.read_csv("example/generated_captial_cost.csv")

# Data Extract and Shaping (Pre Processing)


def generate_data():
    # Material
    material_cost_matrix, material_amount_matrix = cmg.generate_material_usage_cost_matrix(
        process_df, material_usage)
    material_cost_matrix = material_cost_matrix.values
    material_amount_matrix = material_amount_matrix.values

    row, col = material_cost_matrix.shape
    material_cost_matrix = material_cost_matrix.reshape(row, 1, col)
    material_amount_matrix = material_amount_matrix.reshape(row, 1, col)

    # Employee
    monthy_employee_cost_matrix, daily_employee_cost_matrix = cmg.generate_employee_usage_cost_matrix(
        process_df, employee_usage)
    monthy_employee_cost_matrix = monthy_employee_cost_matrix.values
    daily_employee_cost_matrix = daily_employee_cost_matrix.values

    row, col = monthy_employee_cost_matrix.shape
    monthy_employee_cost_matrix = monthy_employee_cost_matrix.reshape(
        row, 1, col)
    row, col = daily_employee_cost_matrix.shape
    daily_employee_cost_matrix = daily_employee_cost_matrix.reshape(
        row, 1, col)
    print(
        f'Monthy Employee Cost matrix shape {monthy_employee_cost_matrix.shape} & Daily Employee Cost Matrix Shape {daily_employee_cost_matrix.shape}')

    # Capital Cost
    unit_cost_matrix, machine_hour_matrix, life_time_matrix = cmg.generate_capital_cost_matrix(
        process_df, capital_cost_df=capital_cost_usage)

    capital_cost_matrix = unit_cost_matrix.values

    machine_hour_matrix = machine_hour_matrix.values

    life_time_matrix = life_time_matrix.values

    row, col = capital_cost_matrix.shape

    capital_cost_matrix = capital_cost_matrix.reshape(row, 1, col)

    row, col = machine_hour_matrix.shape

    machine_hour_matrix = machine_hour_matrix.reshape(row, 1, col)

    row, col = life_time_matrix.shape

    life_time_matrix = life_time_matrix.reshape(row, 1, col)

    # Time Usage
    duration_matrix = cmg.generate_duration_matrix(
        process_df=process_df, use_3d=True)

    result_matrix = cmg.generate_price_matrix(
        process_df=process_df, use_3d=True
    )

    return material_cost_matrix, material_amount_matrix, monthy_employee_cost_matrix, daily_employee_cost_matrix, capital_cost_matrix, machine_hour_matrix, life_time_matrix, duration_matrix, result_matrix
