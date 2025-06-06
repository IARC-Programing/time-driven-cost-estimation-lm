import pandas as pd
from scipy.stats import variation, iqr


def get_data(folder_name):
    process_df = pd.read_csv(f"{folder_name}/generated_process_data.csv")
    material_usage_df = pd.read_csv(
        f"{folder_name}/generated_material_usage.csv")
    employee_usage_df = pd.read_csv(
        f"{folder_name}/generated_employee_usage.csv")
    capital_cost_df = pd.read_csv(f"{folder_name}/generated_captial_cost.csv")
    return process_df, material_usage_df, employee_usage_df, capital_cost_df
