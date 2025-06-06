import pandas as pd
from scipy.stats import variation, iqr


def display_input_variation(
    process_df,
    material_usage_df,
    employee_usage,
    capital_cost_df,
):
    process_des = process_df.describe()

    cost_dict = {
        "data": "Total Cost",
        "records": process_des["cost"]["count"],
        "types": len(process_df["process_id"].unique()),
        "min": process_des["cost"]["min"],
        "mean": round(process_des["cost"]["mean"], 2),
        "max": process_des["cost"]["max"],
        "sd": process_des["cost"]["std"],
        "variation": variation(process_df["cost"]),
        "iqr": iqr(process_df["cost"]),
    }

    material_des = material_usage_df.describe()

    material_amount = {
        "data": "Material Amount",
        "records": material_des["amount"]["count"],
        "types": len(material_usage_df["name"].unique()),
        "min": material_des["amount"]["min"],
        "mean": round(material_des["amount"]["mean"], 2),
        "max": material_des["amount"]["max"],
        "sd": material_des["amount"]["std"],
        "variation": variation(material_usage_df["amount"]),
        "iqr": iqr(material_usage_df["amount"]),
    }

    material_unit_cost = {
        "data": "Material Unit Cost",
        "records": material_des["unit_cost"]["count"],
        "types": len(material_usage_df["name"].unique()),
        "min": material_des["unit_cost"]["min"],
        "mean": round(material_des["unit_cost"]["mean"], 2),
        "max": material_des["unit_cost"]["max"],
        "sd": material_des["unit_cost"]["std"],
        "variation": variation(material_usage_df["unit_cost"]),
        "iqr": iqr(material_usage_df["unit_cost"]),
    }

    employee_usage["duration"] = (
        employee_usage["duration"] * employee_usage["amount"]
    )
    ec_des = employee_usage.describe()

    ec_unit_cost = {
        "data": "Labor Unit Cost",
        "records": ec_des["cost"]["count"],
        "types": len(employee_usage["employee_name"].unique()),
        "min": ec_des["cost"]["min"],
        "mean": round(ec_des["cost"]["mean"], 2),
        "max": ec_des["cost"]["max"],
        "sd": ec_des["cost"]["std"],
        "variation": variation(employee_usage["cost"]),
        "iqr": iqr(employee_usage["cost"]),
    }

    ec_duration = {
        "data": "Labor Duration",
        "records": ec_des["duration"]["count"],
        "types": len(employee_usage["employee_name"].unique()),
        "min": ec_des["duration"]["min"],
        "mean": round(ec_des["duration"]["mean"], 2),
        "max": ec_des["duration"]["max"],
        "sd": ec_des["duration"]["std"],
        "variation": variation(employee_usage["duration"]),
        "iqr": iqr(employee_usage["duration"]),
    }

    ec_day_amount = {
        "data": "Labor Day Amount",
        "records": ec_des["day_amount"]["count"],
        "types": len(employee_usage["employee_name"].unique()),
        "min": ec_des["day_amount"]["min"],
        "mean": round(ec_des["day_amount"]["mean"], 2),
        "max": ec_des["day_amount"]["max"],
        "sd": ec_des["day_amount"]["std"],
        "variation": variation(employee_usage["day_amount"]),
        "iqr": iqr(employee_usage["day_amount"]),
    }

    capital_des = capital_cost_df.describe()

    cc_unit_cost = {
        "data": "Capital Cost",
        "records": capital_des["cost"]["count"],
        "types": len(capital_cost_df["name"].unique()),
        "min": capital_des["cost"]["min"],
        "mean": round(capital_des["cost"]["mean"], 2),
        "max": capital_des["cost"]["max"],
        "sd": capital_des["cost"]["std"],
        "variation": variation(capital_cost_df["cost"]),
        "iqr": iqr(capital_cost_df["cost"])
    }

    cc_dayamount = {
        "data": "Capital Cost Day Amount",
        "records": capital_des["day_amount"]["count"],
        "types": len(capital_cost_df["name"].unique()),
        "min": capital_des["day_amount"]["min"],
        "mean": round(capital_des["day_amount"]["mean"], 2),
        "max": capital_des["day_amount"]["max"],
        "sd": capital_des["day_amount"]["std"],
        "variation": variation(capital_cost_df["day_amount"]),
        "iqr": iqr(capital_cost_df["day_amount"])
    }

    cc_duration = {
        "data": "Capital Cost Duration",
        "records": capital_des["duration"]["count"],
        "types": len(capital_cost_df["name"].unique()),
        "min": capital_des["duration"]["min"],
        "mean": round(capital_des["duration"]["mean"], 2),
        "max": capital_des["duration"]["max"],
        "sd": capital_des["duration"]["std"],
        "variation": variation(capital_cost_df["duration"]),
        "iqr": iqr(capital_cost_df["duration"])
    }

    data_variation = pd.DataFrame(
        [
            cost_dict,
            material_unit_cost,
            material_amount,
            ec_unit_cost,
            ec_duration,
            ec_day_amount,
            cc_unit_cost,
            cc_dayamount,
            cc_duration,
        ]
    )

    # try:
    #     display(data_variation)
    # except:
    #     print("Not Run in Jupyter Notebook")
    #     print(data_variation)
    return data_variation


def display_input_variation_by_directory(folder_name):
    process_df = pd.read_csv(f"{folder_name}/generated_process_data.csv")
    material_usage_df = pd.read_csv(
        f"{folder_name}/generated_material_usage.csv")
    employee_usage_df = pd.read_csv(
        f"{folder_name}/generated_employee_usage.csv")
    capital_cost_df = pd.read_csv(f"{folder_name}/generated_captial_cost.csv")

    result_variation = display_input_variation(
        process_df,
        material_usage_df,
        employee_usage_df,
        capital_cost_df,
    )

    return result_variation
