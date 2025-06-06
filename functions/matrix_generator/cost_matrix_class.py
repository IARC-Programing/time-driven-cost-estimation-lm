import pandas as pd
import numpy as np
import importlib

# fmt:off
import cost_matrix_generator as cmg
importlib.reload(cmg)
# fmt:on


class CostMatrixGenerator:
    def __init__(self):
        self.process_df = pd.DataFrame()
        self.employee_usage = pd.DataFrame()
        self.material_usage = pd.DataFrame()
        self.capital_cost_usage = pd.DataFrame()
        self.data_directory = "example"

    def change_data_directory(self, new_directory):
        self.data_directory = new_directory

    def load_data(self):
        self.process_df = pd.read_csv(
            f"{self.data_directory}/generated_process_data.csv"
        )
        self.employee_usage = pd.read_csv(
            f"{self.data_directory}/generated_employee_usage.csv"
        )
        self.material_usage = pd.read_csv(
            f"{self.data_directory}/generated_material_usage.csv"
        )
        self.capital_cost_usage = pd.read_csv(
            f"{self.data_directory}/generated_captial_cost.csv"
        )

    def remove_outlier_iqr(self, iqr_index=1.5):
        process_df = self.process_df
        q1 = np.percentile(process_df["cost"], 25)
        q3 = np.percentile(process_df["cost"], 75)
        iqr = q3 - q1
        lower_bound = q1 - (iqr_index * iqr)
        upper_bound = q3 + (iqr_index * iqr)
        process_df = process_df[
            (process_df["cost"] > lower_bound) & (
                process_df["cost"] < upper_bound)
        ]
        removed_data = self.process_df[
            ~self.process_df["process_id"].isin(process_df["process_id"])
        ]
        self.process_df = process_df
        print("Outliers removed")
        print(f"Amount of process to remove {len(removed_data)}")
        return process_df

    def generate_data(self):
        # Material
        material_cost_matrix, material_amount_matrix = (
            cmg.generate_material_usage_cost_matrix(
                self.process_df, self.material_usage
            )
        )
        material_cost_matrix = material_cost_matrix.values
        material_amount_matrix = material_amount_matrix.values

        row, col = material_cost_matrix.shape
        material_cost_matrix = material_cost_matrix.reshape(row, 1, col)
        material_amount_matrix = material_amount_matrix.reshape(row, 1, col)

        # Employee
        (
            employee_cost_matrix,
            employee_duration_matrix,
            employee_day_amount_matrix,
        ) = cmg.generate_employee_usage_cost_matrix(
            self.process_df, self.employee_usage
        )
        employee_cost_matrix = employee_cost_matrix.values
        employee_duration_matrix = employee_duration_matrix.values
        employee_day_amount_matrix = employee_day_amount_matrix.values

        # Reshape Matrix
        #  Employee Cost
        row, col = employee_cost_matrix.shape
        employee_cost_matrix = employee_cost_matrix.reshape(
            row, 1, col)

        #  Employee Duration
        row, col = employee_duration_matrix.shape
        employee_duration_matrix = employee_duration_matrix.reshape(
            row, 1, col
        )

        #  Employee Day Amount
        row, col = employee_day_amount_matrix.shape
        employee_day_amount_matrix = employee_day_amount_matrix.reshape(
            row, 1, col
        )

        print(
            f" Employee Cost matrix shape {employee_cost_matrix.shape}"
        )

        # Capital Cost
        (
            capital_cost_matrix,
            day_amount_matrix,
            capital_cost_duration_matrix,
        ) = cmg.generate_capital_cost_matrix(
            self.process_df, capital_cost_df=self.capital_cost_usage
        )

        # Get Values
        capital_cost_matrix = capital_cost_matrix.values
        day_amount_matrix = day_amount_matrix.values
        capital_cost_duration_matrix = capital_cost_duration_matrix.values

        # Reshape Matrix
        row, col = capital_cost_matrix.shape
        capital_cost_matrix = capital_cost_matrix.reshape(row, 1, col)

        row, col = day_amount_matrix.shape
        day_amount_matrix = day_amount_matrix.reshape(row, 1, col)

        row, col = capital_cost_duration_matrix.shape
        capital_cost_duration_matrix = capital_cost_duration_matrix.reshape(
            row, 1, col)

        result_matrix = cmg.generate_price_matrix(
            process_df=self.process_df, use_3d=True
        )

        return (
            material_cost_matrix,
            material_amount_matrix,
            employee_cost_matrix,
            employee_duration_matrix,
            employee_day_amount_matrix,
            capital_cost_matrix,
            day_amount_matrix,
            capital_cost_duration_matrix,  # New On Finetuning
            result_matrix,
        )

    def train_test_split(self, train_rate):
        process_df_size = len(self.process_df)
        trained_size = train_rate * process_df_size
        train_process_df = self.process_df.sample(int(trained_size))
        validate_process_df = self.process_df.drop(train_process_df.index)
        # Ensure at least one record per original_material_name in train_process_df
        unique_materials = self.process_df['original_material_name'].unique()
        for material in unique_materials:
            if material not in train_process_df['original_material_name'].values:
                sample_record = self.process_df[self.process_df['original_material_name'] == material].sample(
                    1)
                train_process_df = pd.concat([train_process_df, sample_record])
                validate_process_df = validate_process_df.drop(
                    sample_record.index)

        # Result Matrix
        result_matrix = cmg.generate_price_matrix(
            process_df=train_process_df, use_3d=True
        )

        validate_result_matrix = cmg.generate_price_matrix(
            process_df=validate_process_df, use_3d=True
        )

        # Material
        material_cost_matrix, material_amount_matrix = (
            cmg.generate_material_usage_cost_matrix(
                train_process_df, self.material_usage
            )
        )
        material_cost_matrix = material_cost_matrix.values
        material_amount_matrix = material_amount_matrix.values

        # Material Validate
        validate_material_cost_matrix, validate_material_amount_matrix = (
            cmg.generate_material_usage_cost_matrix(
                validate_process_df, self.material_usage
            )
        )

        validate_material_cost_matrix = validate_material_cost_matrix.values
        validate_material_amount_matrix = validate_material_amount_matrix.values

        row, col = material_cost_matrix.shape
        material_cost_matrix = material_cost_matrix.reshape(row, 1, col)
        material_amount_matrix = material_amount_matrix.reshape(row, 1, col)

        # Employee
        (
            employee_cost_matrix,
            employee_duration_matrix,
            employee_day_amount_matrix,
        ) = cmg.generate_employee_usage_cost_matrix(
            train_process_df, self.employee_usage
        )
        # Employee Extract Value
        employee_cost_matrix = employee_cost_matrix.values
        employee_duration_matrix = employee_duration_matrix.values
        employee_day_amount_matrix = employee_day_amount_matrix.values

        # Employee Reshape
        row, col = employee_cost_matrix.shape
        employee_cost_matrix = employee_cost_matrix.reshape(
            row, 1, col)
        row, col = employee_duration_matrix.shape
        employee_duration_matrix = employee_duration_matrix.reshape(
            row, 1, col)
        row, col = employee_day_amount_matrix.shape
        employee_day_amount_matrix = employee_day_amount_matrix.reshape(
            row, 1, col)

        print(
            f" Employee Cost matrix shape {employee_duration_matrix.shape} "
        )

        # Employee validate
        (
            validate_emp_cost_matrix,
            validate_emp_duration_matrix,
            validate_emp_day_amount_matrix,
        ) = cmg.generate_employee_usage_cost_matrix(
            validate_process_df, self.employee_usage
        )
        # Employee Validate Extract Value
        validate_emp_cost_matrix = validate_emp_cost_matrix.values
        validate_emp_duration_matrix = validate_emp_duration_matrix.values
        validate_emp_day_amount_matrix = validate_emp_day_amount_matrix.values

        # Employee Validate Reshape
        row, col = validate_emp_cost_matrix.shape
        validate_emp_cost_matrix = validate_emp_cost_matrix.reshape(
            row, 1, col
        )
        row, col = validate_emp_duration_matrix.shape
        validate_emp_duration_matrix = validate_emp_duration_matrix.reshape(
            row, 1, col
        )
        row, col = validate_emp_day_amount_matrix.shape
        validate_emp_day_amount_matrix = validate_emp_day_amount_matrix.reshape(
            row, 1, col
        )

        # Capital Cost
        capital_cost_matrix, day_amount_matrix, capital_duration_matrix = (
            cmg.generate_capital_cost_matrix(
                train_process_df, capital_cost_df=self.capital_cost_usage
            )
        )

        # Capital Cost Extract Value
        capital_cost_matrix = capital_cost_matrix.values
        day_amount_matrix = day_amount_matrix.values
        capital_duration_matrix = capital_duration_matrix.values

        # Capital Cost Reshape for 3D
        row, col = capital_cost_matrix.shape
        capital_cost_matrix = capital_cost_matrix.reshape(row, 1, col)
        row, col = day_amount_matrix.shape
        day_amount_matrix = day_amount_matrix.reshape(row, 1, col)
        row, col = capital_duration_matrix.shape
        capital_duration_matrix = capital_duration_matrix.reshape(row, 1, col)

        # Capital Cost validate
        (
            valdiate_capital_cost_matrix,
            validate_dayamount_matrix,
            validate_capital_duration_matrix,
        ) = cmg.generate_capital_cost_matrix(
            validate_process_df, capital_cost_df=self.capital_cost_usage
        )

        # Capital Cost Validate Value Extraction
        valdiate_capital_cost_matrix = valdiate_capital_cost_matrix.values
        validate_dayamount_matrix = validate_dayamount_matrix.values
        validate_capital_duration_matrix = validate_capital_duration_matrix.values

        # Capital Cost Validate Reshape
        row, col = valdiate_capital_cost_matrix.shape
        valdiate_capital_cost_matrix = valdiate_capital_cost_matrix.reshape(
            row, 1, col)

        row, col = validate_dayamount_matrix.shape
        validate_dayamount_matrix = validate_dayamount_matrix.reshape(
            row, 1, col)

        row, col = validate_capital_duration_matrix.shape
        validate_capital_duration_matrix = validate_capital_duration_matrix.reshape(
            row, 1, col
        )

        validate_payload = {
            "validate_process_df": validate_process_df,
            "validate_result_matrix": validate_result_matrix,
            "validate_material_cost_matrix": validate_material_cost_matrix,
            "validate_material_amount_matrix": validate_material_amount_matrix,
            "validate_capital_cost_matrix": valdiate_capital_cost_matrix,
            "validate_day_amount_matrix": validate_dayamount_matrix,
            "validate_capital_duration_matrix": validate_capital_duration_matrix,
            "validate_employee_cost_matrix": validate_emp_cost_matrix,
            "validate_employee_duration_matrix": validate_emp_duration_matrix,
            "validate_employee_day_amount_matrix": validate_emp_day_amount_matrix,
        }

        return (
            material_cost_matrix,
            material_amount_matrix,
            employee_cost_matrix,
            employee_duration_matrix,
            employee_day_amount_matrix,
            capital_cost_matrix,
            day_amount_matrix,
            capital_duration_matrix,
            result_matrix,
            validate_payload,
        )

    def train_test_split_without_matrix(self, train_rate):
        process_df_size = len(self.process_df)
        trained_size = train_rate * process_df_size
        train_process_df = self.process_df.sample(int(trained_size))
        validate_process_df = self.process_df.drop(train_process_df.index)

        # Adjust to match new Process dataset
        train_capital_cost = self.capital_cost_usage.copy()
        train_capital_cost = train_capital_cost[
            train_capital_cost["process_id"].isin(
                train_process_df["process_id"])
        ]

        train_employee_usage = self.employee_usage.copy()
        train_employee_usage = train_employee_usage[
            train_employee_usage["process_id"].isin(
                train_process_df["process_id"])
        ]

        train_material_usage = self.material_usage.copy()
        train_material_usage = train_material_usage[
            train_material_usage["process_id"].isin(
                train_process_df["process_id"])
        ]

        # Adjust to match new Process dataset for validation set
        validate_capital_cost = self.capital_cost_usage.copy()
        validate_capital_cost = validate_capital_cost[
            validate_capital_cost["process_id"].isin(
                validate_process_df["process_id"])
        ]

        validate_employee_usage = self.employee_usage.copy()
        validate_employee_usage = validate_employee_usage[
            validate_employee_usage["process_id"].isin(
                validate_process_df["process_id"]
            )
        ]

        validate_material_usage = self.material_usage.copy()
        validate_material_usage = validate_material_usage[
            validate_material_usage["process_id"].isin(
                validate_process_df["process_id"]
            )
        ]

        return (
            train_process_df,
            train_employee_usage,
            train_material_usage,
            train_capital_cost,
            validate_process_df,
            validate_employee_usage,
            validate_material_usage,
            validate_capital_cost,
        )

    def generate_data_from_input(
        self, process_df, material_usage, employee_usage, capital_cost_usage
    ):
        # Material
        material_cost_matrix, material_amount_matrix = (
            cmg.generate_material_usage_cost_matrix(process_df, material_usage)
        )
        material_cost_matrix = material_cost_matrix.values
        material_amount_matrix = material_amount_matrix.values

        row, col = material_cost_matrix.shape
        material_cost_matrix = material_cost_matrix.reshape(row, 1, col)
        material_amount_matrix = material_amount_matrix.reshape(row, 1, col)

        # Employee
        (employee_cost_matrix,
         employee_duration_matrix,
         employee_day_amount_matrix,
         ) = cmg.generate_employee_usage_cost_matrix(process_df, employee_usage)
        employee_cost_matrix = employee_cost_matrix.values
        employee_duration_matrix = employee_duration_matrix.values
        employee_day_amount_matrix = employee_day_amount_matrix.values

        # Reshape Matrix
        #  Employee Cost
        row, col = employee_cost_matrix.shape
        employee_cost_matrix = employee_cost_matrix.reshape(
            row, 1, col)
        #  Employee Duration
        row, col = employee_duration_matrix.shape
        employee_duration_matrix = employee_duration_matrix.reshape(
            row, 1, col)
        #  Employee Day Amount
        row, col = employee_day_amount_matrix.shape
        employee_day_amount_matrix = employee_day_amount_matrix.reshape(
            row, 1, col
        )

        print(
            f" Employee Cost matrix shape {employee_cost_matrix.shape} "
        )

        # Capital Cost
        (
            capital_cost_matrix,
            day_amount_matrix,
            capital_cost_duration_matrix,
        ) = cmg.generate_capital_cost_matrix(
            process_df, capital_cost_df=capital_cost_usage
        )

        # Get Values
        capital_cost_matrix = capital_cost_matrix.values
        day_amount_matrix = day_amount_matrix.values
        capital_cost_duration_matrix = capital_cost_duration_matrix.values

        # Reshape Matrix
        row, col = capital_cost_matrix.shape
        capital_cost_matrix = capital_cost_matrix.reshape(row, 1, col)

        row, col = day_amount_matrix.shape
        day_amount_matrix = day_amount_matrix.reshape(row, 1, col)

        row, col = capital_cost_duration_matrix.shape
        capital_cost_duration_matrix = capital_cost_duration_matrix.reshape(
            row, 1, col)

        result_matrix = cmg.generate_price_matrix(
            process_df=process_df, use_3d=True)

        return (
            material_cost_matrix,
            material_amount_matrix,
            employee_cost_matrix,
            employee_duration_matrix,
            employee_day_amount_matrix,
            capital_cost_matrix,
            day_amount_matrix,
            capital_cost_duration_matrix,  # New On Finetuning
            result_matrix,
        )

    def get_validation_payload(self, validate_process_df):
        validate_result_matrix = cmg.generate_price_matrix(
            process_df=validate_process_df, use_3d=True
        )

        # Material Validate
        validate_material_cost_matrix, validate_material_amount_matrix = (
            cmg.generate_material_usage_cost_matrix(
                validate_process_df, self.material_usage
            )
        )

        validate_material_cost_matrix = validate_material_cost_matrix.values
        validate_material_amount_matrix = validate_material_amount_matrix.values

        # Employee validate
        (
            validate_employee_cost_matrix,
            validate_employee_duration_matrix,
            validate_employee_day_amount_matrix,
        ) = cmg.generate_employee_usage_cost_matrix(
            validate_process_df, self.employee_usage
        )
        # Employee Validate Extract Value
        validate_employee_cost_matrix = validate_employee_cost_matrix.values
        validate_employee_duration_matrix = validate_employee_duration_matrix.values
        validate_employee_day_amount_matrix = validate_employee_day_amount_matrix.values

        # Employee Validate Reshape
        row, col = validate_employee_cost_matrix.shape
        validate_employee_cost_matrix = validate_employee_cost_matrix.reshape(
            row, 1, col
        )
        row, col = validate_employee_duration_matrix.shape
        validate_employee_duration_matrix = validate_employee_duration_matrix.reshape(
            row, 1, col
        )
        row, col = validate_employee_day_amount_matrix.shape
        validate_employee_day_amount_matrix = validate_employee_day_amount_matrix.reshape(
            row, 1, col
        )

        # Capital Cost validate
        (
            valdiate_capital_cost_matrix,
            validate_dayamount_matrix,
            validate_capital_duration_matrix,
        ) = cmg.generate_capital_cost_matrix(
            validate_process_df, capital_cost_df=self.capital_cost_usage
        )

        # Capital Cost Validate Value Extraction
        valdiate_capital_cost_matrix = valdiate_capital_cost_matrix.values
        validate_dayamount_matrix = validate_dayamount_matrix.values
        validate_capital_duration_matrix = validate_capital_duration_matrix.values

        # Capital Cost Validate Reshape
        row, col = valdiate_capital_cost_matrix.shape
        valdiate_capital_cost_matrix = valdiate_capital_cost_matrix.reshape(
            row, 1, col)

        row, col = validate_dayamount_matrix.shape
        validate_dayamount_matrix = validate_dayamount_matrix.reshape(
            row, 1, col)

        row, col = validate_capital_duration_matrix.shape
        validate_capital_duration_matrix = validate_capital_duration_matrix.reshape(
            row, 1, col
        )

        validate_payload = {
            "validate_process_df": validate_process_df,
            "validate_result_matrix": validate_result_matrix,
            "validate_material_cost_matrix": validate_material_cost_matrix,
            "validate_material_amount_matrix": validate_material_amount_matrix,
            "validate_capital_cost_matrix": valdiate_capital_cost_matrix,
            "validate_day_amount_matrix": validate_dayamount_matrix,
            "validate_capital_duration_matrix": validate_capital_duration_matrix,
            "validate_employee_cost_matrix": validate_employee_cost_matrix,
            "validate_employee_duration_matrix": validate_employee_duration_matrix,
            "validate_employee_day_amount_matrix": validate_employee_day_amount_matrix,
        }

        return validate_payload

    def get_data(self):
        return (
            self.process_df,
            self.employee_usage,
            self.material_usage,
            self.capital_cost_usage,
        )
