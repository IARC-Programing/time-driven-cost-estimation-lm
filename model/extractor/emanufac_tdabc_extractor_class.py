import pandas as pd
import requests
import numpy as np
import pickle


class EManufacTDABCExtractor:
    def __init__(self, api_url, api_key, profile_id, costed_place):
        self.api_url = api_url
        self.api_key = api_key
        self.process_df = pd.DataFrame()
        self.material_usage_df = pd.DataFrame()
        self.original_material_usage_df = pd.DataFrame()
        self.original_employee_usage_df = pd.DataFrame()
        self.original_process_df = pd.DataFrame()
        self.capital_cost_df = pd.DataFrame()
        self.employee_usage_df = pd.DataFrame()
        self.original_capital_cost_df = pd.DataFrame()
        self.profile_id = profile_id  # Profile For TDABC
        self.original_procedure_profile_id = ""  # Profile For Factory Current  Method
        self.profile_element_name_for_material = "ต้นทุนวัตถุดิบ"
        self.costed_place = costed_place
        self.running_no_start = ""
        self.running_no_end = ""

    def set_running_no_margin(self, start, end):
        self.running_no_start = start
        self.running_no_end = end
        print("Setting Successfully")

    def fetch_material_usage(self, start_date, end_date, limit, page=1):
        url = f"{self.api_url}/cost-estimation/on-type"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        querystring = {
            "startDate": start_date,
            "endDate": end_date,
            "size": limit,
            "page": page,
            "profile": self.profile_id,
            "elementType": "MATERIAL",
            "placeRestricNotify": "true",
            "correctPlaceOnly": "true",
            "extractOriginalLot": "true",
            "place": self.costed_place,
            "specifyProfileElement": self.profile_element_name_for_material,
            "runningNoStart": self.running_no_start,
            "runningNoEnd": self.running_no_end,
        }
        response = requests.get(url, headers=headers, params=querystring)
        try:
            material_data = response.json()["rows"]
        except Exception as e:
            print("Error in fetch material", e)
            material_data = []

        self.material_usage_df = pd.DataFrame(material_data)
        self.original_material_usage_df = pd.DataFrame(material_data)

        with open("material.pickle", "wb") as handle:
            pickle.dump(self.material_usage_df, handle)

        return (self.material_usage_df,)

    def fetch_employee_usage(self, start_date, end_date, limit, page=1):
        url = f"{self.api_url}/cost-estimation/on-type"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        querystring = {
            "startDate": start_date,
            "endDate": end_date,
            "size": limit,
            "page": page,
            "profile": self.profile_id,
            "elementType": "LABOR",
            "placeRestricNotify": "true",
            "correctPlaceOnly": "true",
            "place": self.costed_place,
            "merged": "true",
            "runningNoStart": self.running_no_start,
            "runningNoEnd": self.running_no_end,
        }
        response = requests.get(url, headers=headers, params=querystring)
        try:
            employee_data = response.json()["rows"]
        except Exception as e:
            print("Error in fetch employee", e)
            employee_data = []

        self.employee_usage_df = pd.DataFrame(employee_data)
        self.original_employee_usage_df = pd.DataFrame(employee_data)

        with open("employee.pickle", "wb") as handle:
            pickle.dump(self.employee_usage_df, handle)

        return (self.employee_usage_df,)

    def fetch_capital_cost_usage(self, start_date, end_date, limit, page=1):
        url = f"{self.api_url}/cost-estimation/on-type"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        querystring = {
            "startDate": start_date,
            "endDate": end_date,
            "size": limit,
            "page": page,
            "profile": self.profile_id,
            "elementType": "CAPITAL_COST",
            "placeRestricNotify": "true",
            "correctPlaceOnly": "true",
            "place": self.costed_place,
            "merged": "true",
            "splitCostDriver": "true",
            "runningNoStart": self.running_no_start,
            "runningNoEnd": self.running_no_end,
        }
        response = requests.get(url, headers=headers, params=querystring)
        try:
            capital_cost_data = response.json()["rows"]
        except Exception as e:
            print("Error in fetch capital cost", e)
            capital_cost_data = []

        self.capital_cost_df = pd.DataFrame(capital_cost_data)
        self.original_capital_cost_df = pd.DataFrame(capital_cost_data)

        with open("capital.pickle", "wb") as handle:
            pickle.dump(self.capital_cost_df, handle)

        return (self.capital_cost_df,)

    def change_profile_element_for_material(self, new_element_name):
        self.profile_element_name_for_material = new_element_name
        print("Success Changing")

    # Profile For Current Factory Cost Estimation Method
    def change_profile_for_original_procedure(self, profile_id):
        self.original_procedure_profile_id = profile_id
        print("Success Changing")

    # Get Data From Current Factory Cost Estimation Method as a Reference or Result
    # Of our new Profile using TDABC
    def fetch_process_data(self, start_date, end_date, limit, page=1):
        url = f"{self.api_url}/cost-estimation"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        querystring = {
            "startDate": start_date,
            "endDate": end_date,
            "size": limit,
            "page": page,
            "profile": self.original_procedure_profile_id,
            "hideResultList": "true",
            "placeRestricNotify": "true",
            "correctPlaceOnly": "true",
            "runningNoStart": self.running_no_start,
            "runningNoEnd": self.running_no_end,
            "place": self.costed_place,
        }
        response = requests.get(url, headers=headers, params=querystring)
        process_data = response.json()["rows"]

        self.process_df = pd.DataFrame(process_data)
        self.original_process_df = pd.DataFrame(process_data)

        with open("process.pickle", "wb") as handle:
            pickle.dump(self.process_df, handle)

        return (self.process_df,)

    def load_from_pickle(self):

        try:
            with open("material.pickle", "rb") as handle:
                material_usage_df = pickle.load(handle)
                self.material_usage_df = material_usage_df
                self.original_material_usage_df = material_usage_df
        except:
            print("Error loading material Pickle")

        try:
            with open("employee.pickle", "rb") as handle:
                employee_usage_df = pickle.load(handle)
                self.employee_usage_df = employee_usage_df
                self.original_employee_usage_df = employee_usage_df
        except:
            print("Error loading employee Pickle")

        try:
            with open("capital.pickle", "rb") as handle:
                capital_cost_df = pickle.load(handle)
                self.capital_cost_df = capital_cost_df
                self.original_capital_cost_df = capital_cost_df
        except:
            print("Error loading capital Pickle")

        try:
            with open("process.pickle", "rb") as handle:
                process_df = pickle.load(handle)
                self.process_df = process_df
                self.original_process_df = process_df
        except:
            print("Error loading Process Pickle")

        # self.capital_cost_df = capital_cost_df
        print("Loaded from pickle Successfully")

    def get_process_list(self):
        return self.process_df

    def get_material_usage(self):
        return self.material_usage_df

    def get_employee_usage(self):
        return self.employee_usage_df

    def get_capital_cost(self):
        return self.capital_cost_df

    def load_material_usage(self, material_usage_df):
        self.material_usage_df = material_usage_df

    def load_employee_usage(self, employee_usage_df):
        self.employee_usage_df = employee_usage_df

    def load_capital_cost(self, capital_cost_df):
        self.capital_cost_df = capital_cost_df

    def load_process(self, process_df):
        self.process_df = process_df

    def load_original_process(self, process_df):
        self.original_process_df = process_df

    def load_original_material_usage(self, material_usage_df):
        self.original_material_usage_df = material_usage_df

    def load_original_employee_usage(self, employee_usage_df):
        self.original_employee_usage_df = employee_usage_df

    def load_original_capital_cost(self, capital_cost_df):
        self.original_capital_cost_df = capital_cost_df

    def adjust_material_usage(self):
        new_material_usage_df = pd.DataFrame()  # self.material_usage_df.copy()
        new_material_usage_df["_id"] = self.original_material_usage_df["material_id"]
        new_material_usage_df["process_id"] = self.original_material_usage_df[
            "process_id"
        ]
        new_material_usage_df["name"] = self.original_material_usage_df["material_name"]
        new_material_usage_df["amount"] = self.original_material_usage_df[
            "used_quantity"
        ]
        new_material_usage_df["unit_cost"] = self.original_material_usage_df[
            "unit_cost"
        ]
        # TODO:  Update in EManufac Code to pick the purchase date instead
        new_material_usage_df["date"] = self.original_material_usage_df["used_date"]
        self.material_usage_df = new_material_usage_df

    def adjust_employee_usage(self):
        new_employee_usage_df = pd.DataFrame()
        new_employee_usage_df["_id"] = self.original_employee_usage_df[
            "artifact_employee_id"
        ]
        new_employee_usage_df["employee_id"] = self.original_employee_usage_df[
            "artifact_employee_id"
        ]
        new_employee_usage_df["process_id"] = self.original_employee_usage_df[
            "process_id"
        ]
        new_employee_usage_df["employee_name"] = self.original_employee_usage_df[
            "artifact_employee_name"
        ]
        new_employee_usage_df["amount"] = self.original_employee_usage_df[
            "average_labor_amount"
        ]
        new_employee_usage_df["date"] = self.original_employee_usage_df["receipt_date"]

        # If more than 1 employee (unit cost is same) we group to one, and sum the duration
        new_employee_usage_df["duration"] = (
            self.original_employee_usage_df["artifact_minute_use"]
            * self.original_employee_usage_df["average_labor_amount"]
        )

        # new_employee_usage_df["type"]
        new_employee_usage_df["type"] = "daily"
        new_employee_usage_df['day_amount'] = 1
        try:
            new_employee_usage_df["cost"] = self.original_employee_usage_df[
                "average_daily_labor_cost"
            ]
        except:
            new_employee_usage_df["cost"] = 0

        new_employee_usage_df = new_employee_usage_df.dropna(subset=["cost"])

        self.employee_usage_df = new_employee_usage_df

    def adjust_capital_cost(self):
        new_capital_cost_df = pd.DataFrame()
        try:
            new_capital_cost_df["_id"] = self.original_capital_cost_df[
                "artifact_cost_title"
            ]
            new_capital_cost_df["process_id"] = self.original_capital_cost_df[
                "process_id"
            ]
            new_capital_cost_df["name"] = self.original_capital_cost_df[
                "artifact_cost_title"
            ]
            new_capital_cost_df["cost"] = self.original_capital_cost_df[
                "artifact_capital_cost"
            ]
            new_capital_cost_df["day_amount"] = self.original_capital_cost_df[
                "average_day_amount"
            ]
            new_capital_cost_df["hour_amount"] = self.original_capital_cost_df[
                "average_hour_amount"
            ]
            new_capital_cost_df["unit_cost"] = self.original_capital_cost_df[
                "artifact_unit_cost"
            ]
            new_capital_cost_df["duration"] = self.original_capital_cost_df[
                "artifact_used_time"
            ]
            new_capital_cost_df["date"] = self.original_capital_cost_df["receipt_date"]

            # new_capital_cost_df["machine_hour"] = new_capital_cost_df["machineHour"]
            # new_capital_cost_df["life_time"] = new_capital_cost_df["lifeTime"]
            # new_capital_cost_df["day_per_month"] = new_capital_cost_df["dayPerMonth"]

            zero_cost = new_capital_cost_df[new_capital_cost_df["cost"] == 0]
            # Remove None machine usage
            new_capital_cost_df = new_capital_cost_df.drop(zero_cost.index)
            self.capital_cost_df = new_capital_cost_df
        except Exception as e:
            print("Error in adjust capital cost", e)

    def adjust_process_df(self):
        temp_process_df = self.original_process_df.copy()
        temp_process_df = temp_process_df[temp_process_df["cost"] > 0]

        self.process_df = temp_process_df

    def save_material_csv(self, destination_folder_path="generated"):
        self.material_usage_df.to_csv(
            f"{destination_folder_path}/generated_material_usage.csv"
        )
        self.original_material_usage_df.to_csv(
            f"{destination_folder_path}/original_material_usage.csv"
        )

    def save_process_csv(self, destination_folder_path="generated"):
        self.process_df.to_csv(
            f"{destination_folder_path}/generated_process_data.csv")
        self.original_process_df.to_csv(
            f"{destination_folder_path}/original_process_data.csv"
        )

    def save_employee_csv(self, destination_folder_path="generated"):
        self.employee_usage_df.to_csv(
            f"{destination_folder_path}/generated_employee_usage.csv"
        )
        self.original_employee_usage_df.to_csv(
            f"{destination_folder_path}/original_employee_usage.csv"
        )

    def save_capital_csv(self, destination_folder_path="generated"):
        self.capital_cost_df.to_csv(
            f"{destination_folder_path}/generated_captial_cost.csv"
        )
        self.original_capital_cost_df.to_csv(
            f"{destination_folder_path}/original_captial_cost.csv"
        )

    def save_csv(self, destination_folder_path="generated"):
        self.save_process_csv(destination_folder_path)
        self.save_material_csv(destination_folder_path)
        self.save_employee_csv(destination_folder_path)
        self.save_capital_csv(destination_folder_path)
