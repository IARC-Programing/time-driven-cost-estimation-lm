import pandas as pd
import importlib
import datetime
import sys
import os
import requests

# fmt:off
sys.path.append('../14-New-Final-Model/extractor')

import emanufac_tdabc_extractor_class as tdabc_extractor
importlib.reload(tdabc_extractor)

# fmt:on


# "https://vy.autth.theduckcreator.in.th/api/v1"
url = "http://localhost:3007/api/v1"
api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY1ZmJhOTM2ZmRmNTFkMDM4MDRlODlkMyIsInVzZXJuYW1lIjoic3VwZXJ1c2VyIiwiZXhwIjoxNzUwNjYzNDM4LCJpYXQiOjE3NDI4ODc0Mzh9.b7VQbxhiJO03ionBBD5H8I-IYxO4psfenkdeb1s1bvc"
start_date = ""  # datetime.datetime(2024, 2,7, 6,0)
end_date = ""  # datetime.datetime(2025, 2,12,12,30)
profile_id = "671d24916f0d9131b270c036"
cost_place = "65fbd3398382640527b795c3"

limit = 1000


def get_data(running_start="68309336",
             runing_end="68309858",
             directory_name="complicate_time_series_may_2024"):
    # Initial Class
    extractor = tdabc_extractor.EManufacTDABCExtractor(
        api_key=api_key, api_url=url, costed_place='65fbd3398382640527b795c3',
        profile_id="671d24916f0d9131b270c036")

    extractor.change_profile_for_original_procedure("671d2021b54378e1c72d2074")
    extractor.set_running_no_margin(running_start, runing_end)

    # Fetch Material
    two_month_change = datetime.timedelta(days=0)

    if end_date == '' and start_date == '':
        print('On First case')
        material_usage = extractor.fetch_material_usage(
            start_date=start_date, end_date=end_date, limit=limit*3, page=1)
    else:
        material_usage = extractor.fetch_material_usage(
            start_date=start_date - two_month_change, end_date=end_date, limit=limit*3, page=1)

    # Adjust Material Usage
    extractor.adjust_material_usage()

    # Fetch Employee Usage
    if end_date == '' and start_date == '':
        print('on first case')
        employee_usage = extractor.fetch_employee_usage(
            start_date=start_date, end_date=end_date, limit=limit*2, page=1)
    else:
        employee_usage = extractor.fetch_employee_usage(
            start_date=start_date - two_month_change, end_date=end_date, limit=limit*2, page=1)

    # Adjust Employee Usage
    extractor.adjust_employee_usage()

    # Get Capital Cost
    if end_date == '' and start_date == '':
        capital_cost_usage = extractor.fetch_capital_cost_usage(
            start_date=start_date, end_date=end_date, limit=limit*2, page=1
        )
    else:
        capital_cost_usage = extractor.fetch_capital_cost_usage(
            start_date=start_date - two_month_change, end_date=end_date, limit=limit*2, page=1
        )

    # Adjust Capital Cost
    extractor.adjust_capital_cost()

    # Get Proces
    process_df = extractor.fetch_process_data(
        start_date, end_date, limit=limit, page=1)

    # Adjust Process
    extractor.adjust_process_df()

    # Create Folder
    try:
        os.mkdir(directory_name)
    except FileExistsError:
        print('Folder is Exist')
        pass

    # Save
    extractor.save_material_csv(directory_name)
    extractor.save_employee_csv(directory_name)
    extractor.save_capital_csv(directory_name)
    extractor.save_process_csv(directory_name)

    print('Success')
