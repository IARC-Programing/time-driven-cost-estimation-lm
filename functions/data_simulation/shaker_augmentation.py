# This find provide the shaker function
# for make data into more variation
# and augment the data for training the model
# In first case study we will shake the data from dataset 1
# And Create the dataset 4

import pandas as pd


def dataset_shake(dataset_name, new_dataset_name, random_rate=10):
    # import data from dataset
    process_df = pd.read_csv(f'data/{dataset_name}/generated_process_data.csv')
    employee_df = pd.read_csv(
        f'data/{dataset_name}/generated_employee_usage.csv')
    material_df = pd.read_csv(
        f'data/{dataset_name}/generated_material_usage.csv')
    capitalcost_df = pd.read_csv(
        f'data/{dataset_name}/generated_captial_cost.csv')

    # Adjust Employee Cost
    # Picking Cost
    example_employee_ids = range(0, 12)
    example_employee_name = ['A', 'B', "C", "D",
                             "E", "F", "G", "H", "I", "J", "K", "L"]
    example_cost = [27769.652, 20508.323, 811.6214, 547.8455,
                    392.4559, 372.7312, 354.8132, 2682.522,
                    2451.125, 24071.33, 22683.42, 22683.42]
    example_day_amount = [26, 26, 1, 1, 1, 1, 1, 7, 7, 7, 7, 7]

    picking_df = employee_df[employee_df['employee_name']
                             == "ตัวแทนพนักงานเฉลี่ยคลังที่ 2 (ดองน้ำแข็ง)"]
    example_picking_df = pd.DataFrame({
        'employee_id': example_employee_ids,
        'employee_name': example_employee_name,
        'cost': example_cost,
        'day_amount': example_day_amount
    })

    example_picking_df['cost'] = example_picking_df['cost'].astype(float)
    example_picking_df['day_amount'] = example_picking_df['day_amount'].astype(
        int)

    # Update the picking_df based on the condition
    for i in range(len(picking_df)):
        cost_mod_index = i % 12
        coefficient = random_rate / 2
        if cost_mod_index % 2 == 0:
            coefficient = -random_rate / 2
        picking_df.loc[i, ['employee_id', 'employee_name', 'cost', 'day_amount']
                       ] = example_picking_df.loc[cost_mod_index, ['employee_id', 'employee_name', 'cost', 'day_amount']]
        # Update Duration +- 5%
        picking_df.loc[i, 'duration'] = picking_df.loc[i, 'duration'] + \
            (coefficient * picking_df.loc[i, 'duration'] / 100)

    # Packing Cost
    packing_df = employee_df[employee_df['employee_name']
                             == "ตัวแทนพนักงานเฉลี่ยคลังที่ 3 (ปูเข้าเป็นกระป๋อง)"]
    packing_df.reset_index(drop=True, inplace=True)
    example_employee_ids = range(12, 24)
    example_employee_name = ['M', 'N', "O", "P",
                             "Q", "R", "S", "T", "U", "V", "W", "X"]
    example_cost = [37769.652, 30508.323, 311.6214, 647.8455,
                    492.4559, 472.7312, 454.8132, 3682.522,
                    3451.125, 34071.33, 22683.42, 32683.42]
    example_day_amount = [26, 26, 1, 1, 1, 1, 1, 7, 7, 7, 7, 7]

    example_packing_df = pd.DataFrame({
        'employee_id': example_employee_ids,
        'employee_name': example_employee_name,
        'cost': example_cost,
        'day_amount': example_day_amount
    })
    example_packing_df['cost'] = example_packing_df['cost'].astype(float)
    example_packing_df['day_amount'] = example_packing_df['day_amount'].astype(
        int)

    # Update the picking_df based on the condition
    for i in range(len(packing_df)):
        cost_mod_index = i % 12
        coefficient = random_rate / 2
        if cost_mod_index % 2 == 0:
            coefficient = -random_rate / 2
        packing_df.loc[i, ['employee_id', 'employee_name', 'cost', 'day_amount']
                       ] = example_packing_df.loc[cost_mod_index, ['employee_id', 'employee_name', 'cost', 'day_amount']]
        # Update Duration +- random_rate %
        packing_df.loc[i, 'duration'] = packing_df.loc[i, 'duration'] + \
            (coefficient * packing_df.loc[i, 'duration'] / 100)

    # Combining
    new_employee_df = pd.concat([picking_df, packing_df], ignore_index=True)

    # Adjust Capital Cost
    new_capital_cost = capitalcost_df.copy()
    for i in range(len(new_capital_cost)):
        cost_mod_index = i % random_rate
        amout_mod_index = i % random_rate / 2
        coefficient = 1
        if cost_mod_index % 2 == 0:
            coefficient = -1

        # Update Cost +- 1 - 10%
        new_capital_cost.loc[i, 'cost'] = new_capital_cost.loc[i, 'cost'] + \
            (coefficient * cost_mod_index *
             new_capital_cost.loc[i, 'cost'] / 100)
        new_capital_cost.loc[i, 'duration'] = new_capital_cost.loc[i, 'duration'] + (
            coefficient * amout_mod_index * new_capital_cost.loc[i, 'duration'] / 100)

    # Adjust Material Cost
    new_material_cost = material_df.copy()
    for i in range(len(new_material_cost)):
        cost_mod_index = i % random_rate
        amout_mod_index = i % random_rate / 2
        coefficient = 1
        if cost_mod_index % 2 == 0:
            coefficient = -1

        # Update Cost +- 1 - 10%
        new_material_cost.loc[i, 'unit_cost'] = new_material_cost.loc[i, 'unit_cost'] + (
            coefficient * cost_mod_index * new_material_cost.loc[i, 'unit_cost'] / 100)
        new_material_cost.loc[i, 'amount'] = new_material_cost.loc[i, 'amount'] + (
            coefficient * amout_mod_index * new_material_cost.loc[i, 'amount'] / 100)

    # Save data
    process_df.to_csv(
        f'data/{new_dataset_name}/generated_process_data.csv', index=False)
    new_employee_df.to_csv(
        f'data/{new_dataset_name}/generated_employee_usage.csv', index=False)
    new_material_cost.to_csv(
        f'data/{new_dataset_name}/generated_material_usage.csv', index=False)
    new_capital_cost.to_csv(
        f'data/{new_dataset_name}/generated_captial_cost.csv', index=False)
    # Print the result
    print(
        f"Data from {dataset_name} has been shaken and saved as {new_dataset_name}.")
