
def adjust_to_match_process(capital_cost_usage,
                            employee_usage,
                            material_usage,
                            new_process_df):
    new_capital_cost = capital_cost_usage.copy()
    new_capital_cost = new_capital_cost[new_capital_cost['process_id'].isin(
        new_process_df['process_id'])]
    new_employee_usage = employee_usage.copy()
    new_employee_usage = new_employee_usage[
        new_employee_usage['process_id'].isin(new_process_df['process_id'])]
    new_material_usage = material_usage.copy()
    new_material_usage = new_material_usage[
        new_material_usage['process_id'].isin(new_process_df['process_id'])]
    return new_capital_cost, new_employee_usage, new_material_usage
