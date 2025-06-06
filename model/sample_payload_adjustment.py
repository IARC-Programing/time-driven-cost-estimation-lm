import numpy as np


def get_sample_payload(
    sample_payload,
    material_element,
    employee_element,
    capital_cost_element,
    material_cost_input,
    material_amount_input,
    employee_input,
    emp_dur_input,
    employee_dayamount_input,
    capital_cost_input,
    day_amount_input,
    capital_cost_dur_input,
    predicted_mc,
    predicted_ec,
    predicted_cc,
    bias,
    result_input,
    result,
    error,
    percent_loss,
    epoch_number,
    sample_number,
    model_weights,
):
    # Sample Payload for Debugging
    # Reshape weight to 2D
    # Material Weight
    i = epoch_number
    j = sample_number
    model_bias = bias

    material_element.check_type("material")
    material_elem_weight = material_element.get_weights()
    x, y, z = np.array(material_elem_weight).shape
    material_elem_weight = np.array(material_elem_weight).reshape(x, y)
    # Material Bias
    material_elem_bias = material_element.get_biases()
    x, y, z = np.array(material_elem_bias).shape
    material_elem_bias = np.array(material_elem_bias).reshape(x, y)
    employee_element.check_type("employee")
    # Employee Weight
    employee_elem_weight = employee_element.get_weights()
    x, y, z = np.array(employee_elem_weight).shape
    employee_elem_weight = np.array(employee_elem_weight).reshape(x, y)
    # Employee Bias
    employee_elem_bias = employee_element.get_biases()
    x, y, z = np.array(employee_elem_bias).shape
    employee_elem_bias = np.array(employee_elem_bias).reshape(x, y)
    # Capital Cost Weight
    capital_cost_element.check_type("capital")
    capital_elem_weight = capital_cost_element.get_weights()
    x, y, z = np.array(capital_elem_weight).shape
    capital_elem_weight = np.array(capital_elem_weight).reshape(x, y)
    # Capital Cost Bias
    capital_elem_bias = capital_cost_element.get_biases()
    x, y, z = np.array(capital_elem_bias).shape
    capital_elem_bias = np.array(capital_elem_bias).reshape(x, y)
    sample_payload.append(
        {
            "epoch": (i - 1),
            "sample": j,
            "material_cost": material_cost_input,
            "material_amount": material_amount_input,
            "material_weight": material_elem_weight,
            "material_bias": material_elem_bias,
            "employee_cost": employee_input,
            "employee_duration": emp_dur_input,
            "employee_dayamount": employee_dayamount_input,
            "employee_weight": employee_elem_weight,
            "employee_bias": employee_elem_bias,
            "capital_cost": capital_cost_input,
            "day_amount": day_amount_input,
            "capital_cost_duration": capital_cost_dur_input,
            "capital_cost_weight": capital_elem_weight,
            "capital_cost_bias": capital_elem_bias,
            "result": result_input,
            "result_predict": result,
        }
    )
    material_costs = material_cost_input.flatten()
    for idx, cost in enumerate(material_costs):
        sample_payload[-1][f"material_cost_{idx + 1}"] = cost
    material_amounts = material_amount_input.flatten()
    for idx, amount in enumerate(material_amounts):
        sample_payload[-1][f"material_amount_{idx + 1}"] = amount
    material_weights = material_elem_weight.flatten()
    for idx, weight in enumerate(material_weights):
        sample_payload[-1][f"material_weight_{idx + 1}"] = weight
    material_biases = material_elem_bias.flatten()
    for idx, bias in enumerate(material_biases):
        sample_payload[-1][f"material_bias_{idx + 1}"] = bias
    employee_costs = employee_input.flatten()
    for idx, cost in enumerate(employee_costs):
        sample_payload[-1][f"employee_cost_{idx + 1}"] = cost
    employee_durations = emp_dur_input.flatten()
    for idx, duration in enumerate(employee_durations):
        sample_payload[-1][f"employee_duration_{idx + 1}"] = duration
    employee_dayamounts = employee_dayamount_input.flatten()
    for idx, amount in enumerate(employee_dayamounts):
        sample_payload[-1][f"employee_dayamount_{idx + 1}"] = amount
    employee_weights = employee_elem_weight.flatten()
    for idx, weight in enumerate(employee_weights):
        sample_payload[-1][f"employee_weight_{idx + 1}"] = weight
    employee_biases = employee_elem_bias.flatten()
    for idx, bias in enumerate(employee_biases):
        sample_payload[-1][f"employee_bias_{idx + 1}"] = bias
    capital_costs = capital_cost_input.flatten()
    for idx, cost in enumerate(capital_costs):
        sample_payload[-1][f"capital_cost_{idx + 1}"] = cost
    day_amounts = day_amount_input.flatten()
    for idx, amount in enumerate(day_amounts):
        sample_payload[-1][f"day_amount_{idx + 1}"] = amount
    capital_cost_durations = capital_cost_dur_input.flatten()
    for idx, duration in enumerate(capital_cost_durations):
        sample_payload[-1][f"capital_cost_duration_{idx + 1}"] = duration
    capital_cost_weights = capital_elem_weight.flatten()
    for idx, weight in enumerate(capital_cost_weights):
        sample_payload[-1][f"capital_cost_weight_{idx + 1}"] = weight
    capital_cost_bias = capital_elem_bias.flatten()
    for idx, bias in enumerate(capital_cost_bias):
        sample_payload[-1][f"capital_cost_bias_{idx + 1}"] = bias
    predicted_mc_reshape = np.array(predicted_mc).flatten()
    for idx, res in enumerate(predicted_mc_reshape):
        sample_payload[-1]["total_material"] = res
    predicted_ec_reshape = np.array(predicted_ec).flatten()
    for idx, res in enumerate(predicted_ec_reshape):
        sample_payload[-1]["total_daily_employee"] = res
    predicted_cc_reshape = np.array(predicted_cc).flatten()
    for idx, res in enumerate(predicted_cc_reshape):
        sample_payload[-1]["total_capital_cost"] = res

    model_bias_reshape = np.array(model_bias).flatten()
    for idx, res in enumerate(model_bias_reshape):
        sample_payload[-1]["model_bias"] = res

    model_weights_reshape = np.array(model_weights).flatten()
    for idx, res in enumerate(model_weights_reshape):
        sample_payload[-1][f"model_weight_{idx + 1}"] = res
    result_reshape = np.array(result_input).flatten()
    for idx, res in enumerate(result_reshape):
        sample_payload[-1]["result"] = res
    result_predict_reshape = np.array(result).flatten()
    for idx, res in enumerate(result_predict_reshape):
        sample_payload[-1]["result_predict"] = res
    sample_payload[-1]["error"] = error
    sample_payload[-1]["error_percent"] = percent_loss

    return sample_payload
