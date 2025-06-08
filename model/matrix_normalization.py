import normalize as norm
import importlib

importlib.reload(norm)


def normalize_payload(
    material_cost_matrix,
    material_amount_matrix,
    employee_cost_matrix,
    employee_duration_matrix,
    employee_day_amount_matrix,
    capital_cost_matrix,
    day_amount_matrix,
    capital_cost_duration_matrix,
    validation_payload,
    display_log=False,
):
    max_data = {}
    min_data = {}
    # Normalized Material Matrix
    max_arr, min_arr = norm.find_max_min(material_cost_matrix)
    normalized_material_cost = norm.normalized(
        material_cost_matrix, max_arr, min_arr)
    validate_material_cost_matrix = validation_payload["validate_material_cost_matrix"]

    # Normalize Validate Material Cost Matrix
    normalized_validate_material_cost = norm.normalized_2d(
        validate_material_cost_matrix, material_cost_matrix, max_arr, min_arr
    )
    max_data["material_cost"] = max_arr
    min_data["material_cost"] = min_arr

    # Normalized Material Amount Matrix
    max_arr, min_arr = norm.find_max_min(material_amount_matrix)
    normalized_material_amount = norm.normalized(
        material_amount_matrix, max_arr, min_arr
    )
    validate_material_amount_matrix = validation_payload[
        "validate_material_amount_matrix"
    ]

    if display_log:
        print("normalized_material_amount", normalized_material_amount)
        print("------------")
        print("normalized_material_cost", normalized_material_cost)
        print("------------")

    normalized_validate_material_amount = norm.normalized_2d(
        validate_material_amount_matrix, material_amount_matrix, max_arr, min_arr
    )
    max_data["material_amount"] = max_arr
    min_data["material_amount"] = min_arr

    # Normalized  Employee Cost Matrix
    max_arr, min_arr = norm.find_max_min(employee_cost_matrix)
    normalized_employee_cost = norm.normalized(
        employee_cost_matrix, max_arr, min_arr)
    validate_employee_cost_matrix = validation_payload["validate_employee_cost_matrix"]
    normalized_validate_employee_cost = norm.normalized(
        validate_employee_cost_matrix, max_arr, min_arr
    )
    max_data["employee_cost"] = max_arr
    min_data["employee_cost"] = min_arr

    # Normalized  Employee Duration Matrix
    max_arr, min_arr = norm.find_max_min(employee_duration_matrix)
    normalized_employee_duration = norm.normalized(
        employee_duration_matrix, max_arr, min_arr
    )
    validate_employee_duration_matrix = validation_payload[
        "validate_employee_duration_matrix"
    ]
    normalized_validate_employee_duration = norm.normalized(
        validate_employee_duration_matrix,
        max_arr,
        min_arr,
    )
    max_data["employee_duration"] = max_arr
    min_data["employee_duration"] = min_arr

    # Normalized Employee Day Amount
    max_arr, min_arr = norm.find_max_min(employee_day_amount_matrix)
    normalized_employee_day_amount = norm.normalized(
        employee_day_amount_matrix, max_arr, min_arr
    )
    validate_employee_day_amount_matrix = validation_payload[
        "validate_employee_day_amount_matrix"
    ]
    normalized_validate_employee_day_amount = norm.normalized(
        validate_employee_day_amount_matrix,
        max_arr,
        min_arr,
    )
    max_data["employee_day_amount"] = max_arr
    min_data["employee_day_amount"] = min_arr

    # Normalized Capital Cost Matrix
    max_arr, min_arr = norm.find_max_min(capital_cost_matrix)
    normalized_capital_cost = norm.normalized(
        capital_cost_matrix, max_arr, min_arr)
    validate_capital_cost_matrix = validation_payload["validate_capital_cost_matrix"]
    normalized_validate_capital_cost = norm.normalized(
        validate_capital_cost_matrix, max_arr, min_arr
    )
    max_data["capital_cost"] = max_arr
    min_data["capital_cost"] = min_arr

    # Normalized Day Amount Matrix
    max_arr, min_arr = norm.find_max_min(day_amount_matrix)
    normalized_day_amount = norm.normalized(
        day_amount_matrix, max_arr, min_arr)
    validate_day_amount_matrix = validation_payload["validate_day_amount_matrix"]
    normalized_validate_day_amount = norm.normalized(
        validate_day_amount_matrix, max_arr, min_arr
    )
    max_data["day_amount"] = max_arr
    min_data["day_amount"] = min_arr

    # Normalized Capital Cost Duration Matrix
    validate_capital_duration_matrix = validation_payload[
        "validate_capital_duration_matrix"
    ]
    max_arr, min_arr = norm.find_max_min(capital_cost_duration_matrix)
    normalized_capital_cost_duration = norm.normalized(
        capital_cost_duration_matrix, max_arr, min_arr
    )
    normalized_validate_capital_duration = norm.normalized(
        validate_capital_duration_matrix, max_arr, min_arr
    )
    max_data["capital_cost_duration"] = max_arr
    min_data["capital_cost_duration"] = min_arr

    return (
        normalized_material_cost,
        normalized_material_amount,
        normalized_validate_material_cost,
        normalized_validate_material_amount,
        normalized_employee_cost,
        normalized_employee_duration,
        normalized_employee_day_amount,
        normalized_validate_employee_cost,
        normalized_validate_employee_duration,
        normalized_validate_employee_day_amount,
        normalized_capital_cost,
        normalized_capital_cost_duration,
        normalized_day_amount,
        normalized_validate_capital_cost,
        normalized_validate_capital_duration,
        normalized_validate_day_amount,
        max_data,
        min_data,
    )


def normalize_prediction_payload(
    max_data,
    min_data,
    material_cost_matrix,
    material_amount_matrix,
    employee_cost_matrix,
    employee_duration_matrix,
    employee_day_amount_matrix,
    capital_cost_matrix,
    day_amount_matrix,
    capital_cost_duration_matrix,
):
    # Normalized Material Matrix
    max_arr, min_arr = norm.find_max_min(material_cost_matrix)
    normalized_material_cost = norm.normalized(
        material_cost_matrix, max_arr, min_arr)
    validate_material_cost_matrix = validation_payload["validate_material_cost_matrix"]

    # Normalize Validate Material Cost Matrix
    normalized_validate_material_cost = norm.normalized_2d(
        validate_material_cost_matrix, material_cost_matrix, max_arr, min_arr
    )
    max_data["material_cost"] = max_arr
    min_data["material_cost"] = min_arr

    # Normalized Material Amount Matrix
    max_arr, min_arr = norm.find_max_min(material_amount_matrix)
    normalized_material_amount = norm.normalized(
        material_amount_matrix, max_arr, min_arr
    )
    validate_material_amount_matrix = validation_payload[
        "validate_material_amount_matrix"
    ]

    if display_log:
        print("normalized_material_amount", normalized_material_amount)
        print("------------")
        print("normalized_material_cost", normalized_material_cost)
        print("------------")

    normalized_validate_material_amount = norm.normalized_2d(
        validate_material_amount_matrix, material_amount_matrix, max_arr, min_arr
    )
    max_data["material_amount"] = max_arr
    min_data["material_amount"] = min_arr

    # Normalized  Employee Cost Matrix
    max_arr, min_arr = norm.find_max_min(employee_cost_matrix)
    normalized_employee_cost = norm.normalized(
        employee_cost_matrix, max_arr, min_arr)
    validate_employee_cost_matrix = validation_payload["validate_employee_cost_matrix"]
    normalized_validate_employee_cost = norm.normalized(
        validate_employee_cost_matrix, max_arr, min_arr
    )
    max_data["employee_cost"] = max_arr
    min_data["employee_cost"] = min_arr

    # Normalized  Employee Duration Matrix
    max_arr, min_arr = norm.find_max_min(employee_duration_matrix)
    normalized_employee_duration = norm.normalized(
        employee_duration_matrix, max_arr, min_arr
    )
    validate_employee_duration_matrix = validation_payload[
        "validate_employee_duration_matrix"
    ]
    normalized_validate_employee_duration = norm.normalized(
        validate_employee_duration_matrix,
        max_arr,
        min_arr,
    )
    max_data["employee_duration"] = max_arr
    min_data["employee_duration"] = min_arr

    # Normalized Employee Day Amount
    max_arr, min_arr = norm.find_max_min(employee_day_amount_matrix)
    normalized_employee_day_amount = norm.normalized(
        employee_day_amount_matrix, max_arr, min_arr
    )
    validate_employee_day_amount_matrix = validation_payload[
        "validate_employee_day_amount_matrix"
    ]
    normalized_validate_employee_day_amount = norm.normalized(
        validate_employee_day_amount_matrix,
        max_arr,
        min_arr,
    )
    max_data["employee_day_amount"] = max_arr
    min_data["employee_day_amount"] = min_arr

    # Normalized Capital Cost Matrix
    max_arr, min_arr = norm.find_max_min(capital_cost_matrix)
    normalized_capital_cost = norm.normalized(
        capital_cost_matrix, max_arr, min_arr)
    validate_capital_cost_matrix = validation_payload["validate_capital_cost_matrix"]
    normalized_validate_capital_cost = norm.normalized(
        validate_capital_cost_matrix, max_arr, min_arr
    )
    max_data["capital_cost"] = max_arr
    min_data["capital_cost"] = min_arr

    # Normalized Day Amount Matrix
    max_arr, min_arr = norm.find_max_min(day_amount_matrix)
    normalized_day_amount = norm.normalized(
        day_amount_matrix, max_arr, min_arr)
    validate_day_amount_matrix = validation_payload["validate_day_amount_matrix"]
    normalized_validate_day_amount = norm.normalized(
        validate_day_amount_matrix, max_arr, min_arr
    )
    max_data["day_amount"] = max_arr
    min_data["day_amount"] = min_arr

    # Normalized Capital Cost Duration Matrix
    validate_capital_duration_matrix = validation_payload[
        "validate_capital_duration_matrix"
    ]
    max_arr, min_arr = norm.find_max_min(capital_cost_duration_matrix)
    normalized_capital_cost_duration = norm.normalized(
        capital_cost_duration_matrix, max_arr, min_arr
    )
    normalized_validate_capital_duration = norm.normalized(
        validate_capital_duration_matrix, max_arr, min_arr
    )
    max_data["capital_cost_duration"] = max_arr
    min_data["capital_cost_duration"] = min_arr

    return (
        normalized_material_cost,
        normalized_material_amount,
        normalized_validate_material_cost,
        normalized_validate_material_amount,
        normalized_employee_cost,
        normalized_employee_duration,
        normalized_employee_day_amount,
        normalized_validate_employee_cost,
        normalized_validate_employee_duration,
        normalized_validate_employee_day_amount,
        normalized_capital_cost,
        normalized_capital_cost_duration,
        normalized_day_amount,
        normalized_validate_capital_cost,
        normalized_validate_capital_duration,
        normalized_validate_day_amount,
        max_data,
        min_data,
    )
