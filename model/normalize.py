import numpy as np


def find_max_min(input_array):
    input_np_arr = np.array(input_array)
    x, y, z = input_np_arr.shape
    if z == 0:
        return None, None
    input_np_arr = input_np_arr.reshape(-y, z)
    # Find Max
    max_array = np.max(input_np_arr, axis=0)
    min_array = np.min(input_np_arr, axis=0)
    data_amount = max_array - min_array
    data_margin = np.multiply(data_amount, 0.15)
    max_with_added = np.add(max_array, data_margin)
    # Find Min
    min_with_added = np.subtract(min_array, data_margin)
    min_with_added[min_with_added < 0] = 0

    return max_with_added, min_with_added


def normalized(input_array, max_arr, min_arr):
    input_np_arr = np.array(input_array, dtype=np.float32)
    x, y, z = input_np_arr.shape
    if z == 0:
        return input_np_arr
    input_np_arr = input_np_arr.reshape(-y, z)
    normalized_array = input_np_arr.copy()
    for i in range(0, len(input_np_arr)):
        for j in range(0, len(input_np_arr[i])):
            max = max_arr[j]
            min = min_arr[j]
            if max == min:
                new_data = 1
            else:
                new_data = (input_np_arr[i][j] - min) / (max - min)
            normalized_array[i][j] = new_data

    normalized_array = normalized_array.reshape(x, y, z)
    return normalized_array


def normalized_2d(validate_payload, train_payload, max_arr, min_arr):
    x, y, z = train_payload.shape

    if z == 0:
        return validate_payload
    print('----')
    print("Validate Payload size", validate_payload.shape)
    print("Train Payload size", train_payload.shape)
    x, y = np.array(validate_payload).shape
    validate_payload = validate_payload.reshape(x, 1, z)
    result = normalized(validate_payload, max_arr, min_arr)
    return result


def denormalize(value, max, min):
    data_amount = max - min
    data_margin = np.multiply(data_amount, 0.15)
    max_with_added = max + data_margin
    # Find Min
    min_with_added = min - data_margin
    if min_with_added < 0:
        min_with_added = 0

    denormalized_vaule = (
        value * (max_with_added - min_with_added)) + min_with_added

    return denormalized_vaule
