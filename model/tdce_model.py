import numpy as np
import random
import pandas as pd
import importlib
import time

import material_network as mn
import time_driven_network as tdn
import loss
import normalize as norm
import matrix_normalization as mnorm
import sample_payload_adjustment as spa
import weight_activation as wa

importlib.reload(mn)
importlib.reload(tdn)
importlib.reload(loss)
importlib.reload(norm)
importlib.reload(mnorm)
importlib.reload(spa)
importlib.reload(wa)


# Generated from Gemini
def generate_random_numbers():
    # Generate a random number between 0 and 1 (exclusive)
    random_val = random.random()
    number1 = random_val
    # The second number is another random value between 0 and (1 - number1)
    number2 = random.random() * (1 - number1)
    # The third number is simply 1 minus the sum of the first two
    number3 = 1 - number1 - number2

    return np.array([number1, number2, number3]).reshape(3, 1)


# Model Reference
# Early Stopping https://medium.com/@juanc.olamendy/understanding-early-stopping-a-key-to-preventing-overfitting-in-machine-learning-17554fc321ff


class TDCEModel:
    def __init__(self):
        self.material_element = mn.MaterialNetwork()
        self.employee_element = tdn.TimeDrivenNetwork()
        self.capital_cost_element = tdn.TimeDrivenNetwork()
        random_weight = np.array([1.0, 1.0, 1.0]).reshape(
            3, 1
        )  # generate_random_numbers()
        self.weights = random_weight
        self.bias = np.full((1, 1), 0.0)
        self.loss = loss.mse
        self.loss_prime = loss.mse_prime
        self.loss_percent = loss.rmspe
        self.gradient = np.array([[0, 0, 0]])
        self.prediction_error = np.array([[0, 0, 0]])
        self.weight_list = np.array([[[0], [0], [0]]])
        self.epoch_error = []
        self.sample_errors = []
        self.material_learning_rate = 0.001
        self.employee_learning_rate = 0.001
        self.capital_cost_learning_rate = 0.001
        self.max_data = {}
        self.min_data = {}
        self.sample_payload = []
        self.use_early_stopping = False
        self.patience = 10
        self.use_model_weight = False

    def inital_inside_element(
        self,
        material_layer,
        employee_layer,
        capital_cost_layer,
    ):
        self.material_element.add(material_layer)
        self.material_element.use(loss.mse, loss.mse_prime)
        self.employee_element.add(employee_layer)
        self.employee_element.use(loss.mse, loss.mse_prime)
        self.capital_cost_element.add(capital_cost_layer)
        self.capital_cost_element.use(loss.mse, loss.mse_prime)
        print("Initial Successfully")

    # For Setting New Network Element

    def set_material_element(self, material_network):
        self.material_element = material_network

    def set_employee_element(self, employee_element):
        self.employee_element = employee_element

    def set_capital_cost_element(self, capital_cost_element):
        self.capital_cost_element = capital_cost_element

    def use(self, loss, loss_prime, loss_percent):
        self.loss = loss
        self.loss_prime = loss_prime
        self.loss_percent = loss_percent

    def set_learning_rate(self, material_lr, employee_lr, cc_lr):
        self.material_learning_rate = material_lr
        self.employee_learning_rate = employee_lr
        self.capital_cost_learning_rate = cc_lr
        print("Learning Rate Set Successfully")
        print(
            f"Material LL {material_lr},  Employee LL {employee_lr}, Capital Cost LL {cc_lr}"
        )

    def activate_early_stopping(self):
        self.use_early_stopping = True
        print("Activate Early Stopping Successfully")

    def deactivate_early_stopping(self):
        self.use_early_stopping = False
        print("Deactivate Early Stopping Successfully")

    def activete_model_weight(self):
        self.use_model_weight = True
        print("Activate Model Weight Successfully")

    def deactivate_model_weight(self):
        self.use_model_weight = False
        print("Deactivate Model Weight Successfully")

    # For Early Stopping
    def edit_patience_round(self, patience_round):
        self.patience = patience_round

    def fit_with_validation(
        self,
        material_cost_matrix,
        material_amount_matrix,
        employee_cost_matrix,
        employee_duration_matrix,
        employee_day_amount_matrix,
        capital_cost_matrix,
        day_amount_matrix,
        capital_cost_duration_matrix,
        result_matrix,
        epoch,
        learning_rate,
        validation_payload,
        display_round_log=False,
    ):
        results = []
        epoch_errors = []
        sample_errors = []
        sample_payload = []

        # For Early Stopping
        patience_counter = 0
        best_validation_error = np.inf

        # print('Material Cost Matrix')
        # print(material_cost_matrix)
        # print('-----------------------------------')
        # print('Material Amount Matrix')
        # print(material_amount_matrix)
        # print('------------------------------')
        # print('Daily Employee Matrix')
        # print(daily_employee_cost_matrix)
        # print('------------------------------')
        # print('Monthly Employee Matrix')
        # print(monthly_employee_cost_matrix)
        # print('------------------------------')
        print(f"W:  Initial Weight {self.weights}")

        # Duration Represent the amount of process
        samples = len(result_matrix)
        validate_result_matrix = validation_payload["validate_result_matrix"]

        # Some of validate payload use normalized_validate because size is not
        # same as train payload if they are in 3D it will use only normalized function

        (
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
        ) = mnorm.normalize_payload(
            material_cost_matrix=material_cost_matrix,
            material_amount_matrix=material_amount_matrix,
            employee_cost_matrix=employee_cost_matrix,
            employee_duration_matrix=employee_duration_matrix,
            employee_day_amount_matrix=employee_day_amount_matrix,
            capital_cost_matrix=capital_cost_matrix,
            day_amount_matrix=day_amount_matrix,
            capital_cost_duration_matrix=capital_cost_duration_matrix,
            validation_payload=validation_payload,
            display_log=False,
        )

        self.min_data = min_data
        self.max_data = max_data

        for i in range(epoch):
            error = 0
            sum_error = 0
            sum_error_percent = 0
            sum_validation_error = 0
            sum_validation_error_percent = 0
            start_time = time.time()
            validate_sample_amount = 0

            for j in range(samples):
                # Result
                result_input = result_matrix[j]
                # Material
                # material_cost_input = material_cost_matrix[j]
                material_cost_input = normalized_material_cost[j]

                # material_amount_input = material_amount_matrix[j]
                material_amount_input = normalized_material_amount[j]

                # Employee / Labor
                employee_cost_input = normalized_employee_cost[j]
                employee_duration_input = normalized_employee_duration[j]
                employee_day_amount_input = normalized_employee_day_amount[j]
                # Capital Cost
                capital_cost_input = normalized_capital_cost[j]
                day_amount_input = normalized_day_amount[j]
                capital_cost_dur_input = normalized_capital_cost_duration[j]

                if j < len(normalized_validate_material_cost):
                    # VALIDATION INPUT
                    validate_material_cost_input = normalized_validate_material_cost[j]
                    validate_material_amount_input = (
                        normalized_validate_material_amount[j]
                    )
                    validate_employee_input = (
                        normalized_validate_employee_cost[j]
                    )
                    validate_emp_dur_input = (
                        normalized_validate_employee_duration[j]
                    )
                    validate_emp_dayamount_input = (
                        normalized_validate_employee_day_amount[j]
                    )
                    validate_capital_cost_input = normalized_validate_capital_cost[j]
                    validate_dayamount_input = normalized_validate_day_amount[j]
                    validate_capital_cost_dur_input = (
                        normalized_validate_capital_duration[j]
                    )
                    validate_result_input = validate_result_matrix[j]

                # Train Material ELE
                predicted_mc = self.material_element.predict_sample(
                    cost_input=material_cost_input, amount_input=material_amount_input
                )

                #  Employee Cost ELE
                predicted_ec = self.employee_element.predict_sample(
                    cost_input=employee_cost_input,
                    time_input=employee_duration_input,
                    day_amount=employee_day_amount_input
                )

                # Capital Cost ELE
                predicted_cc = self.capital_cost_element.predict_sample(
                    cost_input=capital_cost_input,
                    time_input=capital_cost_dur_input,
                    day_amount=day_amount_input,
                )

                # For Validation
                # Predict will not update class vairable while predict_sample
                # which will be update their variable for train

                if j < len(normalized_validate_material_cost):
                    # Test Material ELE
                    validate_predicted_mc = self.material_element.predict(
                        cost_input=validate_material_cost_input,
                        amount_input=validate_material_amount_input,
                    )

                    # Test Monthly Employee Cost ELE
                    validate_predicted_ec = self.employee_element.predict(
                        cost_input=validate_employee_input,
                        time_input=validate_emp_dur_input,
                        day_amount=validate_emp_dayamount_input
                    )

                    # Test Capital Cost ELE
                    validate_predicted_cc = self.capital_cost_element.predict(
                        cost_input=validate_capital_cost_input,
                        time_input=validate_capital_cost_dur_input,
                        day_amount=validate_dayamount_input,
                    )

                    validate_result = (
                        validate_predicted_mc * self.weights[0]
                        + validate_predicted_ec * self.weights[1]
                        + validate_predicted_cc * self.weights[2]
                    ) + self.bias

                # Result Combination and Find Error
                result = (
                    predicted_mc * self.weights[0]
                    + predicted_ec * self.weights[1]
                    + predicted_cc * self.weights[2]
                ) + self.bias

                # if result < 0:
                #     result = [[0]]

                # print(
                #     f"predicted_mc {predicted_mc} predicted_dmc {predicted_dmc} predicted_mec{predicted_mec} predicted_cc{predicted_cc}")

                # Find MSE both Result and validation
                error = self.loss(result_input, result)

                # Find RMSPE both Result and Validate
                percent_loss = self.loss_percent(result_input, result)

                # Find Derivative of Loss
                loss_prime = self.loss_prime(result_input, result)
                bias_error = loss_prime

                if display_round_log:
                    print(
                        f"Epoch {i} Sample {j} : Model level weight {self.weights}")
                    print(f"Epoch {i} Sample {j} : Bias {self.bias}")
                    print(
                        f"Epoch {i} Sample {j} : Result {result}, Validate Result {validate_result}"
                    )
                    print(
                        f"Epoch {i} Sample {j} : Material Result {predicted_mc}  Employee {predicted_ec}  Capital Cost {predicted_cc}"
                    )
                    print(
                        f"Epoch {i} Sample {j} : Actual Result {result_input}")
                    print(
                        f"Epoch {i} Sample {j} : Error (MSE) {error}, Error To Adjust(Loss Prime) {loss_prime}  "
                    )
                    print(
                        f"Epoch {i} Sample {j} : Error Percent {percent_loss} ")
                    print("-----------------")

                # Find Gradient of each weight
                # mse_prime dot Leaky_relu(input)
                # For Adjust Grdient in Model Level
                material_weight_error = np.dot(predicted_mc, loss_prime)
                ec_weight_error = np.dot(predicted_ec, loss_prime)
                cc_weight_error = np.dot(predicted_cc, loss_prime)

                # Append For log keeping
                sample_errors.append(
                    {
                        "epoch": (i - 1),
                        "sample": j,
                        "error": error,
                        "error_percent": percent_loss,
                    }
                )

                # Sample Payload for Debugging
                sample_payload = spa.get_sample_payload(
                    sample_payload=sample_payload,
                    material_element=self.material_element,
                    employee_element=self.employee_element,
                    capital_cost_element=self.capital_cost_element,
                    material_cost_input=material_cost_input,
                    material_amount_input=material_amount_input,
                    employee_input=employee_cost_input,
                    emp_dur_input=employee_duration_input,
                    employee_dayamount_input=employee_day_amount_input,
                    capital_cost_input=capital_cost_input,
                    day_amount_input=day_amount_input,
                    capital_cost_dur_input=capital_cost_dur_input,
                    predicted_mc=predicted_mc,
                    predicted_ec=predicted_ec,
                    predicted_cc=predicted_cc,
                    bias=self.bias,
                    result_input=result_input,
                    result=result,
                    error=error,
                    percent_loss=percent_loss,
                    epoch_number=i,
                    sample_number=j,
                    model_weights=self.weights,
                )

                # print(f'Result {result} & Loss {loss_prime}')
                # print(
                #     f'Material Gradient {material_weight_error}, DE Gradient {de_weight_error}, ME Gradient {me_weight_error}, CC Gradient {cc_weight_error}')

                # Back Propagation of Inside Element
                # mse_prime dot w
                sum_material_error = np.dot(loss_prime, self.weights[0])
                self.material_element.back_propagate(
                    sum_material_error, self.material_learning_rate
                )

                sum_ec_error = np.dot(loss_prime, self.weights[1])
                self.employee_element.back_propagate(
                    sum_ec_error, self.employee_learning_rate
                )

                sum_capital_error = np.dot(loss_prime, self.weights[2])
                self.capital_cost_element.back_propagate(
                    sum_capital_error, self.capital_cost_learning_rate
                )

                if self.use_model_weight is True:
                    # Update Weight
                    overall_gradient = np.array(
                        [
                            material_weight_error[0],
                            ec_weight_error[0],
                            cc_weight_error[0],
                        ]
                    )
                    self.weights -= learning_rate * overall_gradient
                self.bias -= learning_rate * bias_error

                result_input_value = result_input.reshape(1)
                result_input_value = result_input_value[0]

                if j < len(normalized_validate_material_cost):
                    validation_error = self.loss(
                        validate_result_input, validate_result)
                    validation_error_percent = self.loss_percent(
                        validate_result_input, validate_result
                    )
                    sum_validation_error += validation_error
                    sum_validation_error_percent += validation_error_percent
                    validate_sample_amount += 1

                sum_error += error
                sum_error_percent += percent_loss

            end_time = time.time()
            validation_error = sum_validation_error / validate_sample_amount
            print(
                f"{i + 1} /{epoch} Epoch Error = {sum_error / samples} ({sum_error_percent / samples} %), Validate Error = {validation_error} ({sum_validation_error_percent / validate_sample_amount}) estimate time {end_time - start_time}"
            )

            epoch_errors.append(
                {
                    "epoch": (i + 1),
                    "error": sum_error / samples,
                    "error_percent": sum_error_percent / samples,
                    "validate_error": sum_validation_error / validate_sample_amount,
                    "validate_error_percent": sum_validation_error_percent / validate_sample_amount,
                }
            )

            if self.use_early_stopping:
                if validation_error < best_validation_error:
                    best_validation_error = validation_error
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter == self.patience:
                        print(f"Early Stopping at Epoch {i + 1}")
                        break

            sum_error = 0
            sum_error_percent = 0
            sum_validation_error = 0
            sum_validation_error_percent = 0

        self.epoch_error = epoch_errors
        self.sample_errors = sample_errors
        self.sample_payload = sample_payload
        epoch_error_df = pd.DataFrame(epoch_errors)
        last_sample_payload = sample_payload[-1]

        print("-------------------")
        print(
            f"Minimum Error =  {epoch_error_df['error'].min()} Minimum Error percent {round(epoch_error_df['error_percent'].min(), 4)} Accuracy {100 - epoch_error_df['error_percent'].min()}%"
        )
        print(
            f"Average Error =  {epoch_error_df['error'].mean()} Average Error percent {round(epoch_error_df['error_percent'].mean(), 4)} Accuracy {100 - epoch_error_df['error_percent'].mean()}%"
        )
        print(
            f"Maximum Error =  {epoch_error_df['error'].max()} Maximum Error percent {round(epoch_error_df['error_percent'].max(), 2)} Accuracy {100 - epoch_error_df['error_percent'].max()}%"
        )
        print(
            f"Minimum Validation Error =  {epoch_error_df['validate_error'].min()} Minimum Error percent {round(epoch_error_df['validate_error_percent'].min(), 4)} Accuracy {100 - epoch_error_df['validate_error_percent'].min()}%"
        )
        print(
            f"Average Validation Error =  {epoch_error_df['validate_error'].mean()} Average Error percent {round(epoch_error_df['validate_error_percent'].mean(), 4)} Accuracy {100 - epoch_error_df['validate_error_percent'].mean()}%"
        )
        print(
            f"Maximum Validation Error =  {epoch_error_df['validate_error'].max()} Maximum Error percent {round(epoch_error_df['validate_error_percent'].max(), 4)} Accuracy {100 - epoch_error_df['validate_error_percent'].max()}%"
        )
        print(
            f"Last Validation Error =  {epoch_error_df.iloc[-1]['validate_error']} Last Validate Error percent {round(epoch_error_df.iloc[-1]['validate_error_percent'], 4)} Accuracy {100 - epoch_error_df.iloc[-1]['validate_error_percent']}%"
        )
        print(
            f"Final Model - Material Weight {last_sample_payload['material_weight']} Material Bias {last_sample_payload['material_bias']}"
        )
        print(
            f" - Employee Weight {last_sample_payload['employee_weight']} Monthly Employee Bias {last_sample_payload['employee_bias']}"
        )
        print(
            f" - Capital Cost Weight {last_sample_payload['capital_cost_weight']} Capital Cost Bias {last_sample_payload['capital_cost_bias']}"
        )
        print(f" - Model Bias {last_sample_payload['model_bias']}")
        print("-------------------")

        return results

    def get_gradient(self):
        return self.gradient

    def get_prediction_error(self):
        return self.prediction_error

    def get_weight_list(self):
        return self.weight_list

    def get_epoch_error(self):
        return self.epoch_error

    def get_sample_error(self):
        return self.sample_errors

    def get_model_element_weights(self):
        return (
            self.material_element.get_weight_list(),
            self.daily_employee_element.get_weight_list(),
            self.monthly_employee_element.get_weight_list(),
            self.capital_cost_element.get_weight_list(),
        )

    def get_sample_payload(self):
        return self.sample_payload
