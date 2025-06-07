# Time Driven Cost Estimation Learning Model (TDCE)

This is the open-source part of Time-Driven Cost Estimation Learning Model under the research titled "Artificial Neural Network like for Manufacturing Cost Estimation" wish to create neural network like system by using the core equation of time-driven activity-based costing.

## Starting

Creating ENV followed by `.env.example` and then Creating your virtual environment

```
python -m venv venv
```

Install all requirements

```
pip install -r requirement.txt
```

## Repository Structure

The Model is structure into folder model, functions. The model and functions that related to **model** is located in folder model and the **function** contain the function to do the experiment.

## Model

The model is located in `model/` folder. The main file is `tdce_model.py` which is a based model of TDCE.

### Main Model

TDCE class is the main class of our model. It was constructed by 3 elements which are `self.material_element`, `selef.employee_element`, and `self.capital_cost_element`. The material element is non-time driven element, it came from the `material_network.py` while employee and capital cost (or called utility cost in the thesis and article) were time-driven elements, which came from `time_driven_network`.

On this main class, it located function `fit_with_validation` for a training. The procedure of them can be listed as.

- **Normalize the payload** by using function `mnorm.normalize_payload()` which come from `normalize.py`.

- **Training Period** for each epoch or iterations, errors are initialize with 0. Then we iteration over the samples, find the payload of sample, and find the associated validation sample. We send the sample into each model element under the function `predict_sample()` and recieve their results and do that again for the validation data sample. Finally we combined the result weighted by the model-level weight.

- **Back Propagation** Follwed by the training, in the same iteration, we find the derviative of error (MSE) ad we find the gradient of each model-level weight.

```python
   loss_prime = self.loss_prime(result_input, result)
   material_weight_error = np.dot(predicted_mc, loss_prime)

```

For the element-level weight, we pass it to the `back_propagation` function. To make the element-level learnin from the previoused experincanced m

&copy; 2024, Prince of Songkla University under Inteligent Automation Engineering Center
