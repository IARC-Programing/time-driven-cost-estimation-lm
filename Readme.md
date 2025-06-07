---
license: mit
---

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

### Model-Element

For the element-level weight, we pass it to the `back_propagate` function. To make the element-level learning from the previous iteration error. It adjust their weight and biases using gradient descent-liked algorithm as same as the model-level weight and bias. After the `back_propagate` function is called (in file `network.py`) it will forward into the layer inside the model element (`material_fc_layer.py`,`capital_fc_layer.py`, and `employee_fc_layer.py`).

The model is not fully constructed is file `tdce_model.py` because it depends on the input. We will construt it before we use.

## Experiment

On the Artifical Neural Network-Like for Manufacturing Cost Estimation Project, we evaluate the model with the experiment inside the folder `experiment`. We using 4 set of datasets, 3 from simulation and 1 from the actual dataset. The actual dataset is prohibit to display and open. Three simulation datasets consist of **Simple Dataset**, **Complicated Dataset** (High Variation), and **Extended Random Dataset** (High data dimension).

### Data Gathering

The three set of data (Simple,Complicated,Actual) is created by upload the based data into IAEC Manufacturing ERP System "E-Manufac" and retrieve the data using REST API. The Code of data retrival can be located at `functions/data_extractor` directory, while the `emanufac_tdabc_extractor_class.py` is the main code. After the data was retrieved, it saved into the CSV file and the code in `adjust_data.py` was employed to intial preprocessing.

For testing or validate our model, without access to the E-Manufac system. The simulation dataset can be downloaded from HuggingFace Dataset (the actual dataset will not provided).

- [Simple Dataset](https://huggingface.co/datasets/theethawats98/tdce-example-simple-dataset)
- [Complicated Dataset](https://huggingface.co/datasets/theethawats98/tdce-example-complicated-dataset)
- [Extended Random Dataset](https://huggingface.co/datasets/theethawats98/tdce-example-extended-random)

Go to tab **Files and Version** and download their files. File names are initial with the name `generated`. If you want to use our experiment script, please create the directory named `dataset` and then create folder for each set, and put all files of each set insided.

### Construct Experiment

On our research project, we create the validation experiment as a script, which is located in `experiment/experiment_script.py` the main function name is `run_experiment`.

This require the setting parameters like `data_directory`, `epoch` amount, `folder_name` to keep the result, `round_number` to specify the amount of training replication for preventing the error, `learning_rate_list` for learning rate which used for testing, we can use multiple learning rate at one time you run this script.

We also provide the experimented scenario to choose and test which boolean parameters.

- **Outlier Removal** - Set the `remove_outlier_qtr` to `True` for enable, with using iterquartile range, you can specify the `outlier_index` to remove (the default is `1.5` which refer to 1.5IQR will reject as outliers).

- **Early Stopping** - You can preventing the high validation error and improve model regularization using the early stopping, by enable `early_stopping` as `True` and also can modify the `patience_round` for the model.

- **Data Augmentation** - Enable or Disable Class balacing for the input material by enable the flag `augmentation`.

The script will load the data from the defined directory and use them to perform the experiment.

---

&copy; 2024, Prince of Songkla University under the Inteligent Automation Engineering Center, Faculty of Engineering.
