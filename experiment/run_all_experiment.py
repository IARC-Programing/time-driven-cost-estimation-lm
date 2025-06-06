import batch_experiment_script as bes
import importlib
importlib.reload(bes)


def run_all_experiment():
    experiment_version = 1
    round_number = 3
    iteration = 100

    dataset_names = ['1-munchkin', '2-chinchilla',
                     '3-vicheanmas', '4-scottishfold']

    for dataset_name in dataset_names:
        bes.run_in_many_learning_rate(dataset_name=dataset_name,
                                      experiment_version=experiment_version,
                                      round_number=round_number,
                                      iteration=iteration)


run_all_experiment()
