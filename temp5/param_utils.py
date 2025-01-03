# param_utils.py
import itertools

def create_parameter_combinations(selected_models):
    """
    Generates all parameter combinations for selected models.
    :param selected_models: List of tuples of model type and parameters.
    :return: List of model types and parameter combinations.
    """
    all_model_params = []
    for model_type, params in selected_models:
        keys, values = zip(*params.items())
        # Ensure values are iterable
        values = [_ensure_iterable(value) for value in values]
        all_params_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        all_model_params.extend([(model_type, param_set) for param_set in all_params_combinations])
    return all_model_params

def _ensure_iterable(value):
    """
    If a value is not an iterable, it will be returned inside a list.
    """
    if isinstance(value, str) or not hasattr(value, '__iter__'):
        return [value]
    return value