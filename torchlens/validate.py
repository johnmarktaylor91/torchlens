# TODO: this should only have top-level validation functions; the rest should be methods of ModelHistory
import pandas as pd


def validate_batch_of_models_and_inputs() -> pd.DataFrame:
    """Given multiple models and several inputs for each, validates the saved activations for all of them
    and returns a Pandas dataframe summarizing the validation results.

    Returns:

    """
    raise NotImplementedError


def validate_multiple_inputs_for_model() -> pd.DataFrame:
    """Given a model and multiple inputs, validates the saved activations for all of them and returns a Pandas
    dataframe summarizing the validation results.

    Returns: Pandas dataframe with summary of validation results
    """
    raise NotImplementedError
