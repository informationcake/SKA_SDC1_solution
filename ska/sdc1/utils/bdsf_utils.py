import pandas as pd

from ska.sdc1.utils.columns import GAUL_COLUMNS, SRL_COLUMNS


def srl_as_df(srl_path):
    """
    Load the source list output by PyBDSF as a pd.DataFrame

    Args:
        srl_path (`str`): Path to source list (.srl file)
    """
    srl_df = pd.read_csv(
        srl_path, skiprows=6, names=SRL_COLUMNS, delim_whitespace=True,
    )
    return srl_df


def gaul_as_df(gaul_path):
    """
    Load the Gaussian list output by PyBDSF as a pd.DataFrame

    Args:
        gaul_path (`str`): Path to Gaussian list (.gaul file)
    """
    gaul_df = pd.read_csv(
        gaul_path, skiprows=6, names=GAUL_COLUMNS, delim_whitespace=True,
    )
    return gaul_df


def score_from_srl(srl_path, truth_path, freq, verbose=False):
    """
    Given source list output by PyBDSF and training truth catalogue,
    calculate the official score for the sources identified in the srl.

    Args:
        srl_path (`str`): Path to source list (.srl file)
        truth_path (`str`): Path to training truth catalogue
        freq (`int`): Image frequency band (560, 1400 or 9200 MHz)
        verbose (`bool`): True to print out size ratio info
    """
    truth_df = load_truth_df(truth_path)

    # Predict size ID and correct the Maj and Min values:
    cat_df = cat_df_from_srl(srl_path)

    scorer = Sdc1Scorer(cat_df, truth_df, freq)
    score = scorer.run(train=True, detail=True, mode=1)

    return score


def load_truth_df(truth_path):
    """
    Load the training area truth catalogue.
    Expected to be in the format as provided on the SDC1 website.

    Args:
        truth_path (`str`): Path to training truth catalogue
    """
    truth_df = pd.read_csv(
        truth_path,
        skiprows=18,
        names=CAT_COLUMNS,
        usecols=range(12),
        delim_whitespace=True,
    )
    return truth_df
