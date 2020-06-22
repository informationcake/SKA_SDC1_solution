import argparse

import pandas as pd
from ska_sdc import Sdc1Scorer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from constants import gauss_to_expo, gauss_to_las
from utils import CAT_COLUMNS, SRL_CAT_COLS, SRL_COLS_TO_DROP, SRL_COLUMNS, SRL_NUM_COLS


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
    cat_df = predict_sizes_from_srl(srl_path, truth_path, freq)

    scorer = Sdc1Scorer(cat_df, truth_df, freq)
    score = scorer.run(train=True, detail=True, mode=1)

    if verbose:
        print_size_ratios(score.match_df)

    return score


def predict_sizes_from_srl(srl_path, truth_path, freq):
    """
    Given a PyBDSF source list, create an SDC1 catalogue, obtain the
    match catalogue from the score pipeline, and use the truth catalogue's size
    class values, together with the source list properties, to build a model
    that can predict the size class for unseen samples.

    Refine the measurements of b_maj and b_min based on these classes, and
    return a catalogue DataFrame which can be scored.

    Args:
        srl_path (`str`): Path to source list (.srl file)
        truth_path (`str`): Path to training truth catalogue
        freq (`int`): Image frequency band (560, 1400 or 9200 MHz)
    """
    # Get match_df:
    cat_df = cat_df_from_srl(srl_path)
    truth_df = load_truth_df(truth_path)
    match_df = get_match_df(cat_df, truth_df, freq)

    # Use the Source_id / id columns as the DataFrame index for cross-mapping
    srl_df = srl_as_df(srl_path)
    match_df = match_df.set_index("id")
    srl_df_full = srl_df.copy()
    srl_df = srl_df.set_index("Source_id")

    # Set the true size ID for the source list, drop NaN values (unmatched sources)
    srl_df["size_id_t"] = match_df["size_id_t"]

    srl_df = prep_srl_df(srl_df)
    srl_df_full = prep_srl_df(srl_df_full)

    # size_id_t already encoded; convert to int
    srl_df["size_id_t"] = srl_df["size_id_t"].values.astype("int")

    # Train full model:
    train_y = srl_df["size_id_t"].values
    train_x = srl_df[SRL_CAT_COLS + SRL_NUM_COLS]
    test_x = srl_df_full[SRL_CAT_COLS + SRL_NUM_COLS]

    classifier, pred_test_y = run_rfc(train_x, train_y, test_x)

    # Update cat_df size column:
    cat_df["size"] = pred_test_y

    # Size class 1: LAS. Convert Gauss -> LAS.
    cat_df.loc[cat_df["size"] == 1, "b_maj"] *= gauss_to_las
    cat_df.loc[cat_df["size"] == 1, "b_min"] *= gauss_to_las

    # Size class 3: Expo. Convert Gauss -> Expo.
    cat_df.loc[cat_df["size"] == 3, "b_maj"] *= gauss_to_expo
    cat_df.loc[cat_df["size"] == 3, "b_min"] *= gauss_to_expo

    # Size class 2: Gauss. These tend to be overestimated since the true source
    # excludes the beam size convolution.
    # TODO: This will make the sizes more accurate, but is obviously artificial
    # cat_df.loc[cat_df["size"] == 2, "b_maj"] *= 0.49
    # cat_df.loc[cat_df["size"] == 2, "b_min"] *= 0.20

    return cat_df


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


def get_match_df(cat_df, truth_df, freq):
    """
    Get the matched sources DataFrame for the passed catalogue DataFrame. Uses the
    scorer to yield the match catalogue.

    Args:
        cat_df (:obj:`pandas.DataFrame`): Catalogue DataFrame to obtain match_df for
        truth_df (:obj:`pandas.DataFrame`): Truth catalogue DataFrame
        freq: (`int`): Frequency band
    """
    scorer = Sdc1Scorer(cat_df, truth_df, freq)
    score = scorer.run(train=True, detail=True, mode=1)

    return score.match_df


def cat_df_from_srl(srl_path):
    """
    Load the source list output by PyBDSF and create a catalogue DataFrame of the
    form required for SDC1.

    Args:
        srl_path (`str`): Path to source list (.srl file)
    """
    srl_df = srl_as_df(srl_path)

    # Instantiate catalogue DataFrame
    cat_df = pd.DataFrame()

    # Source ID
    cat_df["id"] = srl_df["Source_id"]

    # Positions (correct RA degeneracy to be zero)
    cat_df["ra_core"] = srl_df["RA_max"]
    cat_df.loc[cat_df["ra_core"] > 180.0, "ra_core"] -= 360.0
    cat_df["dec_core"] = srl_df["DEC_max"]

    cat_df["ra_cent"] = srl_df["RA"]
    cat_df.loc[cat_df["ra_cent"] > 180.0, "ra_cent"] -= 360.0
    cat_df["dec_cent"] = srl_df["DEC"]

    # Flux and core fraction
    cat_df["flux"] = srl_df["Total_flux"]
    cat_df["core_frac"] = (srl_df["Peak_flux"] - srl_df["Total_flux"]).abs()

    # Bmaj, Bmin (convert deg -> arcsec) and PA
    # TODO: Source list outputs FWHM as major/minor axis measures, but this should
    # differ according to the size class
    cat_df["b_maj"] = srl_df["Maj"] * 3600
    cat_df["b_min"] = srl_df["Min"] * 3600
    cat_df["pa"] = srl_df["PA"]

    # Size class
    # TODO: To be determined (possibly from the S_Code column of src_df)
    cat_df["size"] = 1

    # Class
    # TODO: To be predicted using classifier
    cat_df["class"] = 1
    return cat_df


def prep_srl_df(srl_df):
    """
    Prep the source list DataFrame for model generation/prediction;
    drop NaNs, encode categorical columns, cast numerical columns as floats.

    Args:
        cat_df (:obj:`pandas.DataFrame`): Catalogue DataFrame to obtain match_df for
        truth_df (:obj:`pandas.DataFrame`): Truth catalogue DataFrame
        freq: (`int`): Frequency band
    """
    srl_df = srl_df.dropna()

    # Columns to drop before model building:
    srl_df = srl_df.drop(SRL_COLS_TO_DROP, axis=1)

    # Encode the categorical columns:
    for col in SRL_CAT_COLS:
        lbl = LabelEncoder()
        lbl.fit(list(srl_df[col].values.astype("str")))
        srl_df[col] = lbl.transform(list(srl_df[col].values.astype("str")))

    # Define the numerical columns, and cast as floats
    for col in SRL_NUM_COLS:
        srl_df[col] = srl_df[col].astype(float)

    return srl_df


def run_rfc(train_x, train_y, test_x):
    """
    Train RandomForestClassifier on train_x, train_y; predict class for test_x.

    Args:
        train_x (:obj:`pandas.DataFrame`): Training data
        train_y (:obj:`numpy.array`): Training set truth values
        test_x: (:obj:`pandas.DataFrame`): Test data
    """
    classifier = RandomForestClassifier(random_state=0)
    classifier.fit(train_x, train_y)
    pred_test_y = classifier.predict(test_x)
    return classifier, pred_test_y


def print_size_ratios(match_df):
    """
    Debug method: Output to console the ratios of true b_maj, b_min sizes to those
    calculated.

    Args:
        match_df (:obj:`pandas.DataFrame`): match_df produced by Sdc1Scorer to compare
            true size values with submitted values.
    """
    b_maj_ratio_s = match_df["b_maj_t"] / match_df["b_maj"]
    b_min_ratio_s = match_df["b_min_t"] / match_df["b_min"]
    print("After applying size corrections, mean size ratios are: ")
    for i, name in enumerate(["LAS", "Gauss", "Expo"], 1):
        print("Class {}: {}".format(i, name))
        print(
            "b_maj_t / b_maj: {}".format(b_maj_ratio_s[match_df["size_id"] == i].mean())
        )
        print(
            "b_min_t / b_min: {}".format(b_min_ratio_s[match_df["size_id"] == i].mean())
        )
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", help="Path to source list (.srl) output by PyBDSF", type=str
    )
    parser.add_argument("-t", help="Path to truth training catalogue", type=str)
    parser.add_argument(
        "-f", help="Image frequency band (560||1400||9200, MHz)", default=1400, type=int
    )
    args = parser.parse_args()

    score = score_from_srl(args.s, args.t, args.f, verbose=True)
    print("Score was {}".format(score.value))
    print("Number of detections {}".format(score.n_det))
    print("Number of matches {}".format(score.n_match))
    print("Number of matches that were too far from truth {}".format(score.n_bad))
    print("Number of false detections {}".format(score.n_false))
    print("Score for all matches {}".format(score.score_det))
    print("Accuracy percentage {}".format(score.acc_pc))
