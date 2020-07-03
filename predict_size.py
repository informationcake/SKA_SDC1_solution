import argparse

import numpy as np
from ska_sdc import Sdc1Scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from score_from_srl import cat_df_from_srl, load_truth_df, srl_gaul_df
from utils import SRL_CAT_COLS, SRL_COLS_TO_DROP, SRL_NUM_COLS


def baseline_size_acc(srl_path, truth_path, freq):
    truth_df = load_truth_df(truth_path)
    cat_df = cat_df_from_srl(srl_path)
    match_df = get_match_df(cat_df, truth_df, freq)

    rmse_bmaj = np.sqrt(mean_squared_error(match_df["b_maj_t"], match_df["b_maj"]))
    rmse_bmin = np.sqrt(mean_squared_error(match_df["b_min_t"], match_df["b_min"]))
    print("Baseline b_maj acc: {}".format(1 - rmse_bmaj))
    print("Baseline b_maj acc: {}".format(1 - rmse_bmin))


def score_with_predicted_sizes(srl_path, truth_path, gaul_path, freq):
    regressor_maj, b_maj_predict = predict_sizes(
        srl_path, truth_path, gaul_path, freq, size_col="b_maj_t"
    )
    regressor_min, b_min_predict = predict_sizes(
        srl_path, truth_path, gaul_path, freq, size_col="b_min_t"
    )
    cat_df = cat_df_from_srl(srl_path)
    cat_df.drop(["b_maj", "b_min"], axis=1)
    cat_df["b_maj"] = b_maj_predict
    cat_df["b_min"] = b_min_predict

    truth_df = load_truth_df(truth_path)
    scorer = Sdc1Scorer(cat_df, truth_df, freq)
    score = scorer.run(train=True, detail=True, mode=1)

    return score


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


# TODO: This method for development (tuning model) - remove when no longer needed
def develop_size_regressor(srl_path, truth_path, gaul_path, freq, size_col="b_maj_t"):
    """
    Given a PyBDSF source list, create an SDC1 catalogue, obtain the
    match catalogue from the score pipeline, and use the truth catalogue's size
    values, together with the source list properties, to build a model
    that can predict the size for unseen samples.

    This method is aimed at model development, splitting the training catalogue
    into train/validation catalogues and evaluating the precision of the model.
    Use this for trying different models/tuning hyperparameters.

    Args:
        srl_path (`str`): Path to source list (.srl file)
        truth_path (`str`): Path to training truth catalogue
        gaul_path (`str`): Path to gaussian list (.gaul file)
        freq (`int`): Image frequency band (560, 1400 or 9200 MHz)
        column (`str`): b_maj_t or b_min_t
    """
    # Get match_df:
    cat_df = cat_df_from_srl(srl_path)
    truth_df = load_truth_df(truth_path)
    match_df = get_match_df(cat_df, truth_df, freq)

    # Use the Source_id / id columns as the DataFrame index for easy cross-mapping
    srl_df = srl_gaul_df(gaul_path, srl_path)
    match_df = match_df.set_index("id")
    srl_df = srl_df.set_index("Source_id")

    # Set the true size ID for the source list, and the baseline comparison
    srl_df[size_col] = match_df[size_col]

    srl_df = prep_srl_df(srl_df)

    # Divide the dataset into train and validation subsets
    dev_srl_df = srl_df.iloc[::2, :]
    val_srl_df = srl_df.iloc[1::2, :]

    dev_y = dev_srl_df[size_col].values
    val_y = val_srl_df[size_col].values

    dev_x = dev_srl_df[SRL_CAT_COLS + SRL_NUM_COLS]
    val_x = val_srl_df[SRL_CAT_COLS + SRL_NUM_COLS]

    regressor, pred_val_y = run_rfr(dev_x, dev_y, val_x)

    rmse = np.sqrt(mean_squared_error(val_y, pred_val_y))
    print("Model accuracy {}".format(1 - rmse))


def predict_sizes(srl_path, truth_path, gaul_path, freq, size_col="b_maj_t"):
    """
    Given a PyBDSF source list, create an SDC1 catalogue, obtain the
    match catalogue from the score pipeline, and use the truth catalogue's size
    values, together with the source list properties, to build a model
    that can predict the size for unseen samples.

    Args:
        srl_path (`str`): Path to source list (.srl file)
        truth_path (`str`): Path to training truth catalogue
        gaul_path (`str`): Path to gaussian list (.gaul file)
        freq (`int`): Image frequency band (560, 1400 or 9200 MHz)
        column (`str`): b_maj_t or b_min_t
    """
    # Get match_df:
    cat_df = cat_df_from_srl(srl_path)
    truth_df = load_truth_df(truth_path)
    match_df = get_match_df(cat_df, truth_df, freq)

    # Use the Source_id / id columns as the DataFrame index for cross-mapping
    srl_df = srl_gaul_df(gaul_path, srl_path)
    match_df = match_df.set_index("id")
    srl_df_full = srl_df.copy()
    srl_df = srl_df.set_index("Source_id")

    # Set the true size ID for the source list, drop NaN values (unmatched sources)
    srl_df[size_col] = match_df[size_col]

    srl_df = prep_srl_df(srl_df)
    srl_df_full = prep_srl_df(srl_df_full)

    # Train full model:
    train_y = srl_df[size_col].values
    train_x = srl_df[SRL_CAT_COLS + SRL_NUM_COLS]
    test_x = srl_df_full[SRL_CAT_COLS + SRL_NUM_COLS]

    regressor, pred_test_y = run_rfr(train_x, train_y, test_x)

    return regressor, pred_test_y


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


def run_rfr(train_x, train_y, test_x):
    """
    Train RandomForestRegressor on train_x, train_y; predict class for test_x.

    Args:
        train_x (:obj:`pandas.DataFrame`): Training data
        train_y (:obj:`numpy.array`): Training set truth values
        test_x: (:obj:`pandas.DataFrame`): Test data
    """
    classifier = RandomForestRegressor(random_state=0)
    classifier.fit(train_x, train_y)
    pred_test_y = classifier.predict(test_x)
    return classifier, pred_test_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", help="Path to source list (.srl) output by PyBDSF", type=str
    )
    parser.add_argument("-t", help="Path to truth training catalogue", type=str)
    parser.add_argument(
        "-f", help="Image frequency band (560||1400||9200, MHz)", default=1400, type=int
    )
    parser.add_argument(
        "-g", help="Path to gaussian list (.gaul) output by PyBDSF", type=str
    )
    args = parser.parse_args()

    score_with_predicted_sizes(args.s, args.t, args.g, args.f)
    # develop_size_regressor(args.s, args.t, args.g, args.f, size_col="b_min_t")
    # baseline_size_acc(args.s, args.t, args.f)
