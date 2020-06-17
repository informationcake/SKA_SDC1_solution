import argparse

import pandas as pd
from ska_sdc import Sdc1Scorer

# Catalogue columns for SDC1 submissions
CAT_COLUMNS = [
    "id",
    "ra_core",
    "dec_core",
    "ra_cent",
    "dec_cent",
    "flux",
    "core_frac",
    "b_maj",
    "b_min",
    "pa",
    "size",
    "class",
]

# Columns output by PyBDSF
SRL_COLUMNS = [
    "Source_id",
    "Isl_id",
    "RA",
    "E_RA",
    "DEC",
    "E_DEC",
    "Total_flux",
    "E_Total_flux",
    "Peak_flux",
    "E_Peak_flux",
    "RA_max",
    "E_RA_max",
    "DEC_max",
    "E_DEC_max",
    "Maj",
    "E_Maj",
    "Min",
    "E_Min",
    "PA",
    "E_PA",
    "Maj_img_plane",
    "E_Maj_img_plane",
    "Min_img_plane",
    "E_Min_img_plane",
    "PA_img_plane",
    "E_PA_img_plane",
    "DC_Maj",
    "E_DC_Maj",
    "DC_Min",
    "E_DC_Min",
    "DC_PA",
    "E_DC_PA",
    "DC_Maj_img_plane",
    "E_DC_Maj_img_plane",
    "DC_Min_img_plane",
    "E_DC_Min_img_plane",
    "DC_PA_img_plane",
    "E_DC_PA_img_plane",
    "Isl_Total_flux",
    "E_Isl_Total_flux",
    "Isl_rms",
    "Isl_mean",
    "Resid_Isl_rms",
    "Resid_Isl_mean",
    "S_Code",
]


def score_from_srl(srl_path, truth_path, freq):
    """
    Given source list output by PyBDSF and training truth catalogue,
    calculate the official score for the sources identified in the srl.

    Args:
        srl_path (`str`): Path to source list (.srl file)
        truth_path (`str`): Path to training truth catalogue
        freq (`int`): Image frequency band (560, 1400 or 9200 MHz)
    """
    cat_df = cat_df_from_srl(srl_path)
    truth_df = load_truth_df(truth_path)

    scorer = Sdc1Scorer(cat_df, truth_df, freq)
    score = scorer.run(train=True, detail=True, mode=1)

    return score


def cat_df_from_srl(srl_path):
    """
    Load the source list output by PyBDSF and create a catalogue DataFrame of the
    form required for SDC1.

    Args:
        srl_path (`str`): Path to source list (.srl file)
    """
    srl_df = pd.read_csv(
        srl_path, skiprows=6, names=SRL_COLUMNS, delim_whitespace=True,
    )
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

    score = score_from_srl(args.s, args.t, args.f)
    print("Score was {}".format(score.value))
    print("Number of detections {}".format(score.n_det))
    print("Number of matches {}".format(score.n_match))
    print("Number of matches that were too far from truth {}".format(score.n_bad))
    print("Number of false detections {}".format(score.n_false))
    print("Score for all matches {}".format(score.score_det))
    print("Accuracy percentage {}".format(score.acc_pc))
