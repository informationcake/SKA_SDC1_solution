# Python3.6. Written by Alex Clarke.
# Read in catalogue data and predict classes, then run scorer.

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
import umap
import umap.plot
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
from datashader.transfer_functions import shade, stack
from datashader.utils import export_image
from datashader.colors import *

from ska_sdc import Sdc1Scorer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def cat_df_from_srl(srl_df, use_pred_class=False):
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
    # Source list outputs FWHM as major/minor axis measures
    cat_df["b_maj"] = srl_df["Maj"] * 3600
    cat_df["b_min"] = srl_df["Min"] * 3600
    cat_df["pa"] = srl_df["PA"]
    # Size class
    cat_df["size"] = 2
    # Set default class as 1 (a guess)
    cat_df["class"] = 1
    # set class derived from a model
    if use_pred_class==True:
        # add int classes
        cat_df['class'] = srl_df['class_pred']
        cat_df.loc[cat_df['class'] == 'SSAGN', 'class'] = 1
        cat_df.loc[cat_df['class'] == 'FSAGN', 'class'] = 2
        cat_df.loc[cat_df['class'] == 'SFG', 'class'] = 3
        cat_df['class'] = cat_df['class'].astype(int)
    return cat_df



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def prepare_data(cat_srl_fits, cat_gaul_fits, cat_truth, freq, train=False):
    # load srl and gaul fits catalogues
    df_gaul = Table.read(cat_gaul_fits).to_pandas()
    df_srl = Table.read(cat_srl_fits).to_pandas()
    # add number of gaussians per source
    df_srl['n_gaussians'] = df_gaul.Source_id.value_counts()
    # get catalogue column names in format for scoring
    df_cat_scoring = cat_df_from_srl(df_srl)
    # load truth catalogue
    truth_columns = ["id", "ra_core", "dec_core", "ra_cent", "dec_cent", "flux", "core_frac", "b_maj", "b_min", "pa", "size", "class"]
    df_truth = pd.read_csv(cat_truth, skiprows=18, names=truth_columns, usecols=range(12), delim_whitespace=True)
    # run scoring code, get matched_df output. Truth cat has NaNs in for some reason?
    scorer = Sdc1Scorer(df_cat_scoring, df_truth.dropna(), freq)
    score = scorer.run(train=train, detail=True, mode=1)
    # join on catalogue, add relevant truth columns
    df_match = df_srl.join(score.match_df.set_index('id')[['id_t', 'class_t', 'flux_t', 'core_frac_t']], how='left')
    # Need integer, string and categorical class column types for later use
    df_match.rename(columns={'class_t': 'class_int'}, inplace=True)
    df_match['class_str'] = df_match['class_int']
    df_match.loc[df_match.class_str == 1, 'class_str'] = 'SSAGN'
    df_match.loc[df_match.class_str == 2, 'class_str'] = 'FSAGN'
    df_match.loc[df_match.class_str == 3, 'class_str'] = 'SFG'
    df_match['class_cat'] = df_match['class_str'].astype('category')
    # return input df, df with truth matches (no false det), scorer class, and truth df
    return df_match, score, df_truth



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def join_560_1400(df_560, df_1400, tolerance=1/3600):
    # use astropy to do a positional cross match
    c_560 = SkyCoord(df_560['RA'], df_560['DEC'], unit=(u.deg,u.deg))
    c_1400 = SkyCoord(df_1400['RA'], df_1400['DEC'], unit=(u.deg,u.deg))
    idx, sep2d, dist3d = c_560.match_to_catalog_sky(c_1400, 1)
    #inds = np.nonzero( (sep2d < tolerance*u.deg) )[0]
    df = df_560.copy()
    df['idx1400match'] = idx
    df['idx1400_sep2d'] = sep2d
    df = df.reset_index().join(df_1400.iloc[idx].reset_index(), lsuffix='_560', rsuffix='_1400')
    df = df[df.idx1400_sep2d < tolerance] # remove far away matches
    # remove duplicate matches where same 1400 MHz source appears
    df = df.sort_values('idx1400_sep2d').drop_duplicates(subset=['RA_1400', 'DEC_1400'], keep='first')
    # keep class labels consistent, using 560 MHz labels (same as other freq labels)
    df['class_str'] = df['class_str_560']
    df['class_cat'] = df['class_cat_560']
    df['class_int'] = df['class_int_560']
    return df



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def add_spectral_index(df_in, cat_cube_srl, tolerance=1/3600):
    df = df_in.copy()
    cat_cube = Table.read(cat_cube_srl)
    df_cat_cube = cat_cube.to_pandas()
    # use astropy to do a positional cross match
    c_df = SkyCoord(df['RA'], df['DEC'], unit=(u.deg,u.deg))
    c_cube = SkyCoord(cat_cube['RA'], cat_cube['DEC'], unit=(u.deg,u.deg))
    idx, sep2d, dist3d = c_df.match_to_catalog_sky(c_cube, 1)
    df['idx1400match'] = idx
    df['idx1400_sep2d'] = sep2d
    df = df.reset_index().join(df_cat_cube.iloc[idx].reset_index()[['RA', 'DEC', 'Spec_Indx', 'Total_flux_ch1', 'Total_flux_ch2']], rsuffix='_cube')
    # remove duplicate matches where same 1400 MHz source appears
    df = df[df.idx1400_sep2d < tolerance]
    # ensure there are not any duplicate 1400 MHz sources
    df = df.sort_values('idx1400_sep2d').drop_duplicates(subset=['RA_cube', 'DEC_cube'], keep='first')
    return df



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def get_truth_si_cat(cat560_truth, cat1400_truth):
    truth_columns = ["id", "ra_core", "dec_core", "ra_cent", "dec_cent", "flux", "core_frac", "b_maj", "b_min", "pa", "size", "class"]
    df560_t = pd.read_csv(cat560_truth, skiprows=18, names=truth_columns, usecols=range(12), delim_whitespace=True)
    df1400_t = pd.read_csv(cat1400_truth, skiprows=18, names=truth_columns, usecols=range(12), delim_whitespace=True)
    df = df560_t.set_index('id').join(df1400_t.set_index('id'), rsuffix='_1400', lsuffix='_560')
    df['si'] = np.log(df['flux_560']/df['flux_1400'])/np.log(560e6/1400e6)
    return df



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def add_spectral_index_truth(df_in, df_truth):
    df = df_in.copy()
    df = df.set_index('id_t').join(df_truth[['si', 'flux_560', 'flux_1400']])
    return df



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def plot_flux_hist(df, df_truth, freq):
    bins = 10 ** np.linspace(np.log10(1e-8), np.log10(10), 200)
    # truth cat
    plt.hist(df_truth[df_truth['class']==1].flux, bins=bins, histtype='step', color='C0', label='SS-AGN, truth')
    plt.hist(df_truth[df_truth['class']==2].flux, bins=bins, histtype='step', color='C1', label='FS-AGN, truth')
    plt.hist(df_truth[df_truth['class']==3].flux, bins=bins, histtype='step', color='C2', label='SFG, truth')
    # pybdsf cat
    plt.hist(df[df['class_int']==1].Total_flux, bins=bins, histtype='step', linestyle='--', color='C0', label='SS-AGN, found')
    plt.hist(df[df['class_int']==2].Total_flux, bins=bins, histtype='step', linestyle='--', color='C1', label='FS-AGN, found')
    plt.hist(df[df['class_int']==3].Total_flux, bins=bins, histtype='step', linestyle='--', color='C2', label='SFG, found')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(str(freq)+' MHz')
    plt.xlabel('Total flux, Jy')
    plt.ylabel('Number of sources')
    plt.legend(frameon=False)
    plt.minorticks_on()
    plt.show()



        # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def plot_flux(df, freq):
    df1=df[df.class_t==1] # SS-AGN
    df2=df[df.class_t==2] # FS-AGN
    df3=df[df.class_t==3] # SFG
    plt.scatter(df1.flux_t, df1.flux, s=0.5, marker='o', c='C0', label='SS-AGN')
    plt.scatter(df2.flux_t, df2.flux, s=0.5, marker='o', c='C1', label='FS-AGN')
    plt.scatter(df3.flux_t, df3.flux, s=0.5, marker='o', c='C2', label='SFG')
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=1, scalex=False, scaley=False)
    plt.xlabel('Truth total flux')
    plt.ylabel('Measured total flux')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(str(freq)+' MHz')
    plt.legend(frameon=False, markerscale=4)
    plt.show()



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def plot_any(df, freq, column='flux'):
    # plots any columns from the match_df returned by the scorer
    df1=df[df.class_t==1] # SS-AGN
    df2=df[df.class_t==2] # FS-AGN
    df3=df[df.class_t==3] # SFG
    plt.scatter(df1[column+'_t'], df1[column], s=0.5, marker='o', c='C0', label='SS-AGN')
    plt.scatter(df2[column+'_t'], df2[column], s=0.5, marker='o', c='C1', label='FS-AGN')
    plt.scatter(df3[column+'_t'], df3[column], s=0.5, marker='o', c='C2', label='SFG')
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=1, scalex=False, scaley=False)
    plt.xlabel('Truth, '+column)
    plt.ylabel('Measured, '+column)
    plt.xscale('log')
    plt.yscale('log')
    plt.title(str(freq)+' MHz')
    plt.legend(frameon=False, markerscale=4)
    plt.show()



        # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



# work in progress
def plot_si(df):
    df1=df[df.class_int==1] # SS-AGN
    df2=df[df.class_int==2] # FS-AGN
    df3=df[df.class_int==3] # SFG
    plt.scatter(df1.si, df1.Spectral_index, s=0.5, marker='o', c='C0', label='SS-AGN')
    plt.scatter(df2.si, df2.Spectral_index, s=0.5, marker='o', c='C1', label='FS-AGN')
    plt.scatter(df3.si, df3.Spectral_index, s=0.5, marker='o', c='C2', label='SFG')
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=1, scalex=False, scaley=False)
    plt.xlabel('Truth spectral index')
    plt.ylabel('Measured spectral index')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(frameon=False, markerscale=4)
    plt.show()



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def run_umap(df, features, min_dist=0.2, n_neighbors=50, low_memory=False, target_weight=0.5, sup=False):
    # can set low_memory=True if you see seg faults
    if sup==False:
        u_model = umap.UMAP(min_dist=min_dist, n_neighbors=n_neighbors, low_memory=False).fit(df[features])
    if sup==True:
        u_model = umap.UMAP(min_dist=min_dist, n_neighbors=n_neighbors, low_memory=False, target_weight=target_weight).fit(df[features], y=df.class_int)
    # write 2D coordinates back to dataframe
    u_df = pd.DataFrame(u_model.embedding_, columns=['umap_x', 'umap_y'], index=df.index)
    df = df.copy().join(u_df, how='left') # join UMAP projection to original df
    return u_model, df



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



# only used for quick initial tests, not used now
def plot_umap_internal(df, u_model):
    # use the umap plotting tool inside of umap. It can be quick, but is not as flexible
    umap.plot.points(u_model, labels=df.class_str, theme='fire', width=800, height=800)
    plt.show()



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def plot_umap_classes(df, colour_pred_classes=False, label='unknown', plotsize=500, how='eq_hist', background='white'):
    # create plot of 2D UMAP features coloured by class labels
    cvs = ds.Canvas(plot_width=plotsize, plot_height=plotsize) # always square
    # colour the classes by the label predicted by another classifier? e.g. random forest
    #if colour_pred_classes=True:
    #agg = cvs.points(df, 'umap_x', 'umap_y', ds.count_cat('class_pred'))
    # colour the classes by the truth label
    #if colour_pred_classes=False:
    agg = cvs.points(df, 'umap_x', 'umap_y', ds.count_cat('class_cat'))
    ckey = dict(SSAGN='cyan', FSAGN='yellow', SFG='red')
    img = tf.shade(agg, color_key=ckey, how=how)
    export_image(img, 'UMAP-'+label, fmt='.png', background=background)
    # generate figure with png created and append colourbar axis
    fig = plt.figure(figsize=(10,10)) # y axis larger to fit cbar in
    img = mpimg.imread('UMAP-'+label+'.png')
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
    # legend for class labels
    leg_1 = plt.Line2D((0,0),(0,0), color='cyan', marker='', linestyle='', label='SS-AGN')
    leg_2 = plt.Line2D((0,0),(0,0), color='yellow', marker='', linestyle='', label='FS-AGN')
    leg_3 = plt.Line2D((0,0),(0,0), color='red', marker='', linestyle='', label='SFG')
    leg = plt.legend([leg_1, leg_2, leg_3], ['SS-AGN', 'FS-AGN', 'SFG'], frameon=False)
    leg_texts = leg.get_texts()
    leg_texts[0].set_color('cyan')
    leg_texts[1].set_color('yellow')
    leg_texts[2].set_color('red')
    fig.tight_layout()
    fig.savefig('UMAP-'+label+'.pdf', bbox_inches='tight')
    plt.show()
    plt.close(fig)



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



# need to make plot function per feature to get colorbars and labels properly
def plot_umap_features(df, label='unknown', plotsize=500, how='eq_hist', background='white', feature='n_gaussians'):
    # create png
    cvs = ds.Canvas(plot_width=plotsize, plot_height=plotsize) # always square
    agg = cvs.points(df, 'umap_x', 'umap_y', ds.mean(feature)) # input df column to plot
    #ckey = dict(SSAGN='cyan', FSAGN='yellow', SFG='red')
    img = tf.shade(agg, cmap=(mpl.cm.YlOrBr), how=how)
    export_image(img, 'UMAP-'+label, fmt='.png', background=background)
    # generate figure with png created and append colourbar axis
    fig = plt.figure(figsize=(10,10)) # y axis larger to fit cbar in
    img = mpimg.imread('UMAP-'+label+'.png')
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
    fig.tight_layout()
    fig.savefig('UMAP-'+label+'.pdf', bbox_inches='tight')
    plt.show()
    plt.close(fig)



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



# Take input data, prepare for supervised ML or TSNE
def do_tt_split(data, feature_columns, train_percent=0.5, ttsplit=True):
    print(' Data is shape: {0}'.format(data.shape))
    #test/train split using sklearn function
    if ttsplit==True:
        all_features = data[[*feature_columns]]
        all_classes = data['class_str']
        features_train, features_test, classes_train, classes_test = train_test_split(all_features, all_classes, train_size=train_percent, random_state=0, stratify=all_classes)
        class_names = np.unique(all_classes)
        feature_names = list(all_features)
        #if verbose==True: print('feature names are: ', str(feature_names))
        return {'features_train':features_train, 'features_test':features_test, 'classes_train':classes_train, 'classes_test':classes_test, 'class_names':class_names, 'feature_names':feature_names} #return dictionary. data within dictionary are DataFrames.
    #no test/train split, just return data in format for e.g. tsne or clustering
    if ttsplit==False:
        all_features = data[[*feature_columns]] # don't drop rows without matches, since this feeds into scoring code
        all_classes = data.dropna()['class_str'] # drop rows with no matches (can only calculate f1 score from truth matches)
        class_names = np.unique(data.dropna()['class_str'])
        return {'all_features':all_features, 'all_classes':all_classes, 'class_names':class_names} #return dictionary
        # mis-match of data shapes here is OK, dealt with in metrics function



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def RF_fit(data, n_estimators, n_jobs=-1):
    print(' Fitting a random forest model to the data...')
    #Prepare classifier with hyper parameters
    rfc=RandomForestClassifier(n_jobs=n_jobs, n_estimators=n_estimators, random_state=0, class_weight='balanced')
    #Set up sklean pipeline to keep everything together
    pipeline = Pipeline([ ('classification', RandomForestClassifier(n_jobs=n_jobs, n_estimators=n_estimators, random_state=0, class_weight='balanced')) ])
    #Do the fit
    pipeline.fit(data['features_train'], data['classes_train'])
    return pipeline



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



# Classify using fitted Random Forest model
def RF_classify(pipeline, data, n_jobs=-1, ttsplit=True, proba=False):
    # pipeline is returned from RF_fit
    # data is a dictionary containing test features
    if ttsplit==True:
        print(' Classifying test sources using random forest model...')
        if proba==False:
            classes_pred = pipeline.predict(data['features_test'])
            return classes_pred
        if proba==True:
            classes_pred = pipeline.predict_proba(data['features_test'])
            return classes_pred
    if ttsplit==False: # must have used ttsplit=False option in do_tt_split, implying no test/train split
        print(' Classifying new sources using random forest model (not used in test/train split)...')
        if proba==False:
            classes_pred = pipeline.predict(data['all_features'])
            return classes_pred
        if proba==True:
            classes_pred = pipeline.predict_proba(data['all_features'])
            return classes_pred



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



# Return metrics to assess performance of the model
def metrics(data, classes_pred, ttsplit=True):
    if ttsplit==True:
        d = data['classes_test']
    if ttsplit==False: # must have used ttsplit=False option in prepare_data, implying no test/train split
        d = data['all_classes'].dropna()
    report=classification_report(d, classes_pred, target_names=np.unique(data['class_names']), digits=4)
    print(report)
    print(confusion_matrix(d, classes_pred, labels=data['class_names']))



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def show_score_stats(score):
    print("Score was {}".format(score.value))
    print("Number of detections {}".format(score.n_det))
    print("Number of matches {}".format(score.n_match))
    print("Number of matches too far from truth {}".format(score.n_bad))
    print("Number of false detections {}".format(score.n_false))
    print("Score for all matches {}".format(score.score_det))
    print("Accuracy percentage {}\n".format(score.acc_pc))



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



if __name__ == '__main__':

    # define catalogues, must be in current folder
    # 560 MHz
    cat560_truth = '560mhz_truthcat.csv'
    cat560_srl_train = '560mhz1000hours_PBCOR_trainarea.pybdsm.srl.FITS'
    cat560_gaul_train = '560mhz1000hours_PBCOR_trainarea.pybdsm.gaul.FITS'
    cat560_srl = '560mhz1000hours_PBCOR.pybdsm.srl.FITS'
    cat560_gaul = '560mhz1000hours_PBCOR.pybdsm.gaul.FITS'
    # 1400 MHz
    cat1400_truth = '1400mhz_truthcat.csv'
    cat1400_srl_train = '1400mhz1000hours_PBCOR_trainarea.pybdsm.srl.FITS'
    cat1400_gaul_train = '1400mhz1000hours_PBCOR_trainarea.pybdsm.gaul.FITS'
    cat1400_srl = '1400mhz1000hours_PBCOR.pybdsm.srl.FITS'
    cat1400_gaul = '1400mhz1000hours_PBCOR.pybdsm.gaul.FITS'
    #convolved
    cat1400conv_srl = '1400mhz1000hours_PBCOR_crop_convolved_regrid.pybdsm.srl.FITS'
    cat1400conv_gaul = '1400mhz1000hours_PBCOR_crop_convolved_regrid.pybdsm.gaul.FITS'
    # 9200 Mhz
    # TBD


    # Try to predict class column and run SDC1 scorer
    # -----------------------------------------------

    # 560 MHz

    # prepare catalogue from .FITS files, join on truth catalogue
    print('\n Score on 560 MHz data training area:')
    df_560_train, score_560_train, df_truth_560 = prepare_data(cat560_srl, cat560_gaul, cat560_truth, 560, train=True)
    # define feature columns
    feature_columns = ['Total_flux', 'Peak_flux', 'Maj', 'Min', 'n_gaussians']
    # create dictionary with test/train features and classes
    data_dict = do_tt_split(df_560_train.dropna(), feature_columns, train_percent=0.5, ttsplit=True)
    # train random forest model
    rf_pipeline = RF_fit(data_dict, 100)
    # classify test sources with random forest model
    classes_pred = RF_classify(rf_pipeline, data_dict, n_jobs=-1, ttsplit=True, proba=False)
    metrics(data_dict, classes_pred) # print f1 scores
    # now use the model to predict all class labels
    print(' Score on 560 MHz data without training area, assume class=1:')
    df_560, score_560, df_truth_560 = prepare_data(cat560_srl, cat560_gaul, cat560_truth, 560, train=False)
    show_score_stats(score_560)
    data_dict_all = do_tt_split(df_560, feature_columns, train_percent=0.5, ttsplit=False)
    classes_pred_all = RF_classify(rf_pipeline, data_dict_all, n_jobs=-1, ttsplit=False, proba=False)
    # add class labels to df and re-calculate score
    df_class_pred = pd.DataFrame(classes_pred_all, columns=['class_pred'])
    df_new = df_560.reset_index().join(df_class_pred)
    # get df in format for scoring code
    df_scorer = cat_df_from_srl(df_new, use_pred_class=True)
    # run scoring code, get matched_df output. Truth cat has NaNs in for some reason?
    scorer = Sdc1Scorer(df_scorer, df_truth_560.dropna(), 560)
    print(' Score on 560 MHz data without training area, with RF class prediction:')
    score = scorer.run(train=False, detail=True, mode=1)
    show_score_stats(score)
    # get precision recall f1-score metrics for sources with matches
    data_dict_all = do_tt_split(df_560.dropna(), feature_columns, train_percent=0.5, ttsplit=False)
    classes_pred_all = RF_classify(rf_pipeline, data_dict_all, n_jobs=-1, ttsplit=False, proba=False)
    metrics(data_dict_all, classes_pred_all, ttsplit=False) # print f1 scores


    # 1400 MHz

    # prepare catalogue from .FITS files, join on truth catalogue
    print('\n Score on 1400 MHz data training area:')
    df_1400_train, score_1400_train, df_truth_1400 = prepare_data(cat1400_srl, cat1400_gaul, cat1400_truth, 1400, train=True)
    # define feature columns
    feature_columns = ['Total_flux', 'Peak_flux', 'Maj', 'Min', 'n_gaussians']
    # create dictionary with test/train features and classes
    data_dict = do_tt_split(df_1400_train.dropna(), feature_columns, train_percent=0.5, ttsplit=True)
    # train random forest model
    rf_pipeline = RF_fit(data_dict, 100)
    # classify test sources with random forest model
    classes_pred = RF_classify(rf_pipeline, data_dict, n_jobs=-1, ttsplit=True, proba=False)
    metrics(data_dict, classes_pred) # print f1 scores
    # now use the model to predict all class labels
    print(' Score on 1400 MHz data without training area, assume class=1:')
    df_1400, score_1400, df_truth_560 = prepare_data(cat1400_srl, cat1400_gaul, cat1400_truth, 1400, train=False)
    show_score_stats(score_1400)
    data_dict_all = do_tt_split(df_1400, feature_columns, train_percent=0.5, ttsplit=False)
    classes_pred_all = RF_classify(rf_pipeline, data_dict_all, n_jobs=-1, ttsplit=False, proba=False)
    # add class labels to df and re-calculate score
    df_class_pred = pd.DataFrame(classes_pred_all, columns=['class_pred'])
    df_new = df_1400.reset_index().join(df_class_pred)
    # get df in format for scoring code
    df_scorer = cat_df_from_srl(df_new, use_pred_class=True)
    # run scoring code, get matched_df output. Truth cat has NaNs in for some reason?
    scorer = Sdc1Scorer(df_scorer, df_truth_1400.dropna(), 1400)
    print(' Score on 1400 MHz data without training area, with RF class prediction:')
    score = scorer.run(train=False, detail=True, mode=1)
    show_score_stats(score)
    # get precision recall f1-score metrics for sources with matches
    data_dict_all = do_tt_split(df_1400.dropna(), feature_columns, train_percent=0.5, ttsplit=False)
    classes_pred_all = RF_classify(rf_pipeline, data_dict_all, n_jobs=-1, ttsplit=False, proba=False)
    metrics(data_dict_all, classes_pred_all, ttsplit=False) # print f1 scores


    # 9200 MHz

    #TBD







    #
