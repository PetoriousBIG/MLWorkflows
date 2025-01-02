import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoLars
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import sys
import copy
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.stats import gaussian_kde
from sklearn.model_selection import cross_validate

def fit_estimators(estimators, preprocessor, training_data, training_target):

    trained_estimator_dict = {}
    for estimator_name, estimator in estimators.items():
        print(estimator_name)

        composite_estimator = \
            Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('estimator', estimator)
                ]
            )

        composite_estimator.fit(training_data, training_target.values.ravel())

        trained_estimator_dict[estimator_name] = composite_estimator

    return trained_estimator_dict

def cross_validation(preprocessor, estimators, data, target, scoring, splitter,
                     target_label, trained_estimators, boxplot=True, catplot=True, task=None, **kwargs):

    # perform model survey cross validation
    return_dict = model_survey_cross_validation(preprocessor, estimators, data, target, scoring,
                                                splitter, **kwargs)
    cv_scores_dict = return_dict['cv_scores_dict']

    model_survey_cross_val_and_analysis_helper(cv_scores_dict, target_label, trained_estimators, 
                                               data, target, splitter, boxplot=boxplot, 
                                               catplot=catplot, task=task)

def model_survey_cross_validation(preprocessor, estimators, data, target=None, scoring=None,
                                  splitter=5, return_indices=False, drop_cv_times=True):
    """

    :param preprocessor: instantiated scikit-learn preprocessing pipeline
    :param estimator_dict: dictionary of instantiated scikit-learn predictors
    :param train_cap_x_df:
    :param train_y_df:
    :param scoring: str, callable, list, tuple, or dict, default=None
    :param splitter:
        integer k - performs deterministic (not random) k-fold split
        train_test_split - pass in instantiated object
        LeaveOneOut - pass in instantiated object
        KFold - pass in instantiated object
    :param return_indices: bool, default=False
    :param drop_cv_times: if True cross validation fit and score times will be dropped from scores_dict
    :return:
    """

    cv_scores_dict = {'scoring': scoring}

    for estimator_name, estimator in estimators.items():

        composite_estimator = \
            Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('estimator', estimator)
                ]
            )

        scores_dict = cross_validate(
            estimator=composite_estimator,
            X=data,
            y=target.values.ravel(),
            groups=None,
            scoring=scoring,
            cv=splitter,
            n_jobs=None,
            verbose=0,
            fit_params=None,
            params=None,
            pre_dispatch='2*n_jobs',
            return_train_score=True,
            return_estimator=False,
            return_indices=return_indices,
            error_score=np.nan
        )

        if drop_cv_times:
            del scores_dict['score_time']
            del scores_dict['fit_time']

        cv_scores_dict[estimator_name] = scores_dict

    return_dict = {
        'cv_scores_dict': cv_scores_dict
    }

    return return_dict

def model_survey_cross_val_and_analysis_helper(cv_scores_dict, target_attr, trained_estimator_dict, 
                                               train_cap_x_df, train_y_df, splitter, histplot=False, 
                                               gs_survey_results_df=None, boxplot=False, catplot=False, 
                                               task=None, return_=False):

    # transform the cross validation results for analysis
    return_dict = cv_scores_dict_to_cv_scores_df(cv_scores_dict)
    cv_scores_analysis_df = return_dict['cv_scores_analysis_df']
    cv_scores_grouped_df = return_dict['grouped_df']

    # analysis
    cv_scores_analysis(cv_scores_analysis_df, splitter, target_attr, histplot=histplot,
                       gs_survey_results_df=gs_survey_results_df, boxplot=boxplot, catplot=catplot)

    if not (task == 'classification'):
        plot_pred_vs_actual_survey(trained_estimator_dict, train_cap_x_df, train_y_df, 'train')

    if return_:
        return {
            'cv_scores_grouped_df': cv_scores_grouped_df
        }
    else:
        return None

def cv_scores_dict_to_cv_scores_df(cv_scores_dict):

    return_dict = cross_val_evaluation(cv_scores_dict)
    df_row_dict_list = return_dict['df_row_dict_list']

    cv_scores_analysis_df = pd.DataFrame(df_row_dict_list)

    # TODO: there is a bug in cross_val_evaluation() that is causing duplicate rows in data frame
    # TODO: dedup for now
    cv_scores_analysis_df = cv_scores_analysis_df.drop_duplicates()
    # TODO: line above is unnecessary once bug is fixed

    grouped_df = cv_scores_analysis_df.groupby(['regressor_name', 'score_name_', 'score_type']).mean().reset_index()

    print('\n', grouped_df, '\n')

    min_score = cv_scores_analysis_df.score.min()
    max_score = cv_scores_analysis_df.score.max()

    return_dict = {
        'cv_scores_analysis_df': cv_scores_analysis_df,
        'min_score': min_score,
        'max_score': max_score,
        'grouped_df': grouped_df
    }

    return return_dict

def cross_val_evaluation(scores_dict):
    """
    Takes in a scores dict from a sklearn cross_validation() function and return a score_analysis_dict that can be
    used to analyze the cross validation.
    :param scores_dict:
        first key:value pair
            key = 'scoring', value = a list of scores (metrics) evaluated in sklearn cross_validate() function
        remaining key:value pair(s)
            key = estimator name, value = scores dictionary returned from sklearn cross_validate() function
    :return:
    """

    scoring_list = scores_dict['scoring']
    del scores_dict['scoring']

    max_score = -1 * np.inf
    min_score = np.inf
    df_row_dict_list = []
    for score_name in scoring_list:  # we evaluate a score_name across all the estimators in the survey

        for regressor_name, scoring_dict in scores_dict.items():  # iterate through the estimators

            for cv_score_name, scores in scoring_dict.items():  # iterate though an estimators cross_validate() scores

                score_type = 'test'
                if 'train' in cv_score_name:
                    score_type = 'train'

                if score_name in cv_score_name:  # once we iterate to the score_name we are working on do stuff

                    # get the list of scores
                    scores_list = scoring_dict[cv_score_name]

                    # make scores positive and remove 'neg' from score names if scores are negative
                    return_dict = remove_neg_from_score_name_and_make_neg_score_positive(scores_list, cv_score_name)
                    scores_list = return_dict['scores']
                    score_name_ = return_dict['score_name']

                    # get the min and max score from the scores_list
                    max_ = max(scores_list)
                    if max_ > max_score:
                        max_score = max_

                    min_ = min(scores_list)
                    if min_ < min_score:
                        min_score = min_

                    # save the scores list to cv_scores_analysis_dict
                    for score in scores_list:
                        df_row_dict_list.append(
                            {
                                'regressor_name': regressor_name,
                                'score_name_': score_name_,
                                'score': score,
                                'score_type': score_type
                            }
                        )

    return_dict = {
        'df_row_dict_list': df_row_dict_list,
        'min_score': min_score,
        'max_score': max_score
    }

    return return_dict

def cv_scores_analysis(score_analysis, splitter, target_attr=None, boxplot=True, catplot=True, histplot=False,
                       gs_survey_results_df=None):

    if isinstance(score_analysis, pd.DataFrame):
        analysis_df = score_analysis
    else:
        analysis_df = pd.DataFrame(score_analysis)

    for score_name_ in analysis_df.score_name_.unique():

        # get all estimators with scoring = score_name_
        temp_df = analysis_df.loc[analysis_df.score_name_ == score_name_, :]

        # get the mean score for each estimator by score type (train and test)
        plot_df = (temp_df[['regressor_name', 'score', 'score_type']].groupby(['regressor_name', 'score_type']).
                   score.mean().reset_index())

        hue_order = ['train', 'test']

        if catplot:
            sns.catplot(data=plot_df, x='regressor_name', y='score', hue='score_type', s=100, hue_order=hue_order)
            plt.xticks(rotation=90)
            plt.title(f'means of {splitter.get_n_splits()}-fold cross validation scores\n{score_name_}')
            if target_attr:
                plt.ylabel(f'{target_attr}')
            plt.grid()
            plt.show()

        if boxplot:
            sns.boxplot(data=temp_df, x='regressor_name', y='score', hue='score_type', showmeans=True,
                        meanprops={'marker': 'x', 'markerfacecolor': 'black', 'markeredgecolor': 'black',
                                   'markersize': '6'}, hue_order=hue_order)
            plt.xticks(rotation=90)
            plt.title(f'boxplot of {splitter.get_n_splits()}-fold cross validation scores\n{score_name_}; '
                      f'x marker = mean')
            if target_attr is not None:
                plt.ylabel(f'{target_attr}')
            if score_name_ == 'r2':
                plt.ylabel(f'r2')
            plt.grid()
            plt.show()

        if histplot:
            test_temp_df = temp_df.loc[temp_df.score_type == 'test', :]
            for regressor_name in test_temp_df.regressor_name.unique():

                prep_df = prep_df_for_hist_plot_util_1(test_temp_df, regressor_name, gs_survey_results_df)
                p_value = get_afd_p_value(prep_df)

                print('\n')
                ax = sns.histplot(data=prep_df, x='score', hue='type', common_norm=False, kde=True, bins=20,
                                  stat='density')
                sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
                plt.title(f'{regressor_name}\np_value: {p_value}')
                plt.xlabel(score_name_)
                plt.grid()
                plt.show()

def plot_pred_vs_actual_survey(trained_estimator_dict, cap_x_df, y_df, data_set_name):
    """

    :param trained_estimator_dict:
    :param cap_x_df:
    :param y_df:
    :param data_set_name: data set name (train, validation or test)
    :return:
    """

    for estimator_name, estimator in trained_estimator_dict.items():

        str_type = str(type(estimator))  # for statsmodels.regression - add bias term
        if 'statsmodels.regression' in str_type:
            cap_x_df = sm.add_constant(cap_x_df)

        pred_y_df = estimator.predict(cap_x_df)
        plot_title = f'estimator_name: {estimator_name}; data_set_name: {data_set_name}'
        plot_pred_vs_actual(pred_y_df, y_df, plot_title)

def remove_neg_from_score_name_and_make_neg_score_positive(scores, score_name=None):
    """

    :param score_name:
    :param scores:
    :return:
    """

    # if scores are negative then change the sign to positive - they are negative because scikit-learn follows a
    # convention where higher scores are better in optimization
    neg_score_flag = False
    if (scores <= 0).all():  # r2 sometimes returns small negative values
        neg_score_flag = True
        scores = -1 * scores

    # remove the words 'neg', 'train' and 'test' from score_name
    score_name = '_'.join([token for token in score_name.split('_') if 'neg' not in token])
    score_name = '_'.join([token for token in score_name.split('_') if 'train' not in token])
    score_name = '_'.join([token for token in score_name.split('_') if 'test' not in token])

    return_dict = {
        'scores': scores,
        'score_name': score_name,
        'neg_score_flag': neg_score_flag
    }

    return return_dict

def plot_pred_vs_actual(pred_y_df, train_y_df, plot_title):
    """

    :param pred_y_df:
    :param train_y_df:
    :param plot_title: include estimator name and data set name (train, validation or test)
    :return:
    """
    plt.scatter(train_y_df, pred_y_df)  # plot predicted y vs true y
    plt.plot(train_y_df, train_y_df, 'b')  # plot a line of slope 1 demonstrating perfect predictions
    plt.grid()
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.title(plot_title)
    plt.show()


        
    