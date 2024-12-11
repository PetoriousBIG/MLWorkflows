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


def get_afd_p_value(df, plot_null_distribution=False):
    """
    this code assumes a left tailed test
    :param df:
    :param plot_null_distribution:
    :return:
    """

    # get the kde of the shuffled result - the null distribution
    temp_df = df.loc[df.type == 'target_randomized', :]
    kde = gaussian_kde(temp_df['score'])

    # get R observed
    r_obs = df.loc[df.type == 'target_not_randomized', 'score'].values[0]

    # get the domain
    max_ = max(r_obs, temp_df['score'].max())
    min_ = min(r_obs, temp_df['score'].min())

    if plot_null_distribution:
        # plot the null distribution
        x = np.linspace(min_ - 1, max_ + 1, 1000)
        y = kde(x)
        plt.plot(x, y)
        plt.axvline(r_obs, c='r', linestyle='--', label='R observed')
        plt.grid()
        plt.show()

    # get the p-value - assumes left tailed test
    p_value = kde.integrate_box_1d(-np.inf, r_obs)

    return p_value


def prep_df_for_hist_plot_util_1(test_temp_df, regressor_name, gs_survey_results_df):

    prep_df_1 = test_temp_df.loc[test_temp_df.regressor_name == regressor_name, 'score'].to_frame()
    prep_df_1['type'] = 'target_randomized'

    prep_df_2 = gs_survey_results_df.loc[
        gs_survey_results_df.estimator_name == regressor_name, 'best_test_score'].to_frame(). \
        rename(columns={'best_test_score': 'score'})
    prep_df_2['type'] = 'target_not_randomized'

    prep_df = pd.concat([prep_df_1, prep_df_2], axis=0).reset_index(drop=True)

    return prep_df


def fix_negative_scores_and_name_utility(gs_cv_results_df, mean_train_score, mean_test_score, score):

    # make scores positive and remove 'neg' from score names if scores are negative
    return_dict = remove_neg_from_score_name_and_make_neg_score_positive(gs_cv_results_df[mean_train_score], score)
    gs_cv_results_df[mean_train_score] = return_dict['scores']
    score_ = return_dict['score_name']
    neg_score_flag = return_dict['neg_score_flag']

    mean_train_score_ = 'mean_train_' + score_

    return_dict = remove_neg_from_score_name_and_make_neg_score_positive(gs_cv_results_df[mean_test_score], score)
    gs_cv_results_df[mean_test_score] = return_dict['scores']
    score_ = return_dict['score_name']

    mean_test_score_ = 'mean_test_' + score_

    return_dict = {
        'gs_cv_results_df': gs_cv_results_df,
        'mean_train_score_': mean_train_score_,
        'mean_test_score_': mean_test_score_,
        'neg_score_flag': neg_score_flag
    }

    return return_dict


def prepare_data_for_flexibility_plot(gs_cv_results_df, mean_train_score, rank_test_score, mean_test_score,
                                      neg_score_flag, std_train_score, std_test_score):

    # determine proper sort
    if neg_score_flag:
        ascending = False  # smaller is better metric
    else:
        ascending = True  # bigger is better metric

    # sort by train score and label with index for plotting
    gs_cv_results_df = gs_cv_results_df.sort_values(mean_train_score, ascending=ascending).reset_index(drop=True). \
        reset_index()
    gs_cv_results_df = gs_cv_results_df[['index', rank_test_score, mean_train_score, std_train_score, mean_test_score,
                                         std_test_score]]

    return gs_cv_results_df


def flexibility_plot_util(gs_cv_results_df, mean_train_score, mean_train_score_, mean_test_score, mean_test_score_,
                          rank_test_score, an_estimator_name, score, std_train_score, std_test_score, error_bars=False):

    # plot train and test rmse
    jitter = 0
    sns.lineplot(x='index', y=mean_train_score, data=gs_cv_results_df, label=mean_train_score_, marker='o',
                 color='blue')
    if error_bars:
        for index in gs_cv_results_df.index:
            lower = gs_cv_results_df.loc[index, mean_train_score] - gs_cv_results_df.loc[index, std_train_score]
            upper = gs_cv_results_df.loc[index, mean_train_score] + gs_cv_results_df.loc[index, std_train_score]
            plt.plot([index-jitter, index-jitter], [lower, upper], color='blue')

    sns.lineplot(x='index', y=mean_test_score, data=gs_cv_results_df, label=mean_test_score_, marker='o', color='red')
    if error_bars:
        for index in gs_cv_results_df.index:
            lower = gs_cv_results_df.loc[index, mean_test_score] - gs_cv_results_df.loc[index, std_test_score]
            upper = gs_cv_results_df.loc[index, mean_test_score] + gs_cv_results_df.loc[index, std_test_score]
            plt.plot([index+jitter, index+jitter], [lower, upper], color='red')

    # draw a vertical line at the best index
    best_index = gs_cv_results_df.loc[gs_cv_results_df[rank_test_score] == 1, 'index'].values[0]
    plt.axvline(x=best_index, color='k', linestyle='--')

    # plot title and axis labels
    plt.title(f'{an_estimator_name} flexibility plot\nmin test error at best_index = {best_index}')
    plt.xlabel('flexibility')
    plt.ylabel('_'.join([token for token in score.split('_') if 'neg' not in token]))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # make index an integer on plot
    new_list = range(math.floor(min(gs_cv_results_df.index)), math.ceil(max(gs_cv_results_df.index)) + 1)
    if 10 <= len(new_list) < 100:
        skip = 10
    elif 100 <= len(new_list) < 1000:
        skip = 100
    else:
        skip = 500
    plt.xticks(np.arange(min(new_list), max(new_list) + 1, skip))
    plt.grid()
    plt.show()

    return best_index


def flexibility_plot_regr(gs_cv_results_df, an_estimator_name, score):

    # establish score names
    mean_train_score = 'mean_train_' + score
    std_train_score = 'std_train_' + score
    mean_test_score = 'mean_test_' + score
    std_test_score = 'std_test_' + score
    rank_test_score = 'rank_test_' + score

    # if scores are negative make scores positive and remove 'neg' from score name
    return_dict = fix_negative_scores_and_name_utility(gs_cv_results_df, mean_train_score, mean_test_score, score)
    gs_cv_results_df = return_dict['gs_cv_results_df']
    mean_train_score_ = return_dict['mean_train_score_']
    mean_test_score_ = return_dict['mean_test_score_']
    neg_score_flag = return_dict['neg_score_flag']

    # sort by train score and label with index for plotting
    gs_cv_results_df = prepare_data_for_flexibility_plot(gs_cv_results_df, mean_train_score, rank_test_score,
                                                         mean_test_score, neg_score_flag, std_train_score,
                                                         std_test_score)

    # make the plot
    best_index = flexibility_plot_util(gs_cv_results_df, mean_train_score, mean_train_score_, mean_test_score,
                                       mean_test_score_, rank_test_score, an_estimator_name, score, std_train_score,
                                       std_test_score)

    return gs_cv_results_df, best_index


def nominal_preprocessor(cap_x_df, y_df, nominal_attr, te_random_state=42):
    
    nominal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy='most_frequent')),
            ("target_encoder", TargetEncoder(
                categories='auto',
                target_type='continuous',
                smooth='auto',
                cv=5,
                shuffle=True,
                random_state=te_random_state
            )),
            ("scaler", StandardScaler())
        ]
    )
    preproc_cap_x_nom_attr = nominal_transformer.fit_transform(cap_x_df.loc[:, nominal_attr], y_df.values.ravel())

    preproc_cap_x_nom_attr_df = pd.DataFrame(
        data=preproc_cap_x_nom_attr,
        index=cap_x_df.index,
        columns=cap_x_df.loc[:, nominal_attr].columns
    )
    
    return preproc_cap_x_nom_attr_df


def numerical_preprocessor(cap_x_df, numerical_attr):
    
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler())
        ]
    )
    preproc_cap_x_num_attr = numerical_transformer.fit_transform(cap_x_df.loc[:, numerical_attr])

    preproc_cap_x_num_attr_df = pd.DataFrame(
        data=preproc_cap_x_num_attr,
        index=cap_x_df.index,
        columns=cap_x_df.loc[:, numerical_attr].columns
    )
    
    return preproc_cap_x_num_attr_df
    

def get_preproc_cap_x_df(cap_x_df, y_df, nominal_attr, numerical_attr, te_random_state=42):

    preproc_cap_x_num_attr_df = numerical_preprocessor(cap_x_df, numerical_attr)

    preproc_cap_x_nom_attr_df = nominal_preprocessor(cap_x_df, y_df, nominal_attr, te_random_state=te_random_state)

    preproc_cap_x_df = pd.concat([preproc_cap_x_num_attr_df, preproc_cap_x_nom_attr_df], axis=1)

    return preproc_cap_x_df


def get_model_specific_cv(model_type, cv_folds, l1_ratio_list=None):

    if model_type == 'lasso_lars_cv':

        model_cv = LassoLarsCV(
            fit_intercept=True,
            verbose=False,
            max_iter=500,
            precompute='auto',
            cv=cv_folds,
            max_n_alphas=1000,
            n_jobs=None,
            eps=np.finfo(float).eps,
            copy_X=True,
            positive=False
        )

    elif model_type == 'lasso_cv':

        model_cv = LassoCV(
            eps=0.001,
            n_alphas=100,
            alphas=None,
            fit_intercept=True,
            precompute='auto',
            max_iter=1000,
            tol=0.0001,
            copy_X=True,
            cv=cv_folds,
            verbose=False,
            n_jobs=None,
            positive=False,
            random_state=None,
            selection='cyclic'
        )

    elif model_type == 'elastic_net_cv':

        model_cv = ElasticNetCV(
            l1_ratio=l1_ratio_list,
            eps=0.001,
            n_alphas=100,
            alphas=None,
            fit_intercept=True,
            precompute='auto',
            max_iter=1000,
            tol=0.0001,
            cv=cv_folds,
            copy_X=True,
            verbose=0,
            n_jobs=None,
            positive=False,
            random_state=None,
            selection='cyclic'
        )

    else:
        sys.exit(f'model_type = {model_type} is not recognized')

    return model_cv


def plot_mse_path(fitted_model_cv, model_type, cv_folds):

    # plot the mse path vs alpha
    plt.semilogx(fitted_model_cv.cv_alphas_, fitted_model_cv.mse_path_, ":")

    # plot the average mse path vs alpha
    plt.plot(
        fitted_model_cv.cv_alphas_,
        fitted_model_cv.mse_path_.mean(axis=-1),
        color="black",
        label="Average across the folds",
        linewidth=2,
    )

    # draw a vertical line at the best alpha
    plt.axvline(fitted_model_cv.alpha_, linestyle="--", color="black", label="alpha CV")

    plt.xlabel(r"$\alpha$")
    plt.ylabel("mean square error")
    _ = plt.title(f"{model_type} mean square error on {cv_folds} folds")
    plt.legend()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.show()


def plot_coef_and_mse_path_util(fitted_model_cv, coefs, mse, model):

    ax = plt.gca()
    ax.plot(fitted_model_cv.cv_alphas_, coefs)
    ax.set_xscale("log")
    plt.axvline(x=fitted_model_cv.alpha_, c='k', linestyle='--', label="alpha CV")
    plt.xlabel("alpha")
    plt.ylabel("weights")
    plt.title(f"{model.__class__.__name__} coefficients as a function of the regularization")
    plt.axis("tight")
    plt.legend()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.show()

    ax = plt.gca()
    ax.plot(fitted_model_cv.cv_alphas_, mse)
    ax.set_xscale("log")
    plt.axvline(x=fitted_model_cv.alpha_, c='k', linestyle='--', label="alpha CV")
    plt.xlabel("alpha")
    plt.ylabel("train mse")
    plt.title(f"{model.__class__.__name__} mse as a function of the regularization")
    plt.axis("tight")
    plt.legend()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.show()

    print(f'\n\n')


def get_model(model_type, fitted_model_cv):

    if model_type == 'lasso_lars_cv':

        model = LassoLars(
            alpha=1.0,
            fit_intercept=True,
            verbose=False,
            precompute='auto',
            max_iter=500,
            eps=np.finfo(float).eps,
            copy_X=True,
            fit_path=True,
            positive=False,
            jitter=None,
            random_state=42
        )

    elif model_type == 'lasso_cv':

        model = Lasso(
            alpha=1.0,
            fit_intercept=True,
            precompute=False,
            copy_X=True,
            max_iter=1000,
            tol=0.0001,
            warm_start=False,
            positive=False,
            random_state=None,
            selection='cyclic'
        )

    elif model_type == 'elastic_net_cv':

        l1_ratio = fitted_model_cv.l1_ratio_

        model = ElasticNet(
            alpha=1.0,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            precompute=False,
            max_iter=1000,
            copy_X=True,
            tol=0.0001,
            warm_start=False,
            positive=False,
            random_state=None,
            selection='cyclic'
        )

    else:
        sys.exit(f'model_type = {model_type} is not recognized')

    return model


def print_out_summary_2(model, model_type, fitted_model_cv):

    print(f'\nwe use the scikit-learn {model.__class__.__name__} model fitted '
          f'to the train set to map out the\ncoefficient values and mse along the '
          f'regularization path followed by {fitted_model_cv.__class__.__name__}')

    if model_type == 'elastic_net_cv':
        print(f'the optimum l1_ratio determined by {fitted_model_cv.__class__.__name__} is {fitted_model_cv.l1_ratio_} '
              f'is used')

    print(f'\nnote that this is an approximation to the coefficient values and mse found in the\n'
          f'{fitted_model_cv.__class__.__name__} because here we are using the whole train set')

    print(f'\nas the number of folds in {fitted_model_cv.__class__.__name__} increases this becomes\n'
          f'a better approximation')

    print(f'\nin the future this will be fixed')

    print(f'')
    print(f'')


def plot_mse_coef_trade_space(mse_coef_trade_df, fitted_model_cv):

    print(f'')
    print(f'')

    # Create the figure and the first y-axis
    fig, ax1 = plt.subplots()

    # plot the first line plot on the left y-axis
    sns.lineplot(data=mse_coef_trade_df, x='alpha', y='non_zero_coef_count', ax=ax1, color='k')

    # tick on integers only
    ax1.yaxis.get_major_locator().set_params(integer=True)

    ax1.set_ylabel('non_zero_coef_count (black)')
    ax1.set_xscale("log")
    plt.grid()

    # create the second y-axis
    ax2 = ax1.twinx()

    # plot the second line plot on the right y-axis
    sns.lineplot(data=mse_coef_trade_df, x='alpha', y='frac_mse_increase_above_min_mse', ax=ax2, color='r')
    ax2.set_ylabel('frac_mse_increase_above_min_mse (red)')
    ax2.set_xscale("log")

    plt.axvline(x=fitted_model_cv.alpha_, c='k', linestyle='--', label='alpha cv')
    plt.legend()
    plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

    plt.title(f'mean mse versus number of non zero coefficients trade space')
    plt.show()


def plot_coef_and_mse_path(fitted_model_cv, cap_x_df, y_df, model_type):

    # TODO: instead of using the whole train set use the sme folds as used in the model specific cv

    model = get_model(model_type, fitted_model_cv)

    print_out_summary_2(model, model_type, fitted_model_cv)

    coefs = []
    mse_list = []
    df_row_dict_list = []
    for alpha in fitted_model_cv.cv_alphas_:  # iterate through parameter grid

        # set alpha and fit the model
        model.set_params(**{'alpha': alpha})
        model.fit(cap_x_df, y_df.values.ravel())

        # grab fit coefficients
        coefs.append(model.coef_)

        # get rmse of fit on train data
        mse = mean_squared_error(y_df, model.predict(cap_x_df))
        mse_list.append(mse)

        # prepare df_row_dict_list - we use it to study the mse vs number of zero coefficient trade space
        if model_type == 'elastic_net_cv':
            l1_ratio = fitted_model_cv.l1_ratio_
        else:
            l1_ratio = None

        df_row_dict_list.append(
            {
                'alpha': alpha,
                'l1_ratio': l1_ratio,
                'non_zero_coef_count': count_non_zero_coef(model.coef_),
                'mse': mse,
            }
        )

    plot_coef_and_mse_path_util(fitted_model_cv, coefs, mse_list, model)

    mse_coef_trade_df = pd.DataFrame(df_row_dict_list)
    mse_coef_trade_df['frac_mse_increase_above_min_mse'] = (
            (mse_coef_trade_df.mse - mse_coef_trade_df.mse.min()) / mse_coef_trade_df.mse.min())
    plot_mse_coef_trade_space(mse_coef_trade_df, fitted_model_cv)

    return coefs


def count_non_zero_coef(coef_list):
    return len([coef for coef in coef_list if coef == 0])


def make_working_df(fitted_model_cv):

    df_row_dict_list = []
    for i in range(fitted_model_cv.cv_alphas_.shape[0]):
        for alpha_test_error_rate in fitted_model_cv.mse_path_[i, :]:
            df_row_dict_list.append(
                {
                    'alpha': fitted_model_cv.cv_alphas_[i],
                    'mse': alpha_test_error_rate,
                }
            )

    working_df = pd.DataFrame(df_row_dict_list)

    return working_df


def get_new_alpha(working_df, num_std=1.0):
    
    # num_std establishes how much mse you want to give up to get a simpler model

    # get the alpha at which the mean test error (mse) is minimum
    min_mean_mse_alpha = working_df.groupby('alpha').mean().idxmin().iloc[0]

    # use the min_mean_alpha to get the min_mean
    min_mean_mse = working_df.groupby('alpha').mean().loc[min_mean_mse_alpha].iloc[0]

    # use the min_mean_alpha to get the standard deviation of min_mean
    min_mean_mse_std = working_df.groupby('alpha').std().loc[min_mean_mse_alpha].iloc[0]

    # under this trade what is the new mean mse
    trade_mean_mse = min_mean_mse + num_std * min_mean_mse_std

    # under this trade what is the fractional increase in mse
    percent_increase = (trade_mean_mse - min_mean_mse) / min_mean_mse

    # what is the closest alpha to the new mse
    new_alpha = working_df.groupby('alpha').mean().sub(trade_mean_mse).abs().idxmin().iloc[0]
    
    return_dict = {
        'min_mean_mse_alpha': min_mean_mse_alpha,
        'min_mean_mse': min_mean_mse,
        'min_mean_mse_std': min_mean_mse_std,
        'num_std': num_std,
        'trade_mean_mse': trade_mean_mse,
        'percent_increase': percent_increase,
        'new_alpha': new_alpha
    }

    return return_dict


def new_alpha_helper_plots(working_df, new_alpha, fitted_model_cv, coefs, model_type):

    # plot the mean mse path vs alpha showing the min_mean_alpha and the new_alpha
    ax = sns.lineplot(data=working_df, x='alpha', y='mse', errorbar=('se', 1), label='mean_mse')
    ax.set(xscale='log')
    plt.axvline(x=fitted_model_cv.alpha_, c='k', linestyle='--', label='alpha cv')
    plt.axvline(x=new_alpha, c='r', linestyle='--', label='new_alpha')
    plt.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.5))
    plt.title(f"{model_type} mean mse as a function of the regularization")
    plt.grid()
    plt.show()

    # plot the coefficient paths vs alpha showing the min_mean_alpha and the new_alpha
    ax = plt.gca()
    ax.plot(fitted_model_cv.cv_alphas_, coefs)
    ax.set_xscale("log")
    plt.axvline(x=fitted_model_cv.alpha_, c='k', linestyle='--', label='alpha cv')
    plt.axvline(x=new_alpha, c='r', linestyle='--', label='new_alpha')
    plt.legend()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("alpha")
    plt.ylabel("weights")
    plt.title(f"{model_type} coefficients as a function of the regularization")
    plt.axis("tight")
    plt.grid()
    plt.show()


def convert_attributes_of_fitted_cv(model_type, fitted_model_cv, l1_ratio_list):

    # this code was developed for LassoLarsCV then adapted to ElasticNetCV. The code below serves as an
    # interface to make ElasticNet and LassoCV work with the LassoLarsCV code

    if model_type == 'elastic_net_cv':
        # make a copy of fitted object before modifying the original
        unmodified_fitted_cv = copy.deepcopy(fitted_model_cv)
        # get the best l1 ratio index from the l1 ratio list passed in
        best_l1_ratio_idx = l1_ratio_list.index(fitted_model_cv.l1_ratio_)
        # get the alphas for the best l1 ratio
        fitted_model_cv.cv_alphas_ = fitted_model_cv.alphas_[best_l1_ratio_idx, :]
        # get the mse path for the best l1 ratio
        fitted_model_cv.mse_path_ = fitted_model_cv.mse_path_[best_l1_ratio_idx]
    elif model_type == 'lasso_cv':
        # make a copy of fitted object before modifying the original
        unmodified_fitted_cv = copy.deepcopy(fitted_model_cv)
        fitted_model_cv.cv_alphas_ = fitted_model_cv.alphas_
    elif model_type == 'lasso_lars_cv':
        unmodified_fitted_cv = None
    else:
        sys.exit(f'model_type = {model_type} is not recognized')

    return unmodified_fitted_cv, fitted_model_cv


def print_out_summary_1(model_type, cv_folds, fitted_model_cv, unmodified_fitted_cv, fitted_coef_dict):

    if unmodified_fitted_cv is not None:  # replace modified object with original object
        fitted_model_cv = unmodified_fitted_cv

    if model_type != 'elastic_net_cv':
        l1_ratio = None
    else:
        l1_ratio = fitted_model_cv.l1_ratio_

    print(f'\nscikit-learn model specific cv {fitted_model_cv.__class__.__name__} was fitted with the following '
          f'results')
    print(f'\nbest regularization parameters:')
    print(f'   best alpha: {fitted_model_cv.alpha_}')
    print(f'   best l1_ratio: {l1_ratio}')
    print(f'\nthe model coefficients at the best regularization hyperparameters are:')
    print_out_coef_dict_utility(fitted_coef_dict)

    print(f'\nnumber of attributes with zero coefficient: {count_non_zero_coef(fitted_coef_dict.values())}')

    print(f'\nthe test error estimate of mse of the {cv_folds} cross validation folds\n'
          f'along the regularization path is shown below')
    print(f'')
    print(f'')


def print_out_coef_dict_utility(coef_dict):
    for attr_name, coefficient_value in coef_dict.items():
        print(f'   attribute {attr_name} coefficient: {coefficient_value}')


def inform_1(get_new_alpha_return_dict, fitted_model_cv):

    min_mean_mse_alpha = get_new_alpha_return_dict['min_mean_mse_alpha']
    min_mean_mse = get_new_alpha_return_dict['min_mean_mse']
    min_mean_mse_std = get_new_alpha_return_dict['min_mean_mse_std']
    num_std = get_new_alpha_return_dict['num_std']
    trade_mean_mse = get_new_alpha_return_dict['trade_mean_mse']
    percent_increase = get_new_alpha_return_dict['percent_increase']
    new_alpha = get_new_alpha_return_dict['new_alpha']

    print(f'')
    print(f'')
    print(f'\nnext we consider trading off mse to get simpler model - the plot above shows the trade space\n'
          f'\nthe minimum mean mse from {fitted_model_cv.__class__.__name__} is located at alpha {min_mean_mse_alpha}\n'
          f'\nat that alpha the minimum mean mse from cv is {min_mean_mse}\n'
          f'\nat that alpha the standard deviation of the mse from cv is {min_mean_mse_std}\n'
          f'\nconsider trading {num_std} standard deviations of mse to move towards more regulation\n'
          f'and more zero coefficients\n'
          f'\nthis means moving to a new alpha {new_alpha} where the mse is {trade_mean_mse}\n'
          f'\nthe mse has increased by {percent_increase}\n'
          f'\nthe plots below show the original alpha from {fitted_model_cv.__class__.__name__} and the new alpha\n'
          f'for both the coefficient paths and the mean mse path')
    print(f'')
    print(f'')


def print_out_summary_3(fitted_coef_dict, new_fitted_coef_dict, get_new_alpha_return_dict, fitted_model_cv):

    print(f'\n')
    print(f'\n')

    min_mean_mse_alpha = get_new_alpha_return_dict['min_mean_mse_alpha']
    percent_increase = get_new_alpha_return_dict['percent_increase']
    new_alpha = get_new_alpha_return_dict['new_alpha']

    print(f'by shifting from the best alpha = {min_mean_mse_alpha} given by {fitted_model_cv.__class__.__name__} to '
          f'a new alpha = {new_alpha}\nwe have increased the number of zero coefficients from '
          f'{count_non_zero_coef(fitted_coef_dict.values())} to {count_non_zero_coef(new_fitted_coef_dict.values())}'
          f' giving us a simpler model\n'
          f'\nthe cost of the simpler model is a {percent_increase} increase in mse')

    print(f'\nthe original model coefficients were:\n')
    print_out_coef_dict_utility(fitted_coef_dict)

    print(f'\nthe new model coefficients are:\n')
    print_out_coef_dict_utility(new_fitted_coef_dict)


def model_specific_cv(cap_x_df, y_df, nominal_attr, numerical_attr, model_type, cv_folds=5, l1_ratio_list=None,
                      num_std=1.0, te_random_state=42):

    # number of mse standard deviations to give up for a reduction in variables

    # preprocess the data
    preproc_cap_x_df = get_preproc_cap_x_df(cap_x_df, y_df, nominal_attr, numerical_attr,
                                            te_random_state=te_random_state)
    # get the model and fit it
    model_cv = get_model_specific_cv(model_type, cv_folds, l1_ratio_list)
    fitted_model_cv = model_cv.fit(preproc_cap_x_df, y_df.values.ravel())

    # bring fitted attributes into alignment for this script
    unmodified_fitted_cv, fitted_model_cv = convert_attributes_of_fitted_cv(model_type, fitted_model_cv, l1_ratio_list)

    # load the fitted coefficients from the model specific cv into a dictionary
    fitted_coef_dict = dict(zip(preproc_cap_x_df.columns, fitted_model_cv.coef_))
    print_out_summary_1(model_type, cv_folds, fitted_model_cv, unmodified_fitted_cv, fitted_coef_dict)

    # print out the mse path from the model specific cv
    plot_mse_path(fitted_model_cv, model_type, cv_folds)

    # use the models themselves to map out the coefficient path and the mse path
    coefs = plot_coef_and_mse_path(fitted_model_cv, preproc_cap_x_df, y_df, model_type)

    # trade off mse for variable selection
    working_df = make_working_df(fitted_model_cv)
    get_new_alpha_return_dict = get_new_alpha(working_df, num_std=num_std)
    new_alpha = get_new_alpha_return_dict['new_alpha']

    inform_1(get_new_alpha_return_dict, fitted_model_cv)

    new_alpha_helper_plots(working_df, new_alpha, fitted_model_cv, coefs, model_type)
    new_coefs_idx = list(fitted_model_cv.cv_alphas_).index(new_alpha)
    new_fitted_coef_dict = dict(zip(preproc_cap_x_df.columns, coefs[new_coefs_idx]))

    print_out_summary_3(fitted_coef_dict, new_fitted_coef_dict, get_new_alpha_return_dict, fitted_model_cv)

    if unmodified_fitted_cv is not None:  # replace modified object with original object
        fitted_model_cv = unmodified_fitted_cv

    return_dict = {
        'preproc_cap_x_df': preproc_cap_x_df,
        'model_type': model_type,
        'fitted_model_cv': fitted_model_cv,
        'fitted_coef_dict': fitted_coef_dict,
        'new_fitted_coef_dict': new_fitted_coef_dict
    }

    return return_dict


def fit_stats_models_ols(cap_x_y_df, print_summary=False):

    sm_train_cap_x_df = sm.add_constant(cap_x_y_df.iloc[:, :-1])
    model = sm.OLS(cap_x_y_df.iloc[:, -1], sm_train_cap_x_df).fit()

    if print_summary:
        print(model.summary())

    return model


def predict_with_fitted_stats_model_ols(model, cap_x_df):

    sm_cap_x_df = sm.add_constant(cap_x_df)
    pred_y_df = model.predict(sm_cap_x_df).to_frame().rename(columns={0: 'pred_y'})

    return pred_y_df


class SMWrapper(BaseEstimator, RegressorMixin):
    """
    https://stackoverflow.com/questions/41045752/
    using-statsmodel-estimations-with-scikit-learn-cross-validation-is-it-possible

    A universal sklearn-style wrapper for statsmodels regressors
    cross_val_score(SMWrapper(sm.OLS), X, y, scoring='r2')
    compare to
    cross_val_score(LinearRegression(), X, y, scoring='r2')
    """

    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
        self.model_ = None
        self.results_ = None

    def fit(self, cap_x, y):
        if self.fit_intercept:
            cap_x = sm.add_constant(cap_x)
        self.model_ = self.model_class(y, cap_x)
        self.results_ = self.model_.fit()
        return self

    def predict(self, cap_x):
        if self.fit_intercept:
            cap_x = sm.add_constant(cap_x)
        return self.results_.predict(cap_x)


if __name__ == "__main__":
    pass
