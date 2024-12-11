import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from utils.regression_utils import (cv_scores_dict_to_cv_scores_df, cv_scores_analysis,
                                    plot_pred_vs_actual_survey, flexibility_plot_regr)
from sklearn.utils import resample


def perform_the_train_test_split(df, test_size, train_test_split_random_state, prefix=None, val=False,
                                 classification_threshold=False, prob_cal=False, stratify=False):

    # if val = False then train/test split
    # if val = True then train/validation split - only difference is that the test set is saved as the val set

    if prefix is None:
        prefix = ''
    else:
        prefix = prefix + '_'

    if val:
        small_set_name = 'validation_df.csv'
    elif classification_threshold:
        small_set_name = 'class_thresh_set_df.csv'
    elif prob_cal:
        small_set_name = 'prob_cal_set_df.csv'
    else:
        small_set_name = 'test_df.csv'

    cap_x_df, y_df = df.iloc[:, :-1], df.iloc[:, -1].to_frame()
    if stratify:
        stratify = y_df
    else:
        stratify = None

    train_cap_x_df, test_cap_x_df, train_y_df, test_y_df = \
        train_test_split(cap_x_df, y_df, test_size=test_size, random_state=train_test_split_random_state, shuffle=True,
                         stratify=stratify)

    report_check_split_details_save_data_sets(df, train_cap_x_df, train_y_df, small_set_name, test_cap_x_df, test_y_df,
                                              prefix, stratify)

    del test_cap_x_df, test_y_df

    return train_cap_x_df, train_y_df


def report_check_split_details_save_data_sets(df, train_cap_x_df, train_y_df, small_set_name, test_cap_x_df, test_y_df,
                                              prefix, stratify):
    print(25 * '*')
    print('\ndf.shape:')
    print(df.shape)
    target_attr = None
    if stratify is not None:
        target_attr = train_y_df.columns[0]
        print(f'\ntarget class fractional balance:\n{df[target_attr].value_counts()/df.shape[0]}', sep='')

    print('\n', 25 * '*', sep='')
    print('\ntrain_df.csv:')
    print(train_cap_x_df.shape, train_y_df.shape)
    if stratify is not None:
        print(f'\ntarget class fractional balance:\n{train_y_df[target_attr].value_counts()/train_y_df.shape[0]}',
              sep='')

    print('\n', 25 * '*', sep='')
    print('\n', small_set_name, sep='')
    print(test_cap_x_df.shape, test_y_df.shape)
    if stratify is not None:
        print(f'\ntarget class fractional balance:\n{test_y_df[target_attr].value_counts()/test_y_df.shape[0]}',
              sep='')

    assert (list(train_cap_x_df.index) == list(train_y_df.index))
    assert (list(test_cap_x_df.index) == list(test_y_df.index))

    pd.concat([train_cap_x_df, train_y_df], axis=1).to_csv(prefix + 'train_df.csv', index=True,
                                                           index_label='index')
    pd.concat([test_cap_x_df, test_y_df], axis=1).to_csv(prefix + small_set_name, index=True, index_label='index')


def get_missingness(df, missingness_threshold):

    missingness_drop_list = []
    for attr in df.columns:
        attr_missingness = df[attr].isna().sum() / df.shape[0]
        print(f'{attr} missingness = {attr_missingness}')
        if attr_missingness >= missingness_threshold:
            missingness_drop_list.append(attr)

    print(f'\nmissingness_drop_list:\n{missingness_drop_list}')

    return_dict = {
        'missingness_drop_list': missingness_drop_list
    }

    return return_dict


def model_survey_fit(preprocessor, estimator_dict, train_cap_x_df, train_y_df):

    trained_estimator_dict = {}
    for estimator_name, estimator in estimator_dict.items():
        print(estimator_name)

        composite_estimator = \
            Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('estimator', estimator)
                ]
            )

        composite_estimator.fit(train_cap_x_df, train_y_df.values.ravel())

        trained_estimator_dict[estimator_name] = composite_estimator

    return_dict = {
        'trained_estimator_dict': trained_estimator_dict,
    }

    return return_dict


def model_survey_cross_validation(preprocessor, estimator_dict, train_cap_x_df, train_y_df=None, scoring=None,
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

    for estimator_name, estimator in estimator_dict.items():

        composite_estimator = \
            Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('estimator', estimator)
                ]
            )

        scores_dict = cross_validate(
            estimator=composite_estimator,
            X=train_cap_x_df,
            y=train_y_df.values.ravel(),
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


def train_val_split_for_sklearn_cross_validate(train_cap_x_df, train_y_df, num_train_val_splits=1, val_size=0.50,
                                               val_split_random_state=42):

    splitter = []
    for i in range(val_split_random_state, val_split_random_state + num_train_val_splits):
        return_dict = train_val_split_for_sklearn_cross_validate_helper(train_cap_x_df, train_y_df, val_size=val_size,
                                                                        val_split_random_state=i)
        splitter_i = return_dict['splitter']
        splitter.append(splitter_i[0])

    return_dict = {
        'splitter': splitter
    }

    return return_dict


def train_val_split_for_sklearn_cross_validate_helper(train_cap_x_df, train_y_df, val_size=0.50,
                                                      val_split_random_state=42):

    # use the scikit-learn train / test function to create the train / val split - note that the scikit-learn
    # cross_validate function uses iloc indices therefore we rest indices
    cv_train_df, cv_test_df = train_test_split(
        pd.concat([train_cap_x_df, train_y_df], axis=1).reset_index(),
        test_size=val_size,
        train_size=None,
        random_state=val_split_random_state,
        shuffle=True,
        stratify=None
    )
    splitter = [(cv_train_df.index.values, cv_test_df.index.values)]

    return_dict = {
        'splitter': splitter
    }

    return return_dict


def get_learning_curve_tuple(estimator, scoring, train_cap_x_df, train_y_df, cv=5,
                             train_sizes=np.array([0.1, 0.33, 0.55, 0.78, 1.])):

    learning_curve_tuple = learning_curve(
        estimator=estimator,
        X=train_cap_x_df,
        y=train_y_df.values.ravel(),
        groups=None,
        train_sizes=train_sizes,
        cv=cv,  # None = default 5-fold cross validation
        scoring=scoring,
        exploit_incremental_learning=False,
        n_jobs=None,
        pre_dispatch='all',
        verbose=0,
        shuffle=False,
        random_state=None,  # when shuffle is True only
        error_score=np.nan,
        return_times=False,
        fit_params=None
    )

    return_dict = {
        'learning_curve_tuple': learning_curve_tuple,
        'scoring': scoring
    }

    return return_dict


def learning_curve_tuple_to_data_frame(learning_curve_tuple, scoring):

    df_row_dict_list = []
    for train_size, train_scores, test_scores in zip(learning_curve_tuple[0], learning_curve_tuple[1],
                                                     learning_curve_tuple[2]):

        for i, score in enumerate(train_scores):
            if 'neg' in scoring:
                score = -1 * score
            df_row_dict_list.append(
                {
                    'train_size': train_size,
                    'fold': i,
                    'score_type': 'train',
                    'score': score
                }
            )

        for i, score in enumerate(test_scores):
            if 'neg' in scoring:
                score = -1 * score
            df_row_dict_list.append(
                {
                    'train_size': train_size,
                    'fold': i,
                    'score_type': 'test',
                    'score': score
                }
            )

    df = pd.DataFrame(df_row_dict_list)

    return_dict = {
        'learning_curve_df': df
    }

    return return_dict


def plot_learning_curve(learning_curve_df, estimator_name, scoring):
    ax = sns.lineplot(data=learning_curve_df, x='train_size', y='score', hue='score_type', marker='o')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title(f'estimator name: {estimator_name}\nscoring: {scoring}')
    plt.xlabel('number of samples in training set')
    plt.grid()
    plt.show()


def plot_learning_curves(estimator, estimator_name, scoring, train_cap_x_df, train_y_df, cv=5,
                         train_sizes=np.array([0.1, 0.33, 0.55, 0.78, 1.])):

    return_dict = get_learning_curve_tuple(estimator, scoring, train_cap_x_df, train_y_df, cv, train_sizes)
    learning_curve_tuple = return_dict['learning_curve_tuple']

    return_dict = learning_curve_tuple_to_data_frame(learning_curve_tuple, scoring)
    learning_curve_df = return_dict['learning_curve_df']
    learning_curve_df.head()

    plot_learning_curve(learning_curve_df, estimator_name, scoring)


def model_survey_cross_val_and_analysis_helper(cv_scores_dict, target_attr, trained_estimator_dict, train_cap_x_df,
                                               train_y_df, splitter, histplot=False, gs_survey_results_df=None,
                                               boxplot=False, catplot=False, task=None, return_=False):

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


def model_survey_cross_val_and_analysis(preprocessor, estimator_dict, train_cap_x_df, train_y_df, scoring, splitter,
                                        target_attr, trained_estimator_dict, boxplot=True, catplot=True, task=None,
                                        **kwargs):

    # perform model survey cross validation
    return_dict = model_survey_cross_validation(preprocessor, estimator_dict, train_cap_x_df, train_y_df, scoring,
                                                splitter, **kwargs)
    cv_scores_dict = return_dict['cv_scores_dict']

    model_survey_cross_val_and_analysis_helper(cv_scores_dict, target_attr, trained_estimator_dict, train_cap_x_df,
                                               train_y_df, splitter, boxplot=boxplot, catplot=catplot, task=task)


def select_score_df(gs_cv_results_df, score_of_interest, scoring):

    # drop score attributes that do not apply to the score of interest

    scores_to_drop = [score for score in scoring if score != score_of_interest]
    all_attr_to_drop = []
    for score in scores_to_drop:
        attr_drop_list = [attr for attr in gs_cv_results_df if score in attr]
        all_attr_to_drop.extend(attr_drop_list)

    gs_cv_results_df = gs_cv_results_df.drop(columns=all_attr_to_drop)

    return gs_cv_results_df


def plot_flexibility(grid_search_cv, estimator_name, scoring):

    results_dict_list = []
    for score in scoring:

        # extract results from grid_search_cv as data frame
        gs_cv_results_df = pd.DataFrame(grid_search_cv.cv_results_)

        # select the score data frame of interest
        gs_cv_results_df = select_score_df(gs_cv_results_df, score, scoring)

        # plot the flexibility plot
        gs_cv_results_df, best_index = flexibility_plot_regr(gs_cv_results_df, estimator_name, score)

        # get the best test score
        best_test_score = gs_cv_results_df.loc[gs_cv_results_df.index == best_index, 'mean_test_' + score].values[0]

        # collect the results of the analysis
        results_dict_list.append(
            {
                'score': score,
                'best_test_score': best_test_score,
                'best_index': best_index,
                'grid_search_cv': grid_search_cv,
                'gs_cv_results_df': gs_cv_results_df,
            }
        )

    return_dict = {
        'results_df': pd.DataFrame(results_dict_list)
    }

    return return_dict


def model_tuning_cross_val_and_analysis(tuned_estimator_dict, train_cap_x_df, train_y_df, scoring, splitter,
                                        target_attr, return_indices=False, drop_cv_times=True, shuffle_target=False,
                                        shuffle_target_random_state=42, histplot=False, gs_survey_results_df=None,
                                        boxplot=True, catplot=True, task=None, return_=False):

    cv_scores_dict = {'scoring': scoring}
    best_estimator_dict = {}

    if shuffle_target:  # used to check for false discoveries
        train_y_df = train_y_df.sample(frac=1, random_state=shuffle_target_random_state)

    for estimator_name, tuned_estimator in tuned_estimator_dict.items():

        try:
            best_estimator = tuned_estimator.best_estimator_
        except AttributeError:
            best_estimator = tuned_estimator

        scores_dict = cross_validate(
            estimator=best_estimator,
            X=train_cap_x_df,
            y=train_y_df.values.ravel(),
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
        best_estimator_dict[estimator_name] = best_estimator

    return_dict = model_survey_cross_val_and_analysis_helper(cv_scores_dict, target_attr, best_estimator_dict,
                                                             train_cap_x_df, train_y_df, splitter, histplot=histplot,
                                                             boxplot=boxplot, catplot=catplot,
                                                             gs_survey_results_df=gs_survey_results_df, task=task,
                                                             return_=return_)

    if return_:
        return return_dict
    else:
        return None


def check_for_false_discoveries(tuned_estimator_dict, train_cap_x_df, train_y_df, scoring, splitter, target_attr,
                                return_indices=False, drop_cv_times=True, shuffle_target=True,
                                shuffle_target_random_state=42, histplot=True, gs_survey_results_df=None, task=None):

    model_tuning_cross_val_and_analysis(tuned_estimator_dict, train_cap_x_df, train_y_df, scoring, splitter,
                                        target_attr, return_indices=return_indices, drop_cv_times=drop_cv_times,
                                        shuffle_target=shuffle_target,
                                        shuffle_target_random_state=shuffle_target_random_state, histplot=histplot,
                                        gs_survey_results_df=gs_survey_results_df, boxplot=False, catplot=False,
                                        task=task)


def check_for_complete_unique_attrs(cap_x_df):

    print(f'the data frame has {cap_x_df.shape[0]} rows\n')

    concern_list = []
    for attr in cap_x_df.columns:
        label = ''
        if cap_x_df[attr].nunique() == cap_x_df.shape[0]:
            label = 'examine more closely'
            concern_list.append(attr)
        print(f'{attr} has {cap_x_df[attr].nunique()} unique values and is dtype {cap_x_df[attr].dtype} {label}')

    return concern_list


def sample_data_objects_for_speed_up(cap_x_df, y_df, frac=0.10, random_state=42):

    n_samples = int(frac * cap_x_df.shape[0])
    cap_x_y_df = cap_x_df.join(y_df)

    speed_up_cap_x_y_df = resample(
        cap_x_y_df,
        replace=False,
        n_samples=n_samples,
        random_state=random_state,
        stratify=cap_x_y_df.iloc[:, -1]
    )

    return speed_up_cap_x_y_df.iloc[:, :-1], speed_up_cap_x_y_df.iloc[:, -1]


if __name__ == "__main__":
    pass
