import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import math
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
import time
from sklearn.preprocessing import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from scipy.stats import bootstrap, normaltest, t
import sys
import utils.classification_utils as class_utils
from scipy.optimize import curve_fit
from kneed import KneeLocator
import datetime
import os
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
import imblearn


def test():
    print('test')


linear_estimator_list = ['LinearRegression', 'Ridge', 'ElasticNet', 'Lasso']


def get_speed_up_sample(cap_x_df, y_df, frac):

    print('\nbefore sample')
    print(f'cap_x_df.shape: {cap_x_df.shape}')
    print(f'y_df.shape: {y_df.shape}')

    df = pd.concat([cap_x_df, y_df], axis=1).sample(frac=frac, random_state=42)

    cap_x_df = df.iloc[:, :-1]
    y_df = df.iloc[:, -1].to_frame()

    print('\nafter sample')
    print(f'cap_x_df.shape: {cap_x_df.shape}')
    print(f'y_df.shape: {y_df.shape}')

    return cap_x_df, y_df


def get_columns_from_trained_estimator(trained_estimator):

    try:
        _columns = trained_estimator.steps[0][1].__dict__['_columns']
    except AttributeError as e:
        print(e)
        print(f'using alternative method to extract attributes from trained estimator')
        _columns_num = trained_estimator.__dict__['estimator'].steps[0][1].__dict__['transformers'][0][2]
        _columns_nom = trained_estimator.__dict__['estimator'].steps[0][1].__dict__['transformers'][1][2]
        _columns = [_columns_num, _columns_nom]

    return _columns


def extract_attrs_from_trained_estimator_and_order_them(trained_estimator, train_cap_x_df):

    # extract attribute names from trained preprocessing stage of composite estimator
    _columns = get_columns_from_trained_estimator(trained_estimator)

    col_count = 0
    col_list = []
    for columns in _columns:
        col_count += len(columns)
        col_list.extend(columns)

    # order the extracted attributes in the same ordering seen in teh training data frame
    ordered_col_list = []
    for attr in train_cap_x_df.columns:
        if attr in col_list:
            ordered_col_list.append(attr)

    return ordered_col_list


def eval_trained_estimators_in_trained_estimator_dict(trained_estimator_dict, cap_x_df, y_df, data_set_type=None,
                                                      model_selection_stage=None, plot_pred_vs_actual_flag=True):
    # for regression

    if data_set_type is None:
        data_set_type = ''
    if model_selection_stage is None:
        model_selection_stage = ''

    df_row_dict_list = []
    for estimator_name, trained_estimator in trained_estimator_dict.items():

        eval_dict = eval_trained_estimator(trained_estimator, cap_x_df, y_df, model_selection_stage, estimator_name,
                                           data_set_type, plot_pred_vs_actual_flag=plot_pred_vs_actual_flag)

        ordered_col_list = extract_attrs_from_trained_estimator_and_order_them(trained_estimator, cap_x_df)

        eval_dict['estimator'] = estimator_name
        eval_dict['data_set_type'] = data_set_type
        eval_dict['model_selection_stage'] = model_selection_stage
        eval_dict['number_of_attrs'] = len(ordered_col_list)
        eval_dict['attrs'] = ordered_col_list
        df_row_dict_list.append(eval_dict)

    compare_df = pd.DataFrame(df_row_dict_list)
    compare_df = compare_df[['estimator', 'model_selection_stage', 'data_set_type', 'r_squared', 'rmse',
                             'frac_rmse', 'number_of_attrs', 'attrs']]

    return compare_df


def fit_collection_of_estimators_regression(numerical_attr, nominal_attr, estimator_names, estimator_list, cap_x_df,
                                            y_df, data_set_type, model_selection_stage):

    preprocessor = build_preprocessing_pipeline(numerical_attr, nominal_attr, 'continuous')

    trained_estimator_dict = {}
    for estimator_name, estimator in zip(estimator_names, estimator_list):
        composite_estimator = Pipeline(steps=[('preprocessor', preprocessor), ('estimator', estimator)])

        trained_estimator_dict[estimator_name] = composite_estimator.fit(cap_x_df, y_df.values.ravel())

    compare_df = \
        eval_trained_estimators_in_trained_estimator_dict(trained_estimator_dict, cap_x_df, y_df, data_set_type,
                                                          model_selection_stage)

    return compare_df, trained_estimator_dict


def get_class_eval_dict(eval_dict, score, score_name):

    if isinstance(score, dict):
        eval_dict['ave_' + score_name] = np.mean(list(score.values()))
        for k, v in score.items():
            score[k] = round(v, 3)
        eval_dict[score_name] = score
    else:
        eval_dict[score_name] = score

    return eval_dict


def eval_trained_estimator_class(trained_estimator, cap_x_df, y_df, model_selection_stage, estimator_name,
                                 data_set_type, class_eval_dict, print_eval_results=True):
    """

    :param trained_estimator:
    :param cap_x_df:
    :param y_df:
    :param model_selection_stage:
    :param estimator_name:
    :param data_set_type:
    :param class_eval_dict: key = name of function in classification_utils.py
                            value = [bool, function kwargs]  bool = True then call function
    :param print_eval_results:
    :return:
    """

    if estimator_name is None:
        estimator_name = ''
    if model_selection_stage is None:
        model_selection_stage = ''
    if data_set_type is None:
        data_set_type = ''

    binary = class_eval_dict['binary']
    eval_dict = {}
    for eval_type, eval_type_list in class_eval_dict.items():

        if eval_type == 'binary':
            continue

        if eval_type_list[0]:
            kwargs = eval_type_list[1]
            if eval_type == 'get_precision_recall_curves':
                _, ave_precision_score = \
                    class_utils.get_precision_recall_curves(trained_estimator, cap_x_df, y_df, data_set_type,
                                                            model_selection_stage, binary=binary, **kwargs)
                eval_dict = get_class_eval_dict(eval_dict, ave_precision_score, 'ave_precision_score')
            if eval_type == 'get_roc_curve':
                _, roc_auc_score_ = \
                    class_utils.get_roc_curve(trained_estimator, cap_x_df, y_df, data_set_type, model_selection_stage,
                                              binary=binary, **kwargs)
                eval_dict = get_class_eval_dict(eval_dict, roc_auc_score_, 'roc_auc_score_')

    if print_eval_results:
        print('\n', '*' * 50, sep='')
        print(f'{model_selection_stage} of the {estimator_name} estimator predicting on the {data_set_type} data set')
        print('\nroc_auc_score_:', eval_dict['roc_auc_score_'])
        if not binary:
            print('ave_roc_auc_score_:', eval_dict['ave_roc_auc_score_'])
        print('\nave_precision_score:', eval_dict['ave_precision_score'])
        if not binary:
            print('ave_ave_precision_score:', eval_dict['ave_ave_precision_score'])

    return eval_dict


def eval_trained_estimators_in_trained_estimator_dict_class(trained_estimator_dict, cap_x_df, y_df, data_set_type,
                                                            model_selection_stage, class_eval_dict):
    df_row_dict_list = []
    for estimator_name, trained_estimator in trained_estimator_dict.items():

        if trained_estimator is not None:
            eval_dict = \
                eval_trained_estimator_class(trained_estimator, cap_x_df, y_df, model_selection_stage, estimator_name,
                                             data_set_type, class_eval_dict)
            ordered_col_list = extract_attrs_from_trained_estimator_and_order_them(trained_estimator, cap_x_df)
            len_ordered_col_list = len(ordered_col_list)
        else:
            # added to deal with imbalance ADASYN ValueError: No samples will be generated with the provided ratio
            # settings.
            ordered_col_list = None
            len_ordered_col_list = np.nan
            eval_dict = {}
            for key in class_eval_dict.keys():
                if key == 'get_precision_recall_curves':
                    eval_dict['ave_precision_score'] = np.nan
                elif key == 'get_roc_curve':
                    eval_dict['roc_auc_score_'] = np.nan
                else:
                    print('\n', '*' * 40, '\n', '*' * 40, '\n', '*' * 40)
                    print(f'\nclass_eval_dict key {key} is not recognized as a score')
                    print('\n', '*' * 40, '\n', '*' * 40, '\n', '*' * 40)

        eval_dict['estimator'] = estimator_name
        eval_dict['data_set_type'] = data_set_type
        eval_dict['model_selection_stage'] = model_selection_stage
        eval_dict['number_of_attrs'] = len_ordered_col_list
        eval_dict['attrs'] = ordered_col_list
        df_row_dict_list.append(eval_dict)

    compare_df = pd.DataFrame(df_row_dict_list)

    return compare_df


def fit_collection_of_estimators_classification(numerical_attr, nominal_attr, estimator_names, estimator_list, cap_x_df,
                                                y_df, data_set_type, model_selection_stage, class_eval_dict, re_sample,
                                                sampler_kwargs):

    preprocessor = get_class_preproc_pipeline(numerical_attr, nominal_attr, class_eval_dict)

    trained_estimator_dict = {}
    for estimator_name, estimator in zip(estimator_names, estimator_list):

        if re_sample is None:
            composite_estimator = Pipeline(steps=[('preprocessor', preprocessor), ('estimator', estimator)])
        else:
            print(re_sample)
            if re_sample == 'RandomOverSampler':
                sampler = imblearn.over_sampling.RandomOverSampler(**sampler_kwargs)
            elif re_sample == 'SMOTE':
                sampler = imblearn.over_sampling.SMOTE(**sampler_kwargs)
            elif re_sample == 'ADASYN':
                sampler = imblearn.over_sampling.ADASYN(**sampler_kwargs)
            elif re_sample == 'BorderlineSMOTE':
                sampler = imblearn.over_sampling.BorderlineSMOTE(**sampler_kwargs)
            elif re_sample == 'KMeansSMOTE':
                sampler = imblearn.over_sampling.KMeansSMOTE(**sampler_kwargs)
            elif re_sample == 'SVMSMOTE':
                sampler = imblearn.over_sampling.SVMSMOTE(**sampler_kwargs)

            composite_estimator = imblearn.pipeline.make_pipeline(preprocessor, sampler, estimator)

        try:
            # added to deal with imbalance ADASYN ValueError: No samples will be generated with the provided ratio
            # settings.
            trained_estimator_dict[estimator_name] = composite_estimator.fit(cap_x_df, y_df.values.ravel())
        except (ValueError, RuntimeError) as e:
            print(f'\n{e}\n')
            trained_estimator_dict[estimator_name] = None

    compare_df = \
        eval_trained_estimators_in_trained_estimator_dict_class(trained_estimator_dict, cap_x_df, y_df, data_set_type,
                                                                model_selection_stage, class_eval_dict)

    return compare_df, trained_estimator_dict


def fit_collection_of_estimators(numerical_attr, nominal_attr, estimator_names, estimator_list, cap_x_df, y_df,
                                 data_set_type=None, model_selection_stage=None, prediction_task_type='regression',
                                 class_eval_dict=None, re_sample=None, sampler_kwargs=None):

    if data_set_type is None:
        data_set_type = ''
    if model_selection_stage is None:
        model_selection_stage = ''

    if prediction_task_type == 'regression':

        compare_df, trained_estimator_dict = \
            fit_collection_of_estimators_regression(numerical_attr, nominal_attr, estimator_names, estimator_list,
                                                    cap_x_df, y_df, data_set_type, model_selection_stage)

    elif prediction_task_type == 'classification':

        compare_df, trained_estimator_dict = \
            fit_collection_of_estimators_classification(numerical_attr, nominal_attr, estimator_names, estimator_list,
                                                        cap_x_df, y_df, data_set_type, model_selection_stage,
                                                        class_eval_dict=class_eval_dict, re_sample=re_sample,
                                                        sampler_kwargs=sampler_kwargs)

    else:
        print(f'{prediction_task_type} is not a recognized prediction_task_type')
        sys.exit()

    return compare_df, trained_estimator_dict


def check_out_vifs_of_preprocessed_design_matrices_of_a_collection_of_trained_estimators(
        trained_estimator_dict, cap_x_df, data_set_type=None, model_selection_stage=None):

    if data_set_type is None:
        data_set_type = 'none provided'
    if model_selection_stage is None:
        model_selection_stage = 'none provided'

    i = -1
    temp_df = None
    for estimator_name, trained_estimator in trained_estimator_dict.items():
        i += 1
        print(f'\n', 80 * '*', sep='')
        print(80 * '*', sep='')
        print(f'estimator_name: {estimator_name}; data_set_type: {data_set_type}; model_selection_stage: '
              f'{model_selection_stage}')

        ordered_col_list = \
            extract_attrs_from_trained_estimator_and_order_them(trained_estimator, cap_x_df)

        a_df = pd.DataFrame(
            trained_estimator.steps[0][1].transform(cap_x_df[ordered_col_list]),
            index=cap_x_df[ordered_col_list].index,
            columns=cap_x_df[ordered_col_list].columns
        )

        if i == 0:
            temp_df = a_df.copy()
        elif a_df.equals(temp_df):
            print(f'\nthe data frame for this estimator is the same as the data frame for the first estimator = '
                  f'vifs are the same - do not reprint')
            continue
        attr_eda_utils.print_vifs(a_df, list(a_df.columns), ols_large_vifs=False)


def perform_the_train_test_split(df, test_size, train_test_split_random_state, prefix=None, val=False, stratify=False):

    # if val = False then train/test split
    # if val = True then train/validation split - only difference is that the test set is saved as the val set

    if prefix is None:
        prefix = ''
    else:
        prefix = prefix + '_'

    if val:
        small_set_name = 'validation_df.csv'
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


def build_preprocessing_pipeline(numerical_attr, nominal_attr, target_type):

    numerical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer()),
               ("scaler", StandardScaler())]
    )

    nominal_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")),
               ('target_encoder', TargetEncoder(target_type=target_type, random_state=42)),
               ("scaler", StandardScaler())
               ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numerical_transformer, numerical_attr),
            ('nominal', nominal_transformer, nominal_attr)
        ]
    )

    return preprocessor


def run_grid_search_cv(composite_estimator, param_grid, cap_x_df, y_df, target_attr, prediction_task_type,
                       class_eval_dict):

    #     print(composite_estimator.get_params().keys())

    if prediction_task_type == 'regression':
        scoring = 'neg_mean_squared_error'
    else:
        scoring = class_eval_dict['scoring']
        print(scoring)

    # instantiate the grid search cv
    grid_search_cross_val = GridSearchCV(
        estimator=composite_estimator,
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        cv=5,
        verbose=1,
        pre_dispatch='2*n_jobs',
        error_score='raise',  # np.nan,
        return_train_score=True
    )

    # fit the grid search cv
    grid_search_cross_val.fit(cap_x_df, y_df[target_attr].values.ravel())
    time.sleep(5)

    return grid_search_cross_val


def collect_grid_search_cv_results_regr(grid_search_cross_val, i, estimators, cap_x_df, y_df, target_attr,
                                        a_df_row_dict_list):

    # collect results
    a_df_row_dict = {}
    best_estimator = grid_search_cross_val.best_estimator_
    predictions = best_estimator.predict(cap_x_df)
    a_df_row_dict['iteration'] = i
    a_df_row_dict['estimator'] = estimators[i]
    a_df_row_dict['r_squared'] = r2_score(y_df, predictions)
    a_df_row_dict['train_rmse'] = mean_squared_error(y_df, predictions, squared=False)
    a_df_row_dict['train_relative_rmse'] = a_df_row_dict['train_rmse'] / y_df[target_attr].mean()
    a_df_row_dict['best_estimator'] = grid_search_cross_val.best_estimator_
    a_df_row_dict['best_estimator_hyperparameters'] = grid_search_cross_val.best_params_
    a_df_row_dict_list.append(a_df_row_dict.copy())

    return a_df_row_dict_list


def collect_grid_search_cv_results_class(grid_search_cross_val, i, estimator_list, cap_x_df, y_df, a_df_row_dict_list,
                                         class_eval_dict):

    # collect results
    a_df_row_dict = {}
    best_estimator = grid_search_cross_val.best_estimator_

    a_df_row_dict['iteration'] = i
    a_df_row_dict['estimator'] = estimator_list[i]

    binary = class_eval_dict['binary']
    kwargs = class_eval_dict['get_precision_recall_curves'][1]

    _, ave_precision_score = \
        class_utils.get_precision_recall_curves(best_estimator, cap_x_df, y_df, binary=binary, **kwargs)
    a_df_row_dict['ave_precision_score'] = ave_precision_score

    kwargs = class_eval_dict['get_roc_curve'][1]
    _, roc_auc_score_ = \
        class_utils.get_roc_curve(best_estimator, cap_x_df, y_df, binary=binary, **kwargs)
    a_df_row_dict['roc_auc_score_'] = roc_auc_score_

    a_df_row_dict['best_estimator'] = grid_search_cross_val.best_estimator_
    a_df_row_dict['best_estimator_hyperparameters'] = grid_search_cross_val.best_params_
    a_df_row_dict_list.append(a_df_row_dict.copy())

    return a_df_row_dict_list


def plot_flexibility_and_print_gs_cv_results_regr(grid_search_cross_val, estimator_list, i, fit_flex_curve,
                                                  print_gs_cv_results=False):

    # plot the flexibility plot
    gs_cv_results = pd.DataFrame(grid_search_cross_val.cv_results_).sort_values('rank_test_score')
    gs_cv_results, best_index_candidates = flexibility_plot_regr(gs_cv_results, estimator_list[i], fit_flex_curve)

    print('\n', gs_cv_results[gs_cv_results['rank_test_score'] == 1], sep='')

    if print_gs_cv_results:
        print('\n\n', '*' * 60, sep='')
        print('grid search results (each row is a candidate):\n')
        print(gs_cv_results)
        print('*' * 60, '\n\n', sep='')

    return gs_cv_results, best_index_candidates


# def trim_attrs_using_perm_imp_dict(numerical_attr, nominal_attr, perm_imp_dict):
#
#     estimator_attr_dict = {}
#     for estimator, attr_list in perm_imp_dict.items():
#         estimator_attr_dict[estimator] = {'numerical_attr': [], 'nominal_attr': [], 'non_ml_attr': []}
#         for attr in attr_list:
#             if attr in numerical_attr:
#                 estimator_attr_dict[estimator]['numerical_attr'].append(attr)
#             elif attr in nominal_attr:
#                 estimator_attr_dict[estimator]['nominal_attr'].append(attr)
#             else:
#                 estimator_attr_dict[estimator]['non_ml_attr'].append(attr)
#
#     return estimator_attr_dict
#
#
# def unpack_estimator_attr_dict(estimator_attr_dict, estimator):
#
#     numerical_attr = estimator_attr_dict[estimator]['numerical_attr']
#     nominal_attr = estimator_attr_dict[estimator]['nominal_attr']
#     non_ml_attr = estimator_attr_dict[estimator]['non_ml_attr']
#
#     return numerical_attr, nominal_attr, non_ml_attr


def grid_search_cv_helper_common(i, grid_search_cross_val, estimator_names, cap_x_df):
    # used for both regression and classification

    print('\nbest_model_hyperparameters:\n')
    for hyperparameter, value in grid_search_cross_val.best_params_.items():
        print(f'{hyperparameter}: {value}')

    check_out_vifs_of_preprocessed_design_matrices_of_a_collection_of_trained_estimators(
        {estimator_names[i]: grid_search_cross_val.best_estimator_},
        cap_x_df,
        data_set_type='train',
        model_selection_stage='tuned_instantiation'
    )


def flexibility_plot_class(a_gs_cv_results, an_estimator_name, class_eval_dict, big_is_good=True):

    # get flexibility plot y axis label
    y_label = class_eval_dict['scoring']

    # sort by train score according to ascending and label with index for plotting - trim data frame attributes
    ascending = False
    if big_is_good:
        ascending = True
    a_gs_cv_results = a_gs_cv_results.sort_values('mean_train_score', ascending=ascending).reset_index(drop=True).\
        reset_index()
    a_gs_cv_results = a_gs_cv_results[['index', 'rank_test_score', 'mean_train_score', 'mean_test_score']]

    # get best index - best estimator
    best_index = a_gs_cv_results.loc[a_gs_cv_results['rank_test_score'] == 1, 'index'].values[0]
    print(f'\nbest_index: {best_index}')

    # plot the flexibility curve
    sns.scatterplot(x='index', y='mean_train_score', data=a_gs_cv_results, label='mean_train_score')
    sns.scatterplot(x='index', y='mean_test_score', data=a_gs_cv_results, label='mean_test_score')
    plt.axvline(x=best_index)
    plt.title(f'{an_estimator_name} flexibility plot')
    plt.xlabel('flexibility')
    plt.ylabel(y_label)
    plt.legend()

    # make index an integer on plot
    new_list = range(math.floor(min(a_gs_cv_results.index)), math.ceil(max(a_gs_cv_results.index)) + 1)
    if 10 <= len(new_list) < 100:
        skip = 10
    elif 100 <= len(new_list) < 1000:
        skip = 100
    else:
        skip = 500
    plt.xticks(np.arange(min(new_list), max(new_list) + 1, skip))

    plt.grid()
    plt.show()

    return a_gs_cv_results, best_index


def plot_flexibility_and_print_gs_cv_results_class(grid_search_cross_val, estimator_list, i, class_eval_dict,
                                                   print_gs_cv_results):

    gs_cv_results = pd.DataFrame(grid_search_cross_val.cv_results_).sort_values('rank_test_score')

    gs_cv_results, best_index = flexibility_plot_class(gs_cv_results, estimator_list[i], class_eval_dict)

    print('\n', gs_cv_results[gs_cv_results['rank_test_score'] == 1], sep='')

    if print_gs_cv_results:
        print('\n\n', '*' * 60, sep='')
        print('grid search results (each row is a candidate):\n')
        print(gs_cv_results)
        print('*' * 60, '\n\n', sep='')

    return gs_cv_results, best_index


def grid_search_cv_helper_regression(estimator_names, i, grid_search_cross_val, cap_x_df, fit_flex_curve, y_df,
                                     target_attr, a_df_row_dict_list, print_gs_cv_results=False):

    # results utilities
    _, best_index_candidates = \
        plot_flexibility_and_print_gs_cv_results_regr(grid_search_cross_val, estimator_names, i, fit_flex_curve,
                                                      print_gs_cv_results=print_gs_cv_results)

    # collect results
    a_df_row_dict_list = collect_grid_search_cv_results_regr(grid_search_cross_val, i, estimator_names, cap_x_df, y_df,
                                                             target_attr, a_df_row_dict_list)

    return best_index_candidates, a_df_row_dict_list


def grid_search_cv_helper_classification(estimator_list, i, grid_search_cross_val, cap_x_df, y_df, a_df_row_dict_list,
                                         class_eval_dict, print_gs_cv_results):

    # results utilities
    _, best_index = \
        plot_flexibility_and_print_gs_cv_results_class(grid_search_cross_val, estimator_list, i, class_eval_dict,
                                                       print_gs_cv_results)

    # collect results
    a_df_row_dict_list = \
        collect_grid_search_cv_results_class(grid_search_cross_val, i, estimator_list, cap_x_df, y_df,
                                             a_df_row_dict_list, class_eval_dict)

    return best_index, a_df_row_dict_list


def grid_search_cv_helper(estimator_names, i, grid_search_cross_val, cap_x_df, print_gs_cv_results, y_df,
                          target_attr, a_df_row_dict_list, fit_flex_curve, prediction_task_type, class_eval_dict):

    if prediction_task_type == 'regression':

        best_index_candidates, a_df_row_dict_list = \
            grid_search_cv_helper_regression(estimator_names, i, grid_search_cross_val, cap_x_df, fit_flex_curve, y_df,
                                             target_attr, a_df_row_dict_list, print_gs_cv_results)

    elif prediction_task_type == 'classification':

        best_index, a_df_row_dict_list = \
            grid_search_cv_helper_classification(estimator_names, i, grid_search_cross_val, cap_x_df, y_df,
                                                 a_df_row_dict_list, class_eval_dict, print_gs_cv_results)
        best_index_candidates = [best_index]

    else:
        print(f'{prediction_task_type} is not a recognized prediction_task_type')
        sys.exit()

    grid_search_cv_helper_common(i, grid_search_cross_val, estimator_names, cap_x_df)

    return a_df_row_dict_list, best_index_candidates


def grid_search_cv_wrapper(estimator_names, experiment_dict, numerical_attr, nominal_attr, cap_x_df, y_df,
                           target_attr, perm_imp_dict=None, print_gs_cv_results=False, fit_flex_curve=False,
                           prediction_task_type='regression', class_eval_dict=None):

    # estimator_attr_dict = None
    # if perm_imp_dict is not None:
    #     estimator_attr_dict = trim_attrs_using_perm_imp_dict(numerical_attr, nominal_attr, perm_imp_dict)

    i = -1
    a_df_row_dict_list = []
    best_candidates_dict = {}
    for estimator, param_grid in experiment_dict.items():

        print('\n', '*' * 80, sep='')
        i += 1
        print(estimator_names[i])

        # if perm_imp_dict is not None:
        #     numerical_attr, nominal_attr, non_ml_attr = \
        #         unpack_estimator_attr_dict(estimator_attr_dict, estimator_names[i])

        # build the composite estimator
        if prediction_task_type == 'regression':
            preprocessor = build_preprocessing_pipeline(numerical_attr, nominal_attr, 'continuous')
        elif prediction_task_type == 'classification':
            preprocessor = get_class_preproc_pipeline(numerical_attr, nominal_attr, class_eval_dict)
        else:
            print(f'{prediction_task_type} is not a recognized prediction_task_type')
            sys.exit()
        composite_estimator = Pipeline(steps=[('preprocessor', preprocessor), ('estimator', estimator)])

        grid_search_cross_val = run_grid_search_cv(composite_estimator, param_grid, cap_x_df, y_df, target_attr,
                                                   prediction_task_type, class_eval_dict)

        a_df_row_dict_list, best_index_candidates = \
            grid_search_cv_helper(estimator_names, i, grid_search_cross_val, cap_x_df, print_gs_cv_results, y_df,
                                  target_attr, a_df_row_dict_list, fit_flex_curve, prediction_task_type,
                                  class_eval_dict)

        best_candidates_dict[estimator_names[i]] = best_index_candidates

    time.sleep(5)
    results_df = pd.DataFrame(a_df_row_dict_list)

    return results_df, best_candidates_dict


def plot_accumulated_model_coefficients(coef_results_df, conf_int_for_coef_df):

    coef_results_df = coef_results_df[coef_results_df.attr_names != 'intercept']
    model_selection_stage = coef_results_df.model_selection_stage.iloc[0]
    data_set_type = coef_results_df.data_set_type.iloc[0]

    for estimator_name in coef_results_df.estimator.unique():

        print(f'\n', '*' * 50, sep='')
        print(f'estimator: {estimator_name}; data_set_type: {data_set_type}; '
              f'model_selection_stage: {model_selection_stage}')

        temp_df = coef_results_df[coef_results_df.estimator == estimator_name]

        sns.catplot(data=temp_df, x='attr_names', y='weights', kind='box')
        plt.xticks(rotation=90)
        plt.grid()
        plt.xlabel('coefficient')
        plt.ylabel('coefficient value')
        plt.title('bootstrap estimated coefficient values')
        plt.show()

        drop_list = ['model_selection_stage', 'data_set_type', 'median', 'conf_level', 'high', 'low']
        temp_df = conf_int_for_coef_df[conf_int_for_coef_df.estimator == estimator_name].\
            drop(columns=drop_list)

        # shorten linear regressor names for pretty print out of df
        temp_df['estimator'] = temp_df['estimator'].str[:10]

        print('\nconf_int_for_coef_df:', sep='')
        print(f'these columns dropped for display: {drop_list}\n', sep='')
        print(temp_df.to_string())


def plot_and_get_conf_int_for_coefficients_for_all_models(estimator_names, results_df, train_cap_x_df, train_y_df,
                                                          validation_cap_x_df, validation_y_df, num_bs_samples=10,
                                                          model_selection_stage=None, data_set_type=None):

    bs_coefficients_df, _ = \
        execute_and_plot_bootstrap_eval_with_refit(estimator_names, results_df, train_cap_x_df, train_y_df,
                                                   validation_cap_x_df, validation_y_df,
                                                   num_bs_samples=num_bs_samples,
                                                   model_selection_stage=model_selection_stage,
                                                   data_set_type=data_set_type, plot_print=False)

    conf_int_for_coef_df = get_conf_int_for_coef_df(bs_coefficients_df)
    plot_accumulated_model_coefficients(bs_coefficients_df, conf_int_for_coef_df)


def get_conf_int_for_coef_df(results_df):

    df_row_dict_list = []
    for estimator in results_df.estimator.unique():

        temp_df = results_df.loc[results_df.estimator == estimator, :]
        for attr_name in temp_df.attr_names.unique():

            # get values of weights from bootstrapping
            attr_weight_values = temp_df.loc[temp_df.attr_names == attr_name, 'weights'].values

            # get the bootstrap conf int
            conf_level = 0.95
            results = bootstrap(data=(attr_weight_values, ), statistic=np.mean, confidence_level=conf_level,
                                random_state=42)

            # test if distribution of bootstrapped coefficients is normal
            _, p_value = normaltest(a=attr_weight_values)

            # test if the bootstrapped coefficients are statistically significantly different from 0 - H0 is = 0
            sample_size = attr_weight_values.size
            mean_ = np.mean(attr_weight_values)
            std_err_mean = np.std(attr_weight_values, ddof=1)/np.sqrt(sample_size)
            t_ = mean_/std_err_mean
            t_p_value = t.sf(np.abs(t_), sample_size-1)*2

            df_row_dict_list.append(
                {
                    'estimator': estimator,
                    'attr_name': attr_name,
                    'norm_p_val': p_value,  # tests the null hypothesis that a sample comes from a normal distribution
                    'mean': mean_,
                    'median': np.median(attr_weight_values),
                    'conf_level': conf_level,
                    'low': results.confidence_interval.low,
                    'high': results.confidence_interval.high,
                    'std_err_mean': std_err_mean,
                    't': t_,
                    'P(|t|)': t_p_value,
                    'model_selection_stage': results_df.model_selection_stage.iloc[0],
                    'data_set_type': results_df.data_set_type.iloc[0]
                }
            )

    return pd.DataFrame(df_row_dict_list)


def extract_stad_scaler_mean_and_std_dev(trained_composite_estimator):
    means = trained_composite_estimator.steps[0][1].named_transformers_['numerical']['scaler'].mean_
    variances = trained_composite_estimator.steps[0][1].named_transformers_['numerical']['scaler'].var_
    return means, variances


def extract_coefficients_and_check_extraction(estimator_names, results_df, cap_x_df):  # , non_ml_attr):

    estimator_weights_dict = {}
    list_based_estimator_weights_dict = {}
    for i, estimator_name in enumerate(estimator_names):

        if estimator_name in linear_estimator_list:

            # get the best estimator
            best_estimator = results_df.loc[results_df.estimator == estimator_name, 'best_estimator'].iloc[0]

            estimator_weights_dict, list_based_estimator_weights_dict = \
                extract_coefficients_and_check_extraction_helper(best_estimator, estimator_name, cap_x_df,
                                                                 estimator_weights_dict,
                                                                 list_based_estimator_weights_dict)

    return estimator_weights_dict, list_based_estimator_weights_dict


def extract_coefficients_and_check_extraction_helper(best_estimator, estimator, cap_x_df, estimator_weights_dict,
                                                     list_based_estimator_weights_dict):

    # use the preprocessing column transformer stage to transform the design matrix then add the 1's vector to
    # accommodate the bias weight
    cap_x = best_estimator.steps[0][1].transform(cap_x_df)
    cap_x = np.concatenate((np.ones((cap_x.shape[0], 1)), cap_x), axis=1)

    # extract the weights from the trained composite estimator
    weights = np.concatenate(
        (
            np.array(best_estimator.steps[1][1].intercept_, ndmin=1),
            best_estimator.steps[1][1].coef_
        )
    ).reshape(-1, 1)

    # get an ordered list of attributes from the trained estimator
    ordered_attr_list = extract_attrs_from_trained_estimator_and_order_them(best_estimator, cap_x_df)

    # add weights to dict as tuple - element 0 is attribute name and element 1 is attribute coefficient value
    estimator_weights_dict[estimator] = \
        {
            (attr, weight) for attr, weight in zip(['intercept'] + ordered_attr_list, weights.ravel())
        }

    list_based_estimator_weights_dict[estimator] = \
        {'attr_names': ['intercept'] + ordered_attr_list, 'weights': weights}

    # predict the hard way
    preds_the_hard_way = cap_x @ list_based_estimator_weights_dict[estimator]['weights']

    # predict the easy way
    preds_the_easy_way = best_estimator.predict(cap_x_df).reshape(-1, 1)

    # check prediction - if this test is passed then weights extracted correctly
    assert (np.allclose(preds_the_hard_way, preds_the_easy_way, rtol=1e-05, atol=1e-08, equal_nan=False))

    return estimator_weights_dict, list_based_estimator_weights_dict


def print_out_weights(estimator_names, list_based_estimator_weights_dict, perm_imp_dict=None):

    if perm_imp_dict is None:  # feature permutation importance will not factor in this printing
        perm_imp_gt_0_only = False
        # make a fake dictionary for algorithm below
        perm_imp_dict = {}
        for estimator_name in estimator_names:
            perm_imp_dict[estimator_name] = []
    else:  # feature permutation importance will factor in this printing
        perm_imp_gt_0_only = True

    for estimator_name in estimator_names:

        if estimator_name in linear_estimator_list:

            print('\n', '*' * 50, sep='')
            print(f'estimator: ', estimator_name, '\n')
            for attr_name, weight in zip(
                    list_based_estimator_weights_dict[estimator_name]['attr_names'],
                    list_based_estimator_weights_dict[estimator_name]['weights']
            ):
                if perm_imp_gt_0_only:
                    if attr_name in perm_imp_dict[estimator_name] or attr_name == 'intercept':
                        print(f'attr_name: {attr_name}; weight: {weight[0]}')
                else:
                    print(f'attr_name: {attr_name}; weight: {weight[0]}')


def perm_imp_printer_and_results_collection(perm_imp_multi, train_cap_x_df, stop_reporting_threshold, perm_imp_dict,
                                            estimator, print_perm_imp):

    if print_perm_imp:
        print('\n', 50 * '*', sep='')
        print('\nestimator: ', estimator)

    for metric in perm_imp_multi:  # cycle through metrics

        temp_metric = metric
        if metric == 'neg_mean_squared_error':
            temp_metric = 'sqrt_' + metric
        if print_perm_imp:
            print(f"\nmetric: {temp_metric}")

        perm_imp_results = perm_imp_multi[metric]
        for ii in perm_imp_results.importances_mean.argsort()[::-1]:  # sort importance values descending

            mean_minus_two_std = perm_imp_results.importances_mean[ii] - 2 * perm_imp_results.importances_std[ii]
            if mean_minus_two_std > stop_reporting_threshold:

                feature_name = train_cap_x_df.columns[ii]
                mean_ = perm_imp_results.importances_mean[ii]
                std_dev_ = perm_imp_results.importances_std[ii]

                if metric == 'neg_mean_squared_error':
                    mean_ = np.sqrt(mean_)
                    std_dev_ = np.sqrt(std_dev_)
                    perm_imp_dict[estimator].append(feature_name)

                if print_perm_imp:
                    print(
                        f"    {feature_name:<8}"
                        f" {mean_:.3f}"
                        f" +/- {std_dev_:.3f}"
                    )

    return perm_imp_dict


def organize_bs_perm_imp_results_into_df(ordered_col_dict, results_dict, estimator_names):

    results_df = pd.DataFrame()
    for estimator_name in estimator_names:

        # slice out the estimator
        temp_df = pd.DataFrame({'attribute': ordered_col_dict[estimator_name]})
        temp_df['estimator'] = estimator_name

        # initialize data collection dictionary
        temp_dict = {}
        for attr in ordered_col_dict[estimator_name]:
            temp_dict[attr] = []

        # extract the rank of the attributes discovered by permutation importance
        for bs_sample, bs_sample_results in results_dict.items():
            for i, attr in enumerate(bs_sample_results[estimator_name]):
                temp_dict[attr].append(i)

        # load the lists of ranks for each attribute into the estimator data frame
        temp_df['rank_list'] = 0
        temp_df['rank_list'] = temp_df['rank_list'].astype('object')
        for attr, rank_list in temp_dict.items():
            idx = temp_df.index[temp_df.attribute == attr].tolist()
            temp_df.at[idx[0], 'rank_list'] = rank_list

        # concat to the other estimator data frames
        results_df = pd.concat([results_df, temp_df], axis=0)

    return results_df


def permutation_importance_bootstrap(estimator_names, grid_search_cv_results_df, cap_x_df,  y_df, scoring,
                                     stop_reporting_threshold, num_bs_samples):

    results_dict = {}
    ordered_col_dict = None
    for bs_sample in range(0, num_bs_samples):

        if bs_sample > 0:  # the first is on the original cap_x_df and y_df
            temp_df = pd.concat([cap_x_df, y_df], axis=1)
            res_df, _ = afd_sample('bootstrap', temp_df, bs_sample)
            temp_train_cap_x_df = res_df.iloc[:, :-1]
            temp_train_y_df = res_df.iloc[:, -1]
        else:
            temp_train_cap_x_df = cap_x_df
            temp_train_y_df = y_df

        perm_imp_dict, ordered_col_dict = \
            permutation_importance_helper(estimator_names, grid_search_cv_results_df, temp_train_cap_x_df,
                                          temp_train_y_df, scoring, stop_reporting_threshold, print_perm_imp=False)

        results_dict[bs_sample] = perm_imp_dict

    results_df = organize_bs_perm_imp_results_into_df(ordered_col_dict, results_dict, estimator_names)

    for estimator_name in estimator_names:
        temp_df = results_df.loc[results_df.estimator == estimator_name, :]
        print('\n\n')
        print(temp_df.to_string())

    return results_df


def permutation_importance_helper(estimator_names, results_df, cap_x_df, y_df, scoring, stop_reporting_threshold=0,
                                  print_perm_imp=True):

    perm_imp_dict = {}
    ordered_col_dict = {}
    for i, estimator_name in enumerate(estimator_names):

        perm_imp_dict[estimator_name] = []
        best_estimator = results_df.loc[results_df.estimator == estimator_name, 'best_estimator'].iloc[0]

        ordered_col_list = extract_attrs_from_trained_estimator_and_order_them(best_estimator, cap_x_df)
        ordered_col_dict[estimator_name] = ordered_col_list

        temp_cap_x_df = cap_x_df.loc[:, ordered_col_list].copy()

        perm_imp_multi = permutation_importance(best_estimator, temp_cap_x_df, y_df, n_repeats=10,
                                                random_state=0, scoring=scoring)

        perm_imp_dict = perm_imp_printer_and_results_collection(perm_imp_multi, temp_cap_x_df, stop_reporting_threshold,
                                                                perm_imp_dict, estimator_name, print_perm_imp)

    return perm_imp_dict, ordered_col_dict


def execute_and_plot_bootstrap_eval_plot_print_helper(bs_results_df, model_selection_stage, data_set_type,
                                                      bs_conf_int_df, refit_status):

    sns.catplot(kind='box', x='estimator', y='rmse', data=bs_results_df)
    plt.title(f'assess model performance at the {model_selection_stage}\nmodel selection stage using {data_set_type} - '
              f'{refit_status}')
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()

    sns.catplot(kind='box', x='estimator', y='relative_rmse', data=bs_results_df)
    plt.title(f'assess model performance at the {model_selection_stage}\nmodel selection stage using {data_set_type} - '
              f'{refit_status}')
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()

    print(f'assess model performance at the {model_selection_stage} model selection stage using {data_set_type} - '
          f'{refit_status}')
    print(f'confidence level of the confidence interval: {bs_conf_int_df.conf_level.iloc[0]}')
    print_drop_list = ['model_selection_stage', 'data_set_type', 'median', 'conf_level']
    bs_conf_int_df = bs_conf_int_df.drop(columns=print_drop_list)
    print('\n', bs_conf_int_df.to_string(), sep='')

    return bs_conf_int_df


def get_conf_int_for_attr_in_bs_df(bs_df, attr_list):

    df_row_dict_list = []
    for estimator in bs_df.estimator.unique():

        temp_df = bs_df.loc[bs_df.estimator == estimator, :]

        for attr in attr_list:

            # get values of attribute from bootstrapping
            attr_values = temp_df.loc[:, attr].values

            # get the bootstrap attribute conf int
            conf_level = 0.95
            results = bootstrap(data=(attr_values, ), statistic=np.mean, confidence_level=conf_level,
                                random_state=42)

            # test if distribution of bootstrapped attribute is normal
            _, p_value = normaltest(a=attr_values)

            # log the results
            df_row_dict_list.append(
                {
                    'estimator': estimator,
                    'attr_name': attr,
                    'norm_p_val': p_value,  # tests the null hypothesis that a sample comes from a normal distribution
                    'mean': np.mean(attr_values),
                    'median': np.median(attr_values),
                    'conf_level': conf_level,
                    'low': results.confidence_interval.low,
                    'high': results.confidence_interval.high,
                    'model_selection_stage': bs_df.model_selection_stage.iloc[0],
                    'data_set_type': bs_df.data_set_type.iloc[0]
                }
            )

    return pd.DataFrame(df_row_dict_list)


def execute_and_plot_bootstrap_eval_with_refit(estimator_names, results_df, cap_x_df, y_df, validation_cap_x_df,
                                               validation_y_df, num_bs_samples, model_selection_stage, data_set_type,
                                               plot_print=True):

    bs_results_df = pd.DataFrame()
    bs_coefficients_df = pd.DataFrame()
    for i, estimator_name in enumerate(estimator_names):
        best_estimator = results_df.loc[results_df.estimator == estimator_name, 'best_estimator'].iloc[0]

        temp_bs_results_df, temp_bs_coefficients_df = \
            afd_sample_train_and_eval('bootstrap', num_bs_samples, cap_x_df, y_df, best_estimator, estimator_name,
                                      validation_cap_x_df, validation_y_df, model_selection_stage, data_set_type)
        bs_coefficients_df = pd.concat([bs_coefficients_df, temp_bs_coefficients_df], axis=0)

        temp_bs_results_df = temp_bs_results_df.drop(columns='distribution')
        temp_bs_results_df['relative_rmse'] = temp_bs_results_df['rmse'] / validation_y_df.mean().values[0]
        temp_bs_results_df['estimator'] = estimator_name
        bs_results_df = pd.concat([bs_results_df, temp_bs_results_df], axis=0)

    bs_results_df['model_selection_stage'] = model_selection_stage
    bs_results_df['data_set_type'] = data_set_type
    bs_conf_int_df = get_conf_int_for_attr_in_bs_df(bs_results_df, ['rmse', 'relative_rmse'])

    if plot_print:
        _ = execute_and_plot_bootstrap_eval_plot_print_helper(bs_results_df, model_selection_stage, data_set_type,
                                                              bs_conf_int_df, 'with refit')
    return bs_coefficients_df, bs_results_df


def neg_exp_fit_function(xdata, ydata):

    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    c = ydata[-1]
    a = ydata[0] - ydata[-1]
    b = ((ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])) / (-1.0 * a)
    p0 = [a, b, c]

    return func, p0


def extract_data_for_test_flexibility_curve_fit(a_gs_cv_results):

    xdata = a_gs_cv_results.loc[:, 'index'].values
    ydata = a_gs_cv_results.loc[:, 'mean_test_score'].values

    return xdata, ydata


def fit_flexibility_curve(a_gs_cv_results, function):

    xdata, ydata = extract_data_for_test_flexibility_curve_fit(a_gs_cv_results)

    if function == 'neg_exp_fit_function':
        func, p0 = neg_exp_fit_function(xdata, ydata)
    else:
        print(f'{function} is an unrecognized function')
        sys.exit()

    try:
        popt, _ = curve_fit(f=func, xdata=xdata, ydata=ydata, p0=p0)
    except Exception as e:
        print(e)
        popt = None

    knee_list = []
    if popt is not None:
        for s in np.logspace(0, 2.6, 10):
            knee = KneeLocator(xdata, func(xdata, *popt), S=s, curve='convex', direction='decreasing')
            knee_list.append(knee.elbow)
    knee_list = list(set([item for item in knee_list if item is not None]))

    return func, popt, xdata, knee_list


def flexibility_plot_regr(a_gs_cv_results, an_estimator_name, fit_flex_curve=False):

    # convert negative mse to rmse
    a_gs_cv_results.mean_train_score = np.sqrt(-1 * a_gs_cv_results.mean_train_score)
    a_gs_cv_results.mean_test_score = np.sqrt(-1 * a_gs_cv_results.mean_test_score)

    # sort by train score and label with index for plotting
    a_gs_cv_results = a_gs_cv_results.sort_values('mean_train_score', ascending=False).reset_index(drop=True).\
        reset_index()
    a_gs_cv_results = a_gs_cv_results[['index', 'rank_test_score', 'mean_train_score', 'mean_test_score']]

    # plot train and test rmse
    sns.scatterplot(x='index', y='mean_train_score', data=a_gs_cv_results, label='mean_train_score')
    sns.scatterplot(x='index', y='mean_test_score', data=a_gs_cv_results, label='mean_test_score')

    # plot the fitted function used to determine the optimum flexibility
    knee_list = []
    if 'forest' in an_estimator_name.lower() and fit_flex_curve:
        function = 'neg_exp_fit_function'
        func, popt, xdata, knee_list = fit_flexibility_curve(a_gs_cv_results, function)
        if popt is not None:
            plt.plot(xdata, func(xdata, *popt), 'r-', label=f'{function}')
            for i, knee in enumerate(knee_list):
                plt.axvline(x=knee_list[i])

    best_index = a_gs_cv_results.loc[a_gs_cv_results['rank_test_score'] == 1, 'index'].values[0]
    plt.axvline(x=best_index)
    best_index_candidates = knee_list + [best_index]

    print(f'\nbest_index_candidates: {best_index_candidates}')

    plt.title(f'{an_estimator_name} flexibility plot')
    plt.xlabel('flexibility')
    plt.ylabel('rmse')
    plt.legend()

    # make index an integer on plot
    new_list = range(math.floor(min(a_gs_cv_results.index)), math.ceil(max(a_gs_cv_results.index)) + 1)
    if 10 <= len(new_list) < 100:
        skip = 10
    elif 100 <= len(new_list) < 1000:
        skip = 100
    else:
        skip = 500
    plt.xticks(np.arange(min(new_list), max(new_list) + 1, skip))
    plt.grid()
    plt.show()

    return a_gs_cv_results, best_index_candidates


def eval_trained_estimator(trained_model, cap_x_df, y_df, model_selection_stage=None, estimator_name=None,
                           data_set_type=None, plot_pred_vs_actual_flag=True):

    if estimator_name is None:
        estimator_name = ''
    if model_selection_stage is None:
        model_selection_stage = ''
    if data_set_type is None:
        data_set_type = ''

    predicted = trained_model.predict(cap_x_df)
    eval_dict = dict()
    eval_dict['r_squared'] = r2_score(y_df, predicted)
    eval_dict['rmse'] = mean_squared_error(y_df, predicted, squared=False)
    eval_dict['frac_rmse'] = eval_dict['rmse'] / y_df.values.mean()
    if plot_pred_vs_actual_flag:
        print('\n', '*' * 50, sep='')
        print(f'{model_selection_stage} of the {estimator_name} estimator predicting on the {data_set_type} data set')
        plot_pred_vs_actual(predicted, y_df, estimator_name)
        print('\nr_squared:', eval_dict['r_squared'])
        print('rmse:', eval_dict['rmse'])
        print('frac_rmse:', eval_dict['frac_rmse'])

    return eval_dict


def plot_pred_vs_actual(predicted, y_df, estimator_name=None):

    if estimator_name is None:
        estimator_name = ''

    relative_rmse = mean_squared_error(y_df, predicted, squared=False) / y_df.mean()

    # TODO: Why this hack? No time now.
    try:
        relative_rmse = np.round(relative_rmse, 5)[0]
    except IndexError:
        relative_rmse = np.round(relative_rmse, 5)

    plt.scatter(y_df, predicted)
    slope = 1.0
    intercept = 0
    line_values = [slope * x_value + intercept for x_value in y_df.values]
    plt.plot(y_df, line_values, 'b')
    plt.grid()
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.title(f'{estimator_name} estimator\nrel_rmse: {relative_rmse}')
    plt.show()


def check_for_false_discoveries(estimator_names, cap_x_df, y_df, validation_cap_x_df, validation_y_df, results_df,
                                model_selection_stage=None, data_set_type=None, n_bootstrap_samples=10,
                                n_target_rands=10):

    if model_selection_stage is None:
        model_selection_stage = ''
    if data_set_type is None:
        data_set_type = ''

    afd_param_dict = {'n_bootstrap_samples': n_bootstrap_samples, 'n_target_rands': n_target_rands}

    for i, estimator_name in enumerate(estimator_names):
        best_estimator = results_df.loc[results_df.estimator == estimator_name, 'best_estimator'].iloc[0]
        _ = check_for_false_discovery(afd_param_dict, cap_x_df, y_df, best_estimator, estimator_name,
                                      validation_cap_x_df, validation_y_df, model_selection_stage, data_set_type)


def check_for_false_discovery(run_param_dict, cap_x_df, y_df, estimator, estimator_name, validation_cap_x_df,
                              validation_y_df, model_selection_stage=None, data_set_type=None):

    if model_selection_stage is None:
        model_selection_stage = ''
    if data_set_type is None:
        data_set_type = ''

    # bootstrap samples - alternative distribution
    bootstrap_df, _ = \
        afd_sample_train_and_eval('bootstrap', run_param_dict['n_bootstrap_samples'], cap_x_df, y_df, estimator,
                                  estimator_name, validation_cap_x_df, validation_y_df, model_selection_stage,
                                  data_set_type)

    # randomized target samples - null distribution
    rand_perm_df, _ = afd_sample_train_and_eval('randomize_target', run_param_dict['n_target_rands'], cap_x_df, y_df,
                                                estimator, estimator_name, validation_cap_x_df, validation_y_df,
                                                model_selection_stage, data_set_type)

    # combine results and plot
    temp_results_df = pd.concat([bootstrap_df, rand_perm_df], axis=0)
    plot_title = f'false discovery check of the {model_selection_stage} of\nthe {estimator_name} estimator using the ' \
                 f'{data_set_type} data set'
    plot_null_and_alt_dist(data=temp_results_df, x='rmse', hue='distribution', x_label=data_set_type + ' rmse',
                           y_label='counts', title=plot_title)

    return temp_results_df


def afd_sample(type_of_sample, temp_df, i):

    # get the sample
    if type_of_sample == 'bootstrap':
        # get the bootstrap data set - alternative distribution
        res_df = resample(temp_df, replace=True, n_samples=None, random_state=i)
        distribution_type = 'alt'
    elif type_of_sample == 'randomize_target':
        # randomize the target attribute - null distribution_type
        res_df = temp_df.copy()
        target_name = res_df.columns[-1]
        np.random.seed(i)
        res_df[target_name] = np.random.permutation(res_df[target_name])
        distribution_type = 'null_'
    else:
        print(f'{type_of_sample} is unrecognized')
        sys.exit()

    return res_df, distribution_type


def afd_fit_and_eval(estimator, temp_train_cap_x_df, temp_train_y_df, estimator_name, validation_cap_x_df,
                     validation_y_df, model_selection_stage, data_set_type):

    estimator.fit(temp_train_cap_x_df, temp_train_y_df)
    temp_compare_df = eval_trained_estimators_in_trained_estimator_dict(
        {estimator_name: estimator},
        validation_cap_x_df,
        validation_y_df,
        plot_pred_vs_actual_flag=False,
        model_selection_stage=model_selection_stage,
        data_set_type=data_set_type
    )

    return temp_compare_df


def extract_coefficients_into_df(estimator, estimator_name, cap_x_df, model_selection_stage, data_set_type):

    estimator_weights_dict = {}
    list_based_estimator_weights_dict = {}
    _, list_based_estimator_weights_dict = \
        extract_coefficients_and_check_extraction_helper(estimator, estimator_name, cap_x_df,
                                                         estimator_weights_dict, list_based_estimator_weights_dict)

    estimator_name_key = list(list_based_estimator_weights_dict.keys())[0]

    list_based_estimator_weights_dict[estimator_name_key]['weights'] = \
        list_based_estimator_weights_dict[estimator_name_key]['weights'].ravel()

    coefficients_df = pd.DataFrame(list_based_estimator_weights_dict[estimator_name_key])
    coefficients_df['estimator'] = estimator_name
    coefficients_df['data_set_type'] = data_set_type
    coefficients_df['model_selection_stage'] = model_selection_stage

    return coefficients_df


def execute_and_plot_bootstrap_eval_without_refit(estimator_names, results_df, test_cap_x_df, test_y_df,
                                                  num_bs_samples=10, model_selection_stage=None, data_set_type=None,
                                                  file_name=None, date_time_prefix=None, save_results=False,
                                                  prediction_task_type='regression', class_eval_dict=None):
    if model_selection_stage is None:
        model_selection_stage = ''
    if data_set_type is None:
        data_set_type = ''
    if file_name is None and save_results:
        file_name = 'bs_wo_refit' + '_' + model_selection_stage + '_' + data_set_type + '.csv'

    temp_test_cap_x_df = test_cap_x_df.copy()
    temp_test_y_df = test_y_df.copy()
    temp_df = pd.concat([temp_test_cap_x_df, temp_test_y_df], axis=1)

    bs_results_df = pd.DataFrame()
    for estimator_name in estimator_names:

        best_estimator = results_df.loc[results_df.estimator == estimator_name, 'best_estimator'].iloc[0]
        df_row_dict_list = []
        for bs_sample in range(0, num_bs_samples):
            res_df, _ = afd_sample('bootstrap', temp_df, bs_sample)
            temp_cap_x_df = res_df.iloc[:, :-1]
            temp_y_df = res_df.iloc[:, -1]

            if prediction_task_type == 'regression':
                eval_dict = eval_trained_estimator(best_estimator, temp_cap_x_df, temp_y_df, model_selection_stage,
                                                   estimator_name, data_set_type, plot_pred_vs_actual_flag=False)
            elif prediction_task_type == 'classification':
                eval_dict = \
                    eval_trained_estimator_class(best_estimator, temp_cap_x_df, temp_y_df, model_selection_stage,
                                                 estimator_name, data_set_type, class_eval_dict,
                                                 print_eval_results=False)
            else:
                print(f'{prediction_task_type} is not a recognized prediction_task_type')
                sys.exit()

            eval_dict['estimator'] = estimator_name
            eval_dict['data_set_type'] = data_set_type
            eval_dict['model_selection_stage'] = model_selection_stage
            df_row_dict_list.append(eval_dict)

        bs_results_df = pd.concat([bs_results_df, pd.DataFrame(df_row_dict_list)], axis=0)

    if prediction_task_type == 'regression':
        bs_results_df = bs_results_df.rename(columns={'frac_rmse': 'relative_rmse'})
        bs_conf_int_df = get_conf_int_for_attr_in_bs_df(bs_results_df, ['rmse', 'relative_rmse'])
        bs_conf_int_df = \
            execute_and_plot_bootstrap_eval_plot_print_helper(bs_results_df, model_selection_stage, data_set_type,
                                                              bs_conf_int_df, 'without refit')
    elif prediction_task_type == 'classification':
        bs_conf_int_df = get_conf_int_for_attr_in_bs_df(bs_results_df, ['ave_precision_score', 'roc_auc_score_'])
        bs_conf_int_df = \
            execute_and_plot_bootstrap_eval_plot_print_helper_class(bs_results_df, model_selection_stage, data_set_type,
                                                                    bs_conf_int_df, 'without refit')
    else:
        print(f'{prediction_task_type} is not a recognized prediction_task_type')
        sys.exit()

    if save_results:
        save_df_to_csv_with_time_stamp(bs_conf_int_df, file_name, file_path=None, date_time_prefix=date_time_prefix)

    return bs_results_df


def execute_and_plot_bootstrap_eval_plot_print_helper_class(bs_results_df, model_selection_stage, data_set_type,
                                                            bs_conf_int_df, refit_status):

    sns.catplot(kind='box', x='estimator', y='ave_precision_score', data=bs_results_df)
    plt.title(f'assess model performance at the {model_selection_stage}\nmodel selection stage using {data_set_type} - '
              f'{refit_status}')
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()

    sns.catplot(kind='box', x='estimator', y='roc_auc_score_', data=bs_results_df)
    plt.title(f'assess model performance at the {model_selection_stage}\nmodel selection stage using {data_set_type} - '
              f'{refit_status}')
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()

    print(f'assess model performance at the {model_selection_stage} model selection stage using {data_set_type} - '
          f'{refit_status}')
    print(f'confidence level of the confidence interval: {bs_conf_int_df.conf_level.iloc[0]}')
    print_drop_list = ['model_selection_stage', 'data_set_type', 'median', 'conf_level']
    bs_conf_int_df = bs_conf_int_df.drop(columns=print_drop_list)

    for metric in bs_conf_int_df.attr_name.unique():
        print('\n', bs_conf_int_df[bs_conf_int_df.attr_name == metric].to_string(), sep='')

    return bs_conf_int_df


def save_df_to_csv_with_time_stamp(df, file_name, file_path=None, date_time_prefix=None):

    if file_path is None:
        file_path = ''

    if date_time_prefix is None:
        now = datetime.datetime.now()
        date_time_prefix = str(now).replace('-', '_').replace(' ', '_').replace(':', '_').replace('.', '_')[:-4]
    file_name = date_time_prefix + '_' + file_name

    df.to_csv(os.path.join(file_path, file_name), index=False)


def afd_sample_train_and_eval(type_of_sample, num_resamples, cap_x_df, y_df, estimator, estimator_name,
                              validation_cap_x_df, validation_y_df, model_selection_stage=None, data_set_type=None):

    if model_selection_stage is None:
        model_selection_stage = ''
    if data_set_type is None:
        data_set_type = ''

    temp_df = pd.concat([cap_x_df, y_df], axis=1).copy()

    # bootstrap samples - alternative distribution
    distribution_type = None
    row_dict_list = []
    coefficients_df = pd.DataFrame()
    for i in range(num_resamples):

        # get the sample
        res_df, distribution_type = afd_sample(type_of_sample, temp_df, i)
        temp_train_cap_x_df = res_df.iloc[:, :-1]
        temp_train_y_df = res_df.iloc[:, -1]

        # fit and evaluate the estimator
        temp_compare_df = afd_fit_and_eval(estimator, temp_train_cap_x_df, temp_train_y_df, estimator_name,
                                           validation_cap_x_df, validation_y_df, model_selection_stage, data_set_type)

        if estimator_name in linear_estimator_list:
            # extract coefficients if linear estimator
            temp_coefficients_df = extract_coefficients_into_df(estimator, estimator_name, cap_x_df,
                                                                model_selection_stage, data_set_type)
            coefficients_df = pd.concat([coefficients_df, temp_coefficients_df], axis=0)

        # collect results for data frame
        df_row_dict = {
            'rmse': temp_compare_df.loc[temp_compare_df.estimator == estimator_name, 'rmse'].iloc[0]
        }
        row_dict_list.append(df_row_dict)

    bootstrap_df = pd.DataFrame(row_dict_list)
    bootstrap_df['distribution'] = distribution_type

    return bootstrap_df, coefficients_df


def plot_null_and_alt_dist(data, x, hue, x_label='', y_label='', title='', kde=True):

    print('\n', '*' * 50, sep='')
    print(f'means of the distributions:')
    print(data.groupby('distribution').mean())

    sns.histplot(data=data, x=x, hue=hue, bins=40, kde=kde)
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def avoiding_false_discoveries_class(estimator_names, grid_search_cv_results_df, train_cap_x_df, train_y_df,
                                     validation_cap_x_df, validation_y_df, num_samples, class_eval_dict,
                                     data_set_name=None, model_selection_stage=None):

    if model_selection_stage is None:
        model_selection_stage = ''
    if data_set_name is None:
        data_set_name = ''

    for estimator_name in estimator_names:

        print(f'\n', '*' * 60, sep='')
        print(f'', '*' * 60, sep='')
        print(f'', '*' * 60, sep='')
        print(f'', '*' * 60, sep='')
        print(estimator_name)

        best_estimator = \
            grid_search_cv_results_df.loc[grid_search_cv_results_df.estimator == estimator_name,
                                          'best_estimator'].iloc[0]
        best_estimator = clone(best_estimator)

        avoiding_false_discoveries_class_helper(best_estimator, train_cap_x_df, train_y_df, validation_cap_x_df,
                                                validation_y_df, num_samples, data_set_name, model_selection_stage,
                                                class_eval_dict, estimator_name)


def get_bootstrap_sample_class(num_samples, trained_estimator, train_cap_x_df, train_y_df, validation_cap_x_df,
                               validation_y_df, model_selection_stage, data_set_type, class_eval_dict):
    df_row_dict_list = []
    for i in range(0, num_samples):
        # TODO: I am not preserving or monitoring class imbalance here
        bs_cap_x_df, bs_y_df = class_utils.get_bootstrap_sample(train_cap_x_df, train_y_df, i)
        trained_estimator.fit(bs_cap_x_df, bs_y_df)
        results_dict = eval_trained_estimator_class(trained_estimator, validation_cap_x_df, validation_y_df,
                                                    model_selection_stage, 'trained_estimator', data_set_type,
                                                    class_eval_dict, print_eval_results=False)
        df_row_dict_list.append(
            {
                'distribution': 'bootstrap_sample',
                'ave_precision_score': results_dict['ave_precision_score'],
                'roc_auc_score_': results_dict['roc_auc_score_']
            }
        )

    bs_results_df = pd.DataFrame(df_row_dict_list)

    return bs_results_df


def get_randomized_target_sample_class(num_samples, trained_estimator, train_cap_x_df, train_y_df, validation_cap_x_df,
                                       validation_y_df, model_selection_stage, data_set_type, class_eval_dict):
    df_row_dict_list = []
    for i in range(0, num_samples):
        rt_cap_x_df, rt_y_df = class_utils.get_randomized_target_sample(train_cap_x_df, train_y_df, i)
        trained_estimator.fit(rt_cap_x_df, rt_y_df)
        results_dict = eval_trained_estimator_class(trained_estimator, validation_cap_x_df, validation_y_df,
                                                    model_selection_stage, 'trained_estimator', data_set_type,
                                                    class_eval_dict, print_eval_results=False)
        df_row_dict_list.append(
            {
                'distribution': 'randomized_target_sample',
                'ave_precision_score': results_dict['ave_precision_score'],
                'roc_auc_score_': results_dict['roc_auc_score_']
            }
        )

    rt_results_df = pd.DataFrame(df_row_dict_list)

    return rt_results_df


def avoiding_false_discoveries_class_helper(trained_estimator, train_cap_x_df, train_y_df, validation_cap_x_df,
                                            validation_y_df, num_samples, data_set_type, model_selection_stage,
                                            class_eval_dict, estimator_name):

    # get bootstrap sample - fit and eval on bootstrap samples
    bs_results_df = get_bootstrap_sample_class(num_samples, trained_estimator, train_cap_x_df, train_y_df,
                                               validation_cap_x_df, validation_y_df, model_selection_stage,
                                               data_set_type, class_eval_dict)

    # get kde of bootstrap sample

    # get randomized target sample - fit and eval on randomized target samples + kde on sample
    rt_results_df = get_randomized_target_sample_class(num_samples, trained_estimator, train_cap_x_df, train_y_df,
                                                       validation_cap_x_df, validation_y_df, model_selection_stage,
                                                       data_set_type, class_eval_dict)

    # combine the results
    results_df = pd.concat([bs_results_df, rt_results_df], axis=0)

    # get kde of randomized sample

    # calc alpha - prob of type I error

    # calc beta - prob of type II error

    # plot the histogram
    plot_title = f'false discovery check of the {model_selection_stage} of\nthe {estimator_name} estimator using the ' \
                 f'{data_set_type} data set'
    plot_null_and_alt_dist(data=results_df.drop(columns='roc_auc_score_'), x='ave_precision_score', hue='distribution',
                           x_label=data_set_type + ' ave_precision_score', y_label='counts', title=plot_title,
                           kde=False)
    plot_null_and_alt_dist(data=results_df.drop(columns='ave_precision_score'), x='roc_auc_score_', hue='distribution',
                           x_label=data_set_type + ' roc_auc_score_', y_label='counts', title=plot_title, kde=False)


def report_check_cal_split_details_save_data_sets(a_df, validation_cap_x_df, validation_y_df, cal_cap_x_df, cal_y_df):

    print(25 * '*')
    print('\na_df.shape:')
    print(a_df.shape)
    target_attr = validation_y_df.columns[0]
    print(f'\ntarget class fractional balance:\n{a_df[target_attr].value_counts(normalize=True)}', sep='')

    print('\n', 25 * '*', sep='')
    print('\nvalidation_df.csv:')
    print(validation_cap_x_df.shape, validation_y_df.shape)
    print(f'\ntarget class fractional balance:\n{validation_y_df[target_attr].value_counts(normalize=True)}', sep='')

    print('\n', 25 * '*', sep='')
    print('\ncal_df.csv:')
    print(cal_cap_x_df.shape, cal_y_df.shape)
    print(f'\ntarget class fractional balance:\n{cal_y_df[target_attr].value_counts(normalize=True)}', sep='')

    assert (list(validation_cap_x_df.index) == list(validation_y_df.index))
    assert (list(cal_cap_x_df.index) == list(cal_y_df.index))

    pd.concat([validation_cap_x_df, validation_y_df], axis=1).to_csv('validation_df.csv', index=True,
                                                                     index_label='index')
    pd.concat([cal_cap_x_df, cal_y_df], axis=1).to_csv('cal_df.csv', index=True, index_label='index')


def split_validation_for_calibration(a_df, cal_split_size, cal_split_random_state):

    # change name of original validation_df.csv to original_validation_df.csv
    os.rename('validation_df.csv', 'original_validation_df.csv')

    # split a_df into cap_x_df and y_df
    cap_x_df, y_df = a_df.iloc[:, :-1], a_df.iloc[:, -1].to_frame()

    # # split the data set - val_cal_size associated with the test set
    validation_cap_x_df, cal_cap_x_df, validation_y_df, cal_y_df = \
        train_test_split(cap_x_df, y_df, test_size=cal_split_size, random_state=cal_split_random_state, shuffle=True,
                         stratify=y_df)

    # report check split details save data sets
    report_check_cal_split_details_save_data_sets(a_df, validation_cap_x_df, validation_y_df, cal_cap_x_df, cal_y_df)

    return cal_cap_x_df, cal_y_df, validation_cap_x_df, validation_y_df


def plot_side_by_side(x_left, y_left, left_title, x_right, y_right, right_title, x_label, y_label):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    ax1.plot([0, 1], [0, 1], linestyle='--')
    ax1.plot(x_left, y_left, marker='.')
    ax1.grid()
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title(left_title)

    ax2.plot([0, 1], [0, 1], linestyle='--')
    ax2.plot(x_right, y_right, marker='.')
    ax2.grid()
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.set_title(right_title)

    plt.tight_layout()
    plt.show()


def side_by_side_calibration_curves_helper(best_estimator, estimator_name, cal_best_estimator, cal_estimator_name,
                                           validation_cap_x_df, validation_y_df, calibration_data_set_name,
                                           validation_data_set_name, model_selection_stage, method, ensemble):

    print(f'\ncalibration_data_set: {calibration_data_set_name}')
    print(f'validation_data_set: {validation_data_set_name}')
    print(f'model_selection_stage: {model_selection_stage}')
    print(f'method: {method}')
    print(f'ensemble: {ensemble}\n')

    # get reliability diagram data
    prob_true, prob_pred = class_utils.get_calibration_curve_data(best_estimator, validation_cap_x_df, validation_y_df)
    cal_prob_true, cal_prob_pred = class_utils.get_calibration_curve_data(cal_best_estimator, validation_cap_x_df,
                                                                          validation_y_df)
    # get mean calibration error
    mean_cal_error = class_utils.get_mean_cal_error(best_estimator, validation_cap_x_df, validation_y_df)
    cal_mean_cal_error = class_utils.get_mean_cal_error(cal_best_estimator, validation_cap_x_df, validation_y_df)

    # plot reliability diagrams
    plot_side_by_side(prob_pred, prob_true,
                      estimator_name + f' calibration plot\nmean_cal_error = {mean_cal_error:.3e}',
                      cal_prob_pred, cal_prob_true,
                      cal_estimator_name + f' calibration plot\nmean_cal_error = {cal_mean_cal_error:.3e}',
                      'predicted probability\nof positive class', 'observed probability\nof positive class')


def side_by_side_calibration_curves(grid_search_cv_results_df, calibrated_grid_search_cv_results_df,
                                    validation_cap_x_df, validation_y_df, calibration_data_set_name,
                                    validation_data_set_name, model_selection_stage, method, ensemble):

    all_estimator_names = list(zip(grid_search_cv_results_df.estimator.tolist(),
                                   calibrated_grid_search_cv_results_df.estimator.tolist())
                               )

    for estimator_name, cal_estimator_name in all_estimator_names:
        print(f'\n', '*' * 60, sep='')
        print(f'', '*' * 60, sep='')
        print(f'', '*' * 60, sep='')
        print(f'', '*' * 60, sep='')
        print(estimator_name, ' and ', cal_estimator_name)

        best_estimator = \
            grid_search_cv_results_df.loc[grid_search_cv_results_df.estimator == estimator_name,
                                          'best_estimator'].iloc[0]
        cal_best_estimator = \
            calibrated_grid_search_cv_results_df.loc[
                calibrated_grid_search_cv_results_df.estimator == cal_estimator_name, 'best_estimator'].iloc[0]

        side_by_side_calibration_curves_helper(best_estimator, estimator_name, cal_best_estimator, cal_estimator_name,
                                               validation_cap_x_df, validation_y_df, calibration_data_set_name,
                                               validation_data_set_name, model_selection_stage, method, ensemble)


def get_estimator_names_helper(grid_search_cv_results_df_1, grid_search_cv_results_df_2):

    all_estimator_names = list(zip(grid_search_cv_results_df_1.estimator.tolist(),
                                   grid_search_cv_results_df_2.estimator.tolist()))

    all_estimator_names = [estimator for estimator_tuple in all_estimator_names for estimator in estimator_tuple]

    return all_estimator_names


def side_by_side_estimator_eval(grid_search_cv_results_df, calibrated_grid_search_cv_results_df, validation_cap_x_df,
                                validation_y_df, calibration_data_set_name, validation_data_set_name,
                                model_selection_stage, class_eval_dict, method, ensemble):

    all_grid_search_cv_results_df = pd.concat([grid_search_cv_results_df, calibrated_grid_search_cv_results_df],
                                              axis=0)
    all_estimator_names = get_estimator_names_helper(grid_search_cv_results_df, calibrated_grid_search_cv_results_df)

    _ = \
        execute_and_plot_bootstrap_eval_without_refit(
            all_estimator_names,
            all_grid_search_cv_results_df,
            validation_cap_x_df,
            validation_y_df,
            num_bs_samples=20,
            model_selection_stage=model_selection_stage,
            data_set_type=validation_data_set_name,
            prediction_task_type='classification',
            class_eval_dict=class_eval_dict
        )

    side_by_side_calibration_curves(grid_search_cv_results_df, calibrated_grid_search_cv_results_df,
                                    validation_cap_x_df, validation_y_df, calibration_data_set_name,
                                    validation_data_set_name, model_selection_stage, method, ensemble)

    results_df = pd.DataFrame()
    return results_df


def calibrate_estimators(estimator_names, grid_search_cv_results_df, cal_cap_x_df, cal_y_df, validation_cap_x_df,
                         validation_y_df, class_eval_dict, calibration_data_set_name=None,
                         validation_data_set_name=None, model_selection_stage=None, method='sigmoid', ensemble=True):

    # method can be 'sigmoid' or 'isotonic'

    if model_selection_stage is None:
        model_selection_stage = ''
    if calibration_data_set_name is None:
        calibration_data_set_name = ''
    if validation_data_set_name is None:
        validation_data_set_name = ''

    print(f'\n', '*' * 60, sep='')
    print(f'', '*' * 60, sep='')
    print(f'', '*' * 60, sep='')
    print(f'', '*' * 60, sep='')
    print(f'calibration_data_set: {calibration_data_set_name}')
    print(f'validation_data_set: {validation_data_set_name}')
    print(f'model_selection_stage: {model_selection_stage}')
    print(f'method: {method}')
    print(f'ensemble: {ensemble}\n')

    df_row_dict_list = []
    for i, estimator_name in enumerate(estimator_names):

        best_estimator = \
            grid_search_cv_results_df.loc[grid_search_cv_results_df.estimator == estimator_name,
                                          'best_estimator'].iloc[0]
        cloned_best_estimator = clone(best_estimator)

        df_row_dict_list = \
            calibrate_estimators_helper(cloned_best_estimator, cal_cap_x_df, cal_y_df, validation_cap_x_df,
                                        validation_y_df, validation_data_set_name, model_selection_stage,
                                        class_eval_dict, estimator_name, method, ensemble, df_row_dict_list, i)

    calibrated_grid_search_cv_results_df = pd.DataFrame(df_row_dict_list)

    _ = \
        side_by_side_estimator_eval(grid_search_cv_results_df, calibrated_grid_search_cv_results_df,
                                    validation_cap_x_df, validation_y_df, calibration_data_set_name,
                                    validation_data_set_name, model_selection_stage, class_eval_dict, method, ensemble)

    return calibrated_grid_search_cv_results_df


def calibrate_estimators_helper(trained_estimator, cal_cap_x_df, cal_y_df, validation_cap_x_df, validation_y_df,
                                validation_data_set_name, model_selection_stage, class_eval_dict, estimator_name,
                                method, ensemble, df_row_dict_list, i):

    calibrated_trained_estimator = CalibratedClassifierCV(
        estimator=trained_estimator,
        method=method,
        cv=5,
        n_jobs=-1,  # ignored if cv='prefit'
        ensemble=ensemble,  # ignored if cv='prefit'
    )
    calibrated_trained_estimator.fit(cal_cap_x_df, cal_y_df.values.ravel())

    results_dict = eval_trained_estimator_class(calibrated_trained_estimator, validation_cap_x_df, validation_y_df,
                                                model_selection_stage, 'trained_estimator', validation_data_set_name,
                                                class_eval_dict, print_eval_results=False)
    df_row_dict_list.append(
        {
            'iteration': i,
            'estimator': 'cal_' + method[0:3] + '_' + str(ensemble)[0] + '_' + estimator_name,
            'ave_precision_score': results_dict['ave_precision_score'],
            'roc_auc_score_': results_dict['roc_auc_score_'],
            'best_estimator': calibrated_trained_estimator,
            'best_estimator_hyperparameters': calibrated_trained_estimator.get_params(),
        }
    )

    return df_row_dict_list


if __name__ == '__main__':
    pass
