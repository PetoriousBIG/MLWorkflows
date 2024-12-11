import pandas as pd
from sklearn.model_selection import GridSearchCV
import sys
import matplotlib.pyplot as plt
import random
import numpy as np


def gather_grid_search_results_into_df(tuned_estimator_dict):

    gs_results_df = pd.DataFrame()
    for estimator_name, grid_search_cv in tuned_estimator_dict.items():
        temp_df = pd.DataFrame(grid_search_cv.cv_results_).dropna(axis=1, how='all')
        temp_df['estimator_name'] = estimator_name
        gs_results_df = pd.concat([gs_results_df, temp_df], axis=0)

    keep_attr_list = ['estimator_name', 'params']
    for attr in gs_results_df.columns:
        if (('mean_test_' in attr) or ('std_test_' in attr) or ('mean_train_' in attr) or ('std_train_' in attr) or
                ('rank_test_' in attr)):
            keep_attr_list.append(attr)

    # establish index for picking alt model
    gs_results_df = gs_results_df[keep_attr_list].reset_index(drop=True).reset_index(drop=False)

    return {
        'gs_results_df': gs_results_df
    }


def convert_param_dict_to_param_grid(params_dict):
    param_grid = {}
    for param, param_value in params_dict.items():
        param_grid[param] = [param_value]
    return param_grid


def get_alt_model_with_gs_results_df_index(gs_results_df, alt_model_idx, trained_estimator_dict, scoring, cap_x_df,
                                           y_df):

    # get the family name of the alternative model
    estimator_name = gs_results_df.loc[alt_model_idx, 'estimator_name']

    # get the params_dict of the alternative model
    params_dict = gs_results_df.loc[alt_model_idx, 'params']

    # convert param_dict to param_grid
    param_grid = convert_param_dict_to_param_grid(params_dict)

    # instantiate a model from this model family and set its params using the params_dict
    alt_model = trained_estimator_dict[estimator_name].set_params(**params_dict)

    # instantiate a GridSearchCV with the alt model and fit it
    grid_search_cv = GridSearchCV(
        estimator=alt_model,
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=-1,
        refit=scoring[0],
        cv=10,
        verbose=0,
        pre_dispatch='2*n_jobs',
        error_score=np.nan,
        return_train_score=True
    ).fit(cap_x_df, y_df.values.ravel())

    return {
        'param_grid': param_grid,
        'estimator_name': estimator_name,
        'grid_search_cv': grid_search_cv,
        'alt_model_idx': alt_model_idx
    }


def demo_reg_flow_function(gs_results_df, trained_estimator_dict, scoring, cap_x_df, y_df, estimator_name):

    # pick from model family
    temp_df = gs_results_df.loc[gs_results_df.estimator_name == estimator_name, :]

    # randomly pick from model family to demo of regression flow functionality
    random.seed(42)
    alt_model_idx = random.choice(list(temp_df.index))

    # using the alt_model_idx get the alternative model
    return_dict = get_alt_model_with_gs_results_df_index(gs_results_df, alt_model_idx, trained_estimator_dict, scoring,
                                                         cap_x_df, y_df)

    return return_dict


def is_sequence(row_num_list):

    for i in range(len(row_num_list) - 1):
        diff = row_num_list[i + 1] - row_num_list[i]
        if diff > 1:
            return False

    return True


def the_algo(frac_count, temp_df):

    # algo params - frac_count may vary with model and data

    # set max_count
    max_count = int(frac_count * temp_df.shape[0])
    print(f'\nmax_count: {max_count}\n')

    # algo
    count = 0
    algo_temp_df_index = temp_df.index[0]  # select the first index in case algo doesn't work, so we can see the plot
    alt_model_idx = None
    row_num = -1
    row_num_list = []
    for index, row in temp_df.iterrows():
        row_num += 1
        min_train = min(row['train_minus'], row['train_plus'])
        max_test = max(row['test_minus'], row['test_plus'])
        if min_train > max_test:
            count += 1
            row_num_list.append(row_num)
            if count > max_count:
                if not is_sequence(row_num_list):
                    count = 0
                    row_num_list = []
                    continue
                algo_temp_df_index = index
                alt_model_idx = row['index']
                break

    return algo_temp_df_index, alt_model_idx


def alternative_model_selection_algorithm(gs_results_df, trained_estimator_dict, scoring, cap_x_df, y_df,
                                          estimator_name, frac_count, man_flex_plot_index=None):

    # write an algorithm that returns a gs_results_df row index that will be the alternative model - use the data in
    # gs_results_df to determine the index - see how demo_reg_flow_function() works

    # pick from model family and sort into flex plot format
    temp_df = (gs_results_df.loc[gs_results_df.estimator_name == estimator_name, :].
               sort_values('mean_train_neg_log_loss', ascending=True))

    if man_flex_plot_index is None:
        algo_temp_df_index, alt_model_idx = the_algo(frac_count, temp_df)
    else:
        algo_temp_df_index = None
        alt_model_idx = temp_df.iloc[man_flex_plot_index].loc['index']

    # plot flexibility plot with bets alternative model indicated
    flexibility_best_idx = plot_minus_and_plus_bounds(temp_df, estimator_name, algo_temp_df_index,
                                                      man_flex_plot_index)

    if alt_model_idx is None:
        sys.exit(f'algo failed - train and test bands never separated')

    # using the alt_model_idx get the alternative model
    return_dict = get_alt_model_with_gs_results_df_index(gs_results_df, alt_model_idx, trained_estimator_dict, scoring,
                                                         cap_x_df, y_df)
    return_dict['flex_plot_idx'] = flexibility_best_idx

    return return_dict


def compute_minus_and_plus_bounds(gs_results_df, num_std):

    gs_results_df['train_minus'] = gs_results_df['mean_train_neg_log_loss'] - num_std * gs_results_df[
        'std_train_neg_log_loss']

    gs_results_df['train_plus'] = gs_results_df['mean_train_neg_log_loss'] + num_std * gs_results_df[
        'std_train_neg_log_loss']

    gs_results_df['test_minus'] = gs_results_df['mean_test_neg_log_loss'] - num_std * gs_results_df[
        'std_test_neg_log_loss']

    gs_results_df['test_plus'] = gs_results_df['mean_test_neg_log_loss'] + num_std * gs_results_df[
        'std_test_neg_log_loss']

    return gs_results_df


def plot_minus_and_plus_bounds(temp_df, estimator_name, algo_temp_df_index, man_flex_plot_index):

    flex_plot_x_axis = list(range(temp_df.shape[0]))
    temp_df_index = list(temp_df.index)
    index_map = dict(zip(temp_df_index, flex_plot_x_axis))

    if man_flex_plot_index is None:
        flexibility_best_idx = index_map[algo_temp_df_index]
    else:
        flexibility_best_idx = man_flex_plot_index

    fig, ax = plt.subplots(figsize=(9, 5))
    alpha = 0.2
    for tuple_ in [
        ('train_minus', 'train_plus', 'mean_train_neg_log_loss', 'blue'),
        ('test_minus', 'test_plus', 'mean_test_neg_log_loss', 'red')
    ]:
        ax.plot(flex_plot_x_axis, temp_df[tuple_[0]], color=tuple_[3], alpha=alpha, label=tuple_[0].split('_')[0])
        ax.plot(flex_plot_x_axis, temp_df[tuple_[1]], color=tuple_[3], alpha=alpha)
        ax.plot(flex_plot_x_axis, temp_df[tuple_[2]], color=tuple_[3])
        ax.fill_between(flex_plot_x_axis, temp_df[tuple_[1]], temp_df[tuple_[0]], color=tuple_[3], alpha=alpha)

    ax.axvline(x=flexibility_best_idx, color='k', linestyle='--', label='best_idx')
    ax.set_xlabel('flex_index')
    ax.set_ylabel('neg_log_loss')
    ax.set_title(f'flexibility plot for {estimator_name}')
    ax.legend()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.show()

    return flexibility_best_idx


def alternative_model_selection_algorithm_helper(tuned_estimator_dict, demo_reg_flow, trained_estimator_dict, scoring,
                                                 cap_x_df, y_df, estimator_name, frac_count, num_std=1.0,
                                                 man_flex_plot_index=None):
    # get a data frame to work with
    return_dict = gather_grid_search_results_into_df(tuned_estimator_dict)
    gs_results_df = return_dict['gs_results_df']

    # compute lower and upper bounds for train and test
    gs_results_df = compute_minus_and_plus_bounds(gs_results_df, num_std)
    # plot_minus_and_plus_bounds(gs_results_df, estimator_name)

    if demo_reg_flow:
        # demo functionality of model addition - random model choice
        return_dict = demo_reg_flow_function(gs_results_df, trained_estimator_dict, scoring, cap_x_df, y_df,
                                             estimator_name)
    else:
        return_dict = alternative_model_selection_algorithm(gs_results_df, trained_estimator_dict, scoring, cap_x_df,
                                                            y_df, estimator_name, frac_count,
                                                            man_flex_plot_index=man_flex_plot_index)

    return return_dict


def alternative_model_selection(trained_estimator_dict, tuned_estimator_dict, cap_x_df, y_df, gs_survey_results_df,
                                scoring, param_grids, demo_reg_flow, estimator, frac_count, num_std=1.0,
                                man_flex_plot_index=None, add_alternative_model_to_flow=False):

    # get the alternative model
    return_dict = alternative_model_selection_algorithm_helper(tuned_estimator_dict, demo_reg_flow,
                                                               trained_estimator_dict, scoring, cap_x_df, y_df,
                                                               estimator, frac_count, num_std=num_std,
                                                               man_flex_plot_index=man_flex_plot_index)
    estimator_name = return_dict['estimator_name']
    grid_search_cv = return_dict['grid_search_cv']
    param_grid = return_dict['param_grid']
    flex_plot_index = return_dict['flex_plot_idx']

    # add the model to tuned_estimator_dict
    estimator_name = 'Alternative' + estimator_name
    tuned_estimator_dict[estimator_name] = grid_search_cv

    # add the alt model to gs_survey_results_df
    temp_df = pd.DataFrame(grid_search_cv.cv_results_)
    temp_df['index'] = 0
    temp_df = temp_df[['index', 'rank_test_neg_log_loss', 'mean_train_neg_log_loss', 'std_train_neg_log_loss',
                       'mean_test_neg_log_loss', 'std_test_neg_log_loss']]

    print(temp_df[['mean_train_neg_log_loss', 'mean_test_neg_log_loss']])

    gs_survey_results_df = pd.concat([gs_survey_results_df, pd.DataFrame([
        {
            'estimator_name': estimator_name,
            'score': scoring[0],
            'best_test_score': -1 * temp_df.loc[0, 'mean_test_neg_log_loss'],
            'best_index': flex_plot_index,
            'grid_search_cv': grid_search_cv,
            'gs_cv_results_df': temp_df
        }
    ])], axis=0).sort_values('best_test_score')

    # update the param_grids with the params_grid of the alternative model - empty dict
    param_grids[estimator_name] = param_grid

    if add_alternative_model_to_flow:
        return {
            'tuned_estimator_dict': tuned_estimator_dict,
            'gs_survey_results_df': gs_survey_results_df,
            'param_grids': param_grids
        }
    else:
        return {}


if __name__ == "__main__":
    pass
