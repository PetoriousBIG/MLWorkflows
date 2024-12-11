import numpy as np


def prep_gs_survey_results_df_for_calibration(gs_survey_results_df, scores_grouped_df):

    # fix a name
    grid_search_cv_results_df = gs_survey_results_df.copy().rename(columns={'estimator_name': 'estimator'})

    # extract things and then put them in
    list_of_best_estimator = []
    best_estimator_hyperparameters_list = []
    for index, row in grid_search_cv_results_df.iterrows():
        list_of_best_estimator.append(row['grid_search_cv'].best_estimator_)
        best_estimator_hyperparameters_list.append(row['grid_search_cv'].best_params_)
    grid_search_cv_results_df['best_estimator'] = list_of_best_estimator
    grid_search_cv_results_df['best_estimator_hyperparameters'] = best_estimator_hyperparameters_list

    # add a columns
    grid_search_cv_results_df['iteration'] = range(grid_search_cv_results_df.shape[0])
    grid_search_cv_results_df['ave_precision_score'] = np.nan
    grid_search_cv_results_df['roc_auc_score_'] = np.nan

    grid_search_cv_results_df = grid_search_cv_results_df.drop(columns=['score', 'best_test_score', 'best_index',
                                                                        'grid_search_cv', 'gs_cv_results_df'])

    # order columns
    grid_search_cv_results_df = grid_search_cv_results_df[
        [
            'iteration',
            'estimator',
            'ave_precision_score',
            'roc_auc_score_',
            'best_estimator',
            'best_estimator_hyperparameters'
        ]
    ]

    # add ave_precision_score and roc_auc_score_
    grid_search_cv_results_df = add_ave_precision_and_roc_auc(grid_search_cv_results_df, scores_grouped_df)

    return grid_search_cv_results_df


def add_ave_precision_and_roc_auc(grid_search_cv_results_df, scores_grouped_df):

    # add ave_precision_score and roc_auc_score_
    scores_grouped_df = scores_grouped_df.loc[scores_grouped_df.score_type == 'test', :].drop(columns=['score_type'])
    scores_grouped_df = scores_grouped_df.pivot(index='regressor_name', columns='score_name_', values='score')
    scores_grouped_df.index.name = None
    scores_grouped_df = scores_grouped_df.rename_axis(columns=None).reset_index().rename(columns={'index': 'estimator'})
    scores_grouped_df = scores_grouped_df.sort_values('estimator').reset_index(drop=True)
    grid_search_cv_results_df = grid_search_cv_results_df.sort_values('estimator').reset_index(drop=True)
    grid_search_cv_results_df['ave_precision_score'] = scores_grouped_df['average_precision']
    grid_search_cv_results_df['roc_auc_score_'] = scores_grouped_df['roc_auc']

    return grid_search_cv_results_df


if __name__ == '__main__':
    pass
