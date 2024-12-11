import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, roc_curve
import sys
from sklearn.preprocessing import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.base import clone


def get_pred_class_binary(trained_classifier, cap_x_df, classification_threshold=0.5):

    class_1_proba_preds = trained_classifier.predict_proba(cap_x_df)[:, 1]
    class_preds = np.where(class_1_proba_preds > classification_threshold, 1, 0)

    return class_preds, class_1_proba_preds


def get_prob_class_df(trained_classifier, cap_x_df, y_df, classification_threshold=0.5):

    class_preds, class_1_proba_preds = \
        get_pred_class_binary(trained_classifier, cap_x_df, classification_threshold=classification_threshold)

    prob_class_df = pd.DataFrame(
        {
            'class_1_proba_preds': class_1_proba_preds,
            'pred_class': class_preds,
            'actual_class': y_df,
            'classification_threshold': [classification_threshold] * len(y_df)
        },
        index=cap_x_df.index
    )

    return prob_class_df


def label_binarize_(df, target_attr, print_results=True):

    if df[target_attr].nunique() == 1:
        print(f'df[target_attr].nunique() = {df[target_attr].nunique()} - this case is not implemented')
        raise NotImplementedError()
    elif df[target_attr].nunique() == 2:
        df, lb_name_mapping = label_binarize_binary(df, target_attr, print_results=print_results)
    else:
        classes = df[target_attr].value_counts().index.tolist()
        num_label = [i for i in range(len(classes), 0, -1)]
        lb_name_mapping = dict(zip(classes, num_label))
        df[target_attr] = df[target_attr].map(lb_name_mapping)

    return df, lb_name_mapping


def label_binarize_binary(df, target_attr, neg_label=0, pos_label=1, print_results=True):

    if print_results:
        print(f'\ndf[target_attr] is a string attribute')
        print(f'\ndf.loc[0:5, target_attr]:\n{df.loc[0:4, target_attr]}', sep='')
        print(f'\n{df[target_attr].value_counts(normalize=True)}')
        print(f'\nlabel encode df[target_attr]')

    # make more abundant class the negative label and the rarer class the positive label
    neg_str_label = df[target_attr].value_counts(normalize=True).idxmax()
    if df[target_attr].value_counts(normalize=True).idxmin() == neg_str_label:  # need to break tie
        label_list = df[target_attr].unique().tolist()
        label_list.remove(neg_str_label)
        pos_str_label = label_list[0]
    else:
        pos_str_label = df[target_attr].value_counts(normalize=True).idxmin()

    lb_name_mapping = {
        neg_label: neg_str_label,
        pos_label: pos_str_label
    }

    df.loc[:, target_attr] = label_binarize(
        df[target_attr],
        classes=[
            lb_name_mapping[0],
            lb_name_mapping[1]
        ],
        neg_label=neg_label,
        pos_label=pos_label,
        sparse_output=False
    )
    df[target_attr] = df[target_attr].astype(float)

    if print_results:
        print(f'\nafter label encoding df[target_attr]')
        print(f'\n{df[target_attr].value_counts(normalize=True)}')
        print(f'\ndf.loc[0:5, target_attr]:\n{df.loc[0:4, target_attr]}', sep='')
        print(f'\nlb_name_mapping: {lb_name_mapping}', sep='')

    return df, lb_name_mapping


def get_precision_recall_df(trained_classifier, cap_x_df, y_df, prc_pos_label=None, sample_weight=None,
                            drop_intermediate=False):

    precision, recall, thresholds = precision_recall_curve(
        y_df,
        trained_classifier.predict_proba(cap_x_df)[:, 1],
        pos_label=prc_pos_label,
        sample_weight=sample_weight,
        drop_intermediate=drop_intermediate
    )

    precision_recall_df = pd.DataFrame(
        {
            'precision': precision,
            'recall': recall,
            'thresholds': np.concatenate([thresholds, np.array([np.nan])])
        }
    )

    return precision_recall_df, precision, recall, thresholds


def get_threshold_for_precision_or_recall_prc(trained_classifier, cap_x_df, y_df, metric=None, metric_value=None,
                                              prc_pos_label=None, sample_weight=None, drop_intermediate=False):
    """
    precision/recall curve

    :param trained_classifier:
    :param cap_x_df:
    :param y_df:
    :param metric: 'precision' or 'recall'
    :param metric_value: float between 0 and 1 inclusive
    :param prc_pos_label:
    :param sample_weight:
    :param drop_intermediate:
    :return:
    """

    precision_recall_df, _, _, _ = \
        get_precision_recall_df(trained_classifier, cap_x_df, y_df, prc_pos_label=prc_pos_label,
                                sample_weight=sample_weight, drop_intermediate=drop_intermediate)

    closest_index = (precision_recall_df[metric] - metric_value).abs().idxmin()
    threshold = precision_recall_df.loc[closest_index, 'thresholds']

    return threshold


def get_roc_df(trained_classifier, cap_x_df, y_df, pos_label=None, roc_curve_sample_weight=None,
               drop_intermediate=True):

    fpr, tpr, thresholds = roc_curve(
        y_true=y_df,
        y_score=trained_classifier.predict_proba(cap_x_df)[:, 1],
        pos_label=pos_label,
        sample_weight=roc_curve_sample_weight,
        drop_intermediate=drop_intermediate
    )

    roc_df = pd.DataFrame(
        {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
    )

    return roc_df, fpr, tpr


def get_threshold_for_precision_or_recall_roc(trained_classifier, cap_x_df, y_df, metric=None, metric_value=None,
                                              prc_pos_label=None, prc_sample_weight=None, prc_drop_intermediate=False,
                                              roc_pos_label=None, roc_curve_sample_weight=None,
                                              roc_drop_intermediate=True):

    # :param metric: 'precision' or 'recall'
    # :param metric_value: float between 0 and 1 inclusive

    roc_df, _, _ = \
            get_roc_df(trained_classifier, cap_x_df, y_df, pos_label=roc_pos_label,
                       roc_curve_sample_weight=roc_curve_sample_weight, drop_intermediate=roc_drop_intermediate)

    if metric == 'precision':
        precision_threshold = \
            get_threshold_for_precision_or_recall_prc(trained_classifier, cap_x_df, y_df, metric=metric,
                                                      metric_value=metric_value, prc_pos_label=prc_pos_label,
                                                      sample_weight=prc_sample_weight,
                                                      drop_intermediate=prc_drop_intermediate)
        closest_index = (roc_df['thresholds'] - precision_threshold).abs().idxmin()
        tpr = roc_df.loc[closest_index, 'tpr']
        fpr = roc_df.loc[closest_index, 'fpr']
        results_dict = {
            'precision': metric_value,
            'tpr_recall': tpr,
            'fpr': fpr,
            'threshold': precision_threshold
        }
    elif metric == 'recall':
        closest_index = (roc_df['tpr'] - metric_value).abs().idxmin()
        fpr = roc_df.loc[closest_index, 'fpr']
        threshold = roc_df.loc[closest_index, 'thresholds']
        precision_recall_df, _, _, _ = \
            get_precision_recall_df(trained_classifier, cap_x_df, y_df, prc_pos_label=prc_pos_label,
                                    sample_weight=prc_sample_weight, drop_intermediate=prc_drop_intermediate)
        precision, _ = get_precision_and_recall_for_threshold(precision_recall_df, threshold)
        results_dict = {
            'precision': precision,
            'tpr_recall': metric_value,
            'fpr': fpr,
            'threshold': threshold
        }
    else:
        print(f'{metric} is not a recognized metric')
        sys.exit()

    return results_dict


def get_precision_and_recall_for_threshold(precision_recall_df, threshold):
    closest_index = (precision_recall_df.thresholds - threshold).abs().idxmin()
    thresh_precision = precision_recall_df.loc[closest_index, 'precision']
    thresh_recall = precision_recall_df.loc[closest_index, 'recall']
    return thresh_precision, thresh_recall


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

    return {
        'attr_order': numerical_attr + nominal_attr,
        'preproc_cap_x_df': preproc_cap_x_df
    }


def get_model_specific_cv(model_type, cap_c_s=10, cv_folds=20, penalty='l2', scoring=None, solver='lbfgs', max_iter=100,
                          class_weight=None, l1_ratio_list=None):

    if model_type == 'LogisticRegressionCV':

        model_cv = LogisticRegressionCV(
            Cs=cap_c_s,  # grid of Cs values are chosen in a logarithmic scale between 1e-4 and 1e4
            fit_intercept=True,
            cv=cv_folds,
            dual=False,
            penalty=penalty,
            scoring=scoring,
            solver=solver,
            tol=0.0001,
            max_iter=max_iter,
            class_weight=class_weight,
            n_jobs=None,
            verbose=0,
            refit=True,
            intercept_scaling=1.0,
            random_state=42,
            l1_ratios=l1_ratio_list
        )

    else:

        sys.exit(f'model_type = {model_type} is not recognized')

    return model_cv


def load_cv_coef_paths_into_df(cap_c_s_, coefs_paths_, attr_order, coef_, cap_c_, model_type, plot=True):

    cv_coef_paths_df = pd.DataFrame()

    for cv_split in range(coefs_paths_.shape[0]):
        for cap_c_s, coef__ in zip(cap_c_s_, coefs_paths_[cv_split]):

            array_length = len(coef__[:-1])

            temp_df = pd.DataFrame(
                {
                    # 'cv_split': [cv_split] * array_length,
                    'cap_c': [cap_c_s] * array_length,
                    'coef_value': coef__[:-1],
                    'coef_name': attr_order
                }
            )
            cv_coef_paths_df = pd.concat([cv_coef_paths_df, temp_df], axis=0)

    if plot:
        # plot the coef paths
        print('\n\n')
        fig, ax = plt.subplots()
        sns.lineplot(data=cv_coef_paths_df, x='cap_c', y='coef_value', hue='coef_name')
        ax.set_xscale('log')
        plt.axvline(x=cap_c_, c='k', linestyle='--', label='best cap_c_')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f"{model_type} coefficients as a function of the regularization\n"
                  f"for {coefs_paths_.shape[0]} folds of cross validation")
        plt.grid()
        plt.show()

        print(f'\ncoef_:\n{coef_[0]}')

    return cv_coef_paths_df


def get_cv_scores_df(cap_c_s_, cv_scores_, cap_c_, model_type, coefs_paths_, plot=True):

    cv_scores_df = pd.DataFrame()
    num_cv_splits = cv_scores_.shape[0]

    for cv_split in range(num_cv_splits):

        cv_scores = cv_scores_[cv_split, :]

        temp_df = pd.DataFrame(
            {
                'cap_c': cap_c_s_,
                'log_loss': -1 * cv_scores,
            }
        )
        cv_scores_df = pd.concat([cv_scores_df, temp_df], axis=0)

    cv_scores_df = cv_scores_df.reset_index(drop=True)

    if plot:
        # plot the cv scores
        print('\n')
        fig, ax = plt.subplots()
        sns.lineplot(data=cv_scores_df, x='cap_c', y='log_loss', ax=ax)
        ax.set_xscale('log')
        plt.axvline(x=cap_c_, c='k', linestyle='--', label='best cap_c_')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f'cv_scores for {num_cv_splits} folds of cross validation')
        plt.title(f"{model_type} mean log_loss as a function of the regularization\n"
                  f"for {coefs_paths_.shape[0]} folds of cross validation")
        plt.grid()
        plt.show()

    return cv_scores_df


def print_out_coef_dict_utility(coef_dict):
    for attr_name, coefficient_value in coef_dict.items():
        print(f'   attribute {attr_name} coefficient: {coefficient_value}')


def count_non_zero_coef(coef_list):
    return len([coef for coef in list(coef_list) if coef == 0])


def get_cv_scores(estimator, cap_x_df, y_df, scoring, cv=5):

    cv_scores = cross_validate(
        estimator=estimator,
        X=cap_x_df,
        y=y_df.values.ravel(),
        groups=None,
        scoring=scoring,
        cv=cv,
        n_jobs=None,
        verbose=0,
        fit_params=None,
        params=None,
        pre_dispatch='2*n_jobs',
        return_train_score=False,
        return_estimator=False,
        return_indices=False,
        error_score=np.nan
    )

    return cv_scores['test_score']


def get_ranking_metrics(fitted_model_cv, preproc_cap_x_df, y_df, new_cap_c=None, print_=False, prev_vals_dict=None):

    # clone and configure model
    temp_fitted_model_cv = clone(fitted_model_cv)
    if new_cap_c is None:
        temp_fitted_model_cv.set_params(**{'Cs': fitted_model_cv.C_})
    else:
        temp_fitted_model_cv.set_params(**{'Cs': np.array([new_cap_c])})

    # cv with roc_auc
    cv_scores = get_cv_scores(temp_fitted_model_cv, preproc_cap_x_df, y_df, 'roc_auc', cv=5)
    cv_roc_auc_score = np.mean(cv_scores)

    # cv with average_precision
    cv_scores = get_cv_scores(temp_fitted_model_cv, preproc_cap_x_df, y_df, 'average_precision', cv=5)
    cv_average_precision_score = np.mean(cv_scores)

    cv_average_precision_score_change = None
    cv_roc_auc_score_change = None
    if prev_vals_dict is not None:
        cv_roc_auc_score_change = cv_roc_auc_score - prev_vals_dict['cv_roc_auc_score']
        cv_average_precision_score_change = cv_average_precision_score - prev_vals_dict['cv_average_precision_score']

    if print_:
        print(f'\nthe model cv average_precision_score is {cv_average_precision_score}')
        if prev_vals_dict is not None:
            print(f'   - this is a change of {cv_average_precision_score_change}')
        print(f'\nthe model cv roc_auc_score is {cv_roc_auc_score}')
        if prev_vals_dict is not None:
            print(f'   - this is a change of {cv_roc_auc_score_change}')

    return {
        'cv_roc_auc_score': cv_roc_auc_score,
        'cv_roc_auc_score_change': cv_roc_auc_score_change,
        'cv_average_precision_score': cv_average_precision_score,
        'cv_average_precision_score_change': cv_average_precision_score_change
    }


def print_out_summary_1(penalty, cv_folds, fitted_model_cv, fitted_coef_dict, l1_ratio_list, preproc_cap_x_df, y_df):

    l1_ratio = None
    if l1_ratio_list is not None:
        raise NotImplemented(f'use of l1_ratio has not been developed yet')

    print(f'\nscikit-learn model specific cv {fitted_model_cv.__class__.__name__} was fitted with penalty {penalty} '
          f'with the following results')
    print(f'\nbest regularization parameters:')
    print(f'   best C: {fitted_model_cv.C_[0]}')
    print(f'   best l1_ratio: {l1_ratio}')
    print(f'\nthe model coefficients at the best regularization hyperparameters are:')
    print_out_coef_dict_utility(fitted_coef_dict)

    print(f'\nnumber of attributes with zero coefficient: {count_non_zero_coef(fitted_coef_dict.values())}')

    return_dict = get_ranking_metrics(fitted_model_cv, preproc_cap_x_df, y_df, print_=True)

    print(f'\nthe mean test error estimate of log_loss of the {cv_folds} cross validation folds\n'
          f'along the regularization path is shown below')
    print(f'')
    print(f'')

    return return_dict


def get_new_cap_c(working_df, num_std=1.0):

    # num_std establishes how much log_loss you want to give up to get a simpler model

    # get the cap_c at which the mean test error (log_loss) is minimum
    min_mean_log_loss_cap_c = working_df.groupby('cap_c').mean().idxmin().iloc[0]

    # use the min_mean_cap_c to get the min_mean
    min_mean_log_loss = working_df.groupby('cap_c').mean().loc[min_mean_log_loss_cap_c].iloc[0]

    # use the min_mean_cap_c to get the standard deviation of min_mean
    min_mean_log_loss_std = working_df.groupby('cap_c').std().loc[min_mean_log_loss_cap_c].iloc[0]

    # under this trade what is the new mean log_loss
    trade_mean_log_loss = min_mean_log_loss + num_std * min_mean_log_loss_std

    # under this trade what is the fractional increase in log_loss
    percent_increase = (trade_mean_log_loss - min_mean_log_loss) / min_mean_log_loss

    # what is the closest cap_c to the new log_loss
    new_cap_c = working_df.groupby('cap_c').mean().sub(trade_mean_log_loss).abs().idxmin().iloc[0]

    return_dict = {
        'min_mean_log_loss_cap_c': min_mean_log_loss_cap_c,
        'min_mean_log_loss': min_mean_log_loss,
        'min_mean_log_loss_std': min_mean_log_loss_std,
        'num_std': num_std,
        'trade_mean_log_loss': trade_mean_log_loss,
        'percent_increase': percent_increase,
        'new_cap_c': new_cap_c
    }

    return return_dict


def inform_1(new_cap_c_return_dict, fitted_model_cv, preproc_cap_x_df, y_df, prev_vals_dict):

    min_mean_log_loss_cap_c = new_cap_c_return_dict['min_mean_log_loss_cap_c']
    min_mean_log_loss = new_cap_c_return_dict['min_mean_log_loss']
    min_mean_log_loss_std = new_cap_c_return_dict['min_mean_log_loss_std']
    num_std = new_cap_c_return_dict['num_std']
    trade_mean_log_loss = new_cap_c_return_dict['trade_mean_log_loss']
    percent_increase = new_cap_c_return_dict['percent_increase']
    new_cap_c = new_cap_c_return_dict['new_cap_c']

    return_dict = get_ranking_metrics(fitted_model_cv, preproc_cap_x_df, y_df, new_cap_c=new_cap_c, print_=False,
                                      prev_vals_dict=prev_vals_dict)
    cv_roc_auc_score = return_dict['cv_roc_auc_score']
    cv_roc_auc_score_change = return_dict['cv_roc_auc_score_change']
    cv_average_precision_score = return_dict['cv_average_precision_score']
    cv_average_precision_score_change = return_dict['cv_average_precision_score_change']

    print(f'')
    print(f'')
    print(f'\nnext we consider trading off log_loss to get simpler model - the plot above shows the trade space\n'
          f'\nthe minimum mean log_loss from {fitted_model_cv.__class__.__name__} is located at cap_c '
          f'{min_mean_log_loss_cap_c}\n'
          f'\nat that cap_c the minimum mean log_loss from cv is {min_mean_log_loss}\n'
          f'\nat that cap_c the standard deviation of the log_loss from cv is {min_mean_log_loss_std}\n'
          f'\nconsider trading {num_std} standard deviations of log_loss to move towards more regulation\n'
          f'and more zero coefficients\n'
          f'\nthis means moving to a new cap_c {new_cap_c} where the log_loss is {trade_mean_log_loss}\n'
          f'   - the log_loss has increased by {percent_increase}%\n'
          f'\nthe model cv average_precision_score is {cv_average_precision_score}\n'
          f'   - this is a change of {cv_average_precision_score_change}\n'
          f'\nthe model cv roc_auc_score is {cv_roc_auc_score}\n'
          f'   - this is a change of {cv_roc_auc_score_change}\n'
          f'\nthe plots below show the original cap_c from {fitted_model_cv.__class__.__name__} and the new cap_c\n'
          f'for both the coefficient paths and the mean log_loss path')
    print(f'')
    print(f'')


def new_cap_c_helper_plots(working_df, new_cap_c, fitted_model_cv, model_type, cv_coef_paths_df, coefs_paths_):

    # plot the mean mse path vs cap_c showing the min_mean_cap_c and the new_cap_c
    print('\n')
    ax = sns.lineplot(data=working_df, x='cap_c', y='log_loss', errorbar=('se', 1), label='mean_log_loss')
    ax.set(xscale='log')
    plt.axvline(x=fitted_model_cv.C_, c='k', linestyle='--', label='cap_c cv')
    plt.axvline(x=new_cap_c, c='r', linestyle='--', label='new_cap_c')
    plt.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.5))
    plt.title(f"{model_type} mean log_loss as a function of the regularization\n"
              f"for {coefs_paths_.shape[0]} folds of cross validation")
    plt.grid()
    plt.show()

    # plot the coefficient paths vs cap_c showing the min_mean_cap_c and the new_cap_c
    # ax = plt.gca()
    fig, ax = plt.subplots()
    sns.lineplot(data=cv_coef_paths_df, x='cap_c', y='coef_value', hue='coef_name', ax=ax)
    ax.set_xscale("log")
    plt.axvline(x=fitted_model_cv.C_, c='k', linestyle='--', label='cap_c cv')
    plt.axvline(x=new_cap_c, c='r', linestyle='--', label='new_cap_c')
    # plt.legend()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f"{model_type} coefficients as a function of the regularization\n"
              f"for {coefs_paths_.shape[0]} folds of cross validation")
    plt.axis("tight")
    plt.grid()
    plt.show()


def print_out_summary_3(fitted_coef_dict, new_fitted_coef_dict, new_cap_c_return_dict, fitted_model_cv,
                        preproc_cap_x_df, y_df, prev_vals_dict):

    print(f'\n')
    print(f'\n')

    min_mean_log_loss_cap_c = new_cap_c_return_dict['min_mean_log_loss_cap_c']
    percent_increase = new_cap_c_return_dict['percent_increase']
    new_cap_c = new_cap_c_return_dict['new_cap_c']

    return_dict = get_ranking_metrics(fitted_model_cv, preproc_cap_x_df, y_df, new_cap_c=new_cap_c, print_=False,
                                      prev_vals_dict=prev_vals_dict)
    cv_roc_auc_score = return_dict['cv_roc_auc_score']
    cv_roc_auc_score_change = return_dict['cv_roc_auc_score_change']
    cv_average_precision_score = return_dict['cv_average_precision_score']
    cv_average_precision_score_change = return_dict['cv_average_precision_score_change']

    print(f'by shifting from the best cap_c = {min_mean_log_loss_cap_c} given by '
          f'{fitted_model_cv.__class__.__name__} to '
          f'a new cap_c = {new_cap_c}\nwe have increased the number of zero coefficients from '
          f'{count_non_zero_coef(fitted_coef_dict.values())} to {count_non_zero_coef(new_fitted_coef_dict.values())}'
          f' giving us a simpler model\n'
          f'\nthe cost of the simpler model is a {percent_increase}% increase in log_loss\n'
          f'\nthe model cv average_precision_score is {cv_average_precision_score}\n'
          f'   - this is a change of {cv_average_precision_score_change}\n'
          f'\nthe model cv roc_auc_score is {cv_roc_auc_score}\n'
          f'   - this is a change of {cv_roc_auc_score_change}\n')

    print(f'\nthe original model coefficients were:\n')
    print_out_coef_dict_utility(fitted_coef_dict)

    print(f'\nthe new model coefficients are:\n')
    print_out_coef_dict_utility(new_fitted_coef_dict)


def model_specific_cv(cap_x_df, y_df, nominal_attr, numerical_attr, model_type, te_random_state=42, cap_c_s=10,
                      cv_folds=20, penalty='l2', scoring=None, solver='lbfgs', max_iter=100, class_weight=None,
                      l1_ratio_list=None, num_std=1.0):

    # number of log_loss standard deviations to give up for a reduction in variables

    if penalty == 'elasticnet':
        raise NotImplementedError(f'elasticnet penalty is not implemented')

    # preprocess the data
    return_dict = get_preproc_cap_x_df(cap_x_df, y_df, nominal_attr, numerical_attr, te_random_state=te_random_state)
    attr_order = return_dict['attr_order']
    preproc_cap_x_df = return_dict['preproc_cap_x_df']

    # get the model and fit it
    model_cv = get_model_specific_cv(model_type, cap_c_s=cap_c_s, cv_folds=cv_folds, penalty=penalty, scoring=scoring,
                                     solver=solver, max_iter=max_iter, class_weight=class_weight,
                                     l1_ratio_list=l1_ratio_list)
    fitted_model_cv = model_cv.fit(preproc_cap_x_df, y_df.values.ravel())

    # extract coef paths and plot them
    cv_coef_paths_df = load_cv_coef_paths_into_df(fitted_model_cv.Cs_, fitted_model_cv.coefs_paths_[1.0], attr_order,
                                                  fitted_model_cv.coef_, fitted_model_cv.C_, model_type)

    # print('\n\n')
    # print('fitted_model_cv.l1_ratios_')
    # print(fitted_model_cv.l1_ratios_)

    # load the fitted coefficients from the model specific cv into a dictionary
    fitted_coef_dict = dict(zip(attr_order, *fitted_model_cv.coef_))
    prev_vals_dict = print_out_summary_1(penalty, cv_folds, fitted_model_cv, fitted_coef_dict, l1_ratio_list,
                                         preproc_cap_x_df, y_df)

    # extract cv scores and plot
    cv_scores_df = get_cv_scores_df(fitted_model_cv.Cs_, fitted_model_cv.scores_[1.0], fitted_model_cv.C_, model_type,
                                    fitted_model_cv.coefs_paths_[1.0])

    # trade off log_loss for variable selection
    # working_df = make_working_df(fitted_model_cv)
    new_cap_c_return_dict = get_new_cap_c(cv_scores_df, num_std=num_std)

    new_cap_c = new_cap_c_return_dict['new_cap_c']

    inform_1(new_cap_c_return_dict, fitted_model_cv, preproc_cap_x_df, y_df, prev_vals_dict)

    new_cap_c_helper_plots(cv_scores_df, new_cap_c, fitted_model_cv, model_type, cv_coef_paths_df,
                           fitted_model_cv.coefs_paths_[1.0])
    new_coef_df = cv_coef_paths_df.loc[cv_coef_paths_df.cap_c == new_cap_c, :]
    new_fitted_coef_dict = dict(zip(new_coef_df.coef_name, new_coef_df.coef_value))

    print_out_summary_3(fitted_coef_dict, new_fitted_coef_dict, new_cap_c_return_dict, fitted_model_cv,
                        preproc_cap_x_df, y_df, prev_vals_dict)

    return_dict = {
        'preproc_cap_x_df': preproc_cap_x_df,
        'model_type': model_type,
        'fitted_model_cv': fitted_model_cv,
        'fitted_coef_dict': fitted_coef_dict,
        'new_fitted_coef_dict': new_fitted_coef_dict
    }

    return return_dict


if __name__ == '__main__':
    pass
