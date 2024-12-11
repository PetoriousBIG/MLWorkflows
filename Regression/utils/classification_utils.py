from sklearn.metrics import classification_report, confusion_matrix, PrecisionRecallDisplay, precision_recall_curve, \
    roc_curve, roc_auc_score, average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from scipy.stats import bootstrap, normaltest
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys
from sklearn.utils import resample


def label_encode(df, target_attr, print_results=True):

    print(f'\ndf[target_attr] is a string attribute')
    print(f'\ndf.loc[0:5, target_attr]:\n{df.loc[0:5, target_attr]}', sep='')
    print(f'\nlabel encode df[target_attr]')

    le = LabelEncoder()
    df[target_attr] = le.fit_transform(df[target_attr])

    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    if print_results:
        print(f'\nafter label encoding df[target_attr]')
        print(f'df.loc[0:5, target_attr]:\n{df.loc[0:5, target_attr]}', sep='')
        print(f'\ndf.head():\n{df.head()}', sep='')
        print(f'\nle_name_mapping: {le_name_mapping}', sep='')

    return df, le_name_mapping


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


def get_pred_class_binary(trained_classifier, cap_x_df, classification_threshold=0.5):

    class_1_proba_preds = trained_classifier.predict_proba(cap_x_df)[:, 1]
    class_preds = np.where(class_1_proba_preds > classification_threshold, 1, 0)

    return class_preds, class_1_proba_preds


def get_pred_class_multi_class(trained_classifier, cap_x_df):
    class_preds = trained_classifier.predict_proba(cap_x_df).argmax(axis=1)
    return class_preds


def get_classification_report_helper(y_df, class_preds, labels=None, target_names=None, sample_weight=None, digits=2,
                                     output_dict=False):

    class_report = classification_report(
        y_df,
        class_preds,
        labels=labels,
        target_names=target_names,
        sample_weight=sample_weight,
        digits=digits,
        output_dict=output_dict,
        zero_division=np.nan  # 'warn'
    )

    return class_report


def get_classification_report(trained_classifier, cap_x_df, y_df, data_set_name, model_selection_stage,
                              classification_threshold=0.5, binary=True, labels=None, target_names=None,
                              sample_weight=None, digits=2, print_class_report=True):

    if binary:
        class_preds, _ = get_pred_class_binary(trained_classifier, cap_x_df,
                                               classification_threshold=classification_threshold)
    else:
        class_preds = get_pred_class_multi_class(trained_classifier, cap_x_df)

    class_report = get_classification_report_helper(y_df, class_preds, labels=labels, target_names=target_names,
                                                    sample_weight=sample_weight, digits=digits, output_dict=False)

    class_report_dict = get_classification_report_helper(y_df, class_preds, labels=labels, target_names=target_names,
                                                         sample_weight=sample_weight, digits=digits, output_dict=True)

    if print_class_report:
        print(f'\nclassification report:')
        print(f'data_set_name: {data_set_name}; model_selection_stage: {model_selection_stage}')
        if binary:
            print(f'(note - classification threshold = {round(classification_threshold, 4)})')
        print(f'\n{class_report}')

    return class_report_dict


def get_confusion_matrix(trained_classifier, cap_x_df, y_df, data_set_name, model_selection_stage, binary=True,
                         classification_threshold=0.5, labels=None, sample_weight=None, normalize=None,
                         print_confusion_matrix=True):

    if binary:
        class_preds, _ = get_pred_class_binary(trained_classifier, cap_x_df,
                                               classification_threshold=classification_threshold)
    else:
        class_preds = get_pred_class_multi_class(trained_classifier, cap_x_df)

    conf_matrix = confusion_matrix(
        y_df,
        class_preds,
        labels=labels,
        sample_weight=sample_weight,
        normalize=normalize
    )

    if print_confusion_matrix:
        print(f'\nconfusion matrix:')
        print(f'data_set_name: {data_set_name}; model_selection_stage: {model_selection_stage}')
        if binary:
            print(f'(note - classification threshold = {round(classification_threshold, 4)})')
        print(f'\n{conf_matrix}')

    return conf_matrix


def get_cross_val_score_dict(scoring_dict, estimator, cap_x_df, y_df, cv=5, groups=None, n_jobs=-1, fit_params=None,
                             print_scores=True):

    scores_dict = {}
    for scoring_name, scoring in scoring_dict.items():

        scores_dict[scoring_name] = {}
        scores_dict[scoring_name]['scoring'] = scoring
        scores = cross_val_score(
            estimator,
            X=cap_x_df,
            y=y_df,
            groups=groups,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=0,
            fit_params=fit_params,
            pre_dispatch='2*n_jobs',
            error_score=np.nan
        )
        scores_dict[scoring_name]['scores'] = scores
        scores_dict[scoring_name]['mean'] = np.mean(scores)
        scores_dict[scoring_name]['stdev'] = np.std(scores, ddof=1)

        if print_scores:
            print(f'\n{scoring_name} cross_val_score:')
            print(f'(note - classification_threshold = 0.5)\n', sep='')
            print(f'scores: {scores}')
            print(f'np.mean(scores): {np.mean(scores)}')
            print(f'np.std(scores, ddof=1): {np.std(scores, ddof=1)}')

    return scores_dict


def get_binarize_y_and_y_score(trained_classifier, cap_x_df, y_df):

    binarize_y = label_binarize(
        y=y_df,
        classes=np.unique(y_df),
        neg_label=0,
        pos_label=1,
        sparse_output=False
    )

    y_score = trained_classifier.predict_proba(cap_x_df)

    return binarize_y, y_score


def get_multi_class_precision_recall_curves(trained_classifier, cap_x_df, y_df, data_set_name, model_selection_stage,
                                            prc_pos_label=None, sample_weight=None, drop_intermediate=False,
                                            print_prd=True):

    # https://stackoverflow.com/questions/56090541/how-to-plot-precision-and-recall-of-multiclass-classifier

    binarize_y, y_score = get_binarize_y_and_y_score(trained_classifier, cap_x_df, y_df)

    # plot precision recall curve
    precision = dict()
    recall = dict()
    thresholds = dict()
    precision_recall_df = pd.DataFrame()
    ave_precision_score_dict = dict()
    for i in range(len(np.unique(y_df))):
        precision[i], recall[i], thresholds[i] = \
            precision_recall_curve(
                binarize_y[:, i],
                y_score[:, i],
                pos_label=prc_pos_label,
                sample_weight=sample_weight,
                drop_intermediate=drop_intermediate
            )

        if print_prd:
            plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

        temp_precision_recall_df = pd.DataFrame(
            {'precision': precision[i],
             'recall': recall[i],
             'thresholds': np.concatenate([thresholds[i], np.array([np.nan])]),
             'class': [i] * len(precision[i])
             }
        )
        precision_recall_df = pd.concat([precision_recall_df, temp_precision_recall_df], axis=0)

        # TODO: bring in the parameters for average_precision_score
        ave_precision_score_dict[i] = average_precision_score(
            y_true=binarize_y[:, i],
            y_score=y_score[:, i],
            average='macro',
            pos_label=1,
            sample_weight=None
        )

    if print_prd:
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend(loc="best")
        plt.title(f'precision vs. recall curve\n'
                  f'data_set_name: {data_set_name}; model_selection_stage: {model_selection_stage}')
        plt.grid()
        plt.show()

    return precision_recall_df, ave_precision_score_dict


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


def get_threshold_for_precision_and_recall_equal_prc(trained_classifier, cap_x_df, y_df, prc_pos_label=None,
                                                     sample_weight=None, drop_intermediate=False):
    # precision/recall curve

    precision_recall_df, _, _, _ = \
        get_precision_recall_df(trained_classifier, cap_x_df, y_df, prc_pos_label=prc_pos_label,
                                sample_weight=sample_weight, drop_intermediate=drop_intermediate)

    closest_index = (precision_recall_df['precision'] - precision_recall_df['recall']).abs().idxmin()
    threshold = precision_recall_df.loc[closest_index, 'thresholds']

    precision = precision_recall_df.loc[closest_index, 'precision']
    recall = precision_recall_df.loc[closest_index, 'recall']

    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall
    }


def get_precision_and_recall_for_threshold(precision_recall_df, threshold):
    closest_index = (precision_recall_df.thresholds - threshold).abs().idxmin()
    thresh_precision = precision_recall_df.loc[closest_index, 'precision']
    thresh_recall = precision_recall_df.loc[closest_index, 'recall']
    return thresh_precision, thresh_recall


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


def get_average_precision_score(trained_classifier, cap_x_df, y_df, average='macro', pos_label=1, sample_weight=None):

    ave_precision_score = average_precision_score(
        y_true=y_df,
        y_score=trained_classifier.predict_proba(cap_x_df)[:, 1],
        average=average,
        pos_label=pos_label,
        sample_weight=sample_weight
    )

    return ave_precision_score


def get_binary_precision_recall_curves(trained_classifier, cap_x_df, y_df, data_set_name, model_selection_stage,
                                       prc_pos_label=None, sample_weight=None, drop_intermediate=False,
                                       average_precision=None, estimator_name=None, prd_pos_label=None,
                                       prevalence_pos_label=None, classification_threshold=0.5, print_prc=True,
                                       print_prd=True):

    precision_recall_df, precision, recall, thresholds = \
        get_precision_recall_df(trained_classifier, cap_x_df, y_df, prc_pos_label=prc_pos_label,
                                sample_weight=sample_weight, drop_intermediate=drop_intermediate)

    class_thresh_precision, class_thresh_recall = \
        get_precision_and_recall_for_threshold(precision_recall_df, classification_threshold)

    # TODO: bring in the parameters for average_precision_score
    ave_precision_score = get_average_precision_score(trained_classifier, cap_x_df, y_df)

    if print_prc:
        print(f'\nprecision and recall as a function of classification threshold:\n')
        plt.plot(thresholds, precision[:-1], "b--", label='precision', linewidth=2)
        plt.plot(thresholds, recall[:-1], "g-", label='recall', linewidth=2)
        plt.vlines(classification_threshold, 0, 1.0, "k", "dotted", label=f'{classification_threshold} threshold')
        plt.legend()
        plt.xlabel('classification threshold')
        plt.ylabel('precision and recall')
        print(data_set_name)
        print(model_selection_stage)
        plt.title(f'precision and recall as a function of classification threshold\n'
                  f'data_set_name: {data_set_name}; model_selection_stage: {model_selection_stage}')
        plt.grid()
        plt.show()

    if print_prd:
        print(f'\nprecision-recall curve:\n')
        disp = PrecisionRecallDisplay(precision, recall, average_precision=average_precision,
                                      estimator_name=estimator_name, pos_label=prd_pos_label,
                                      prevalence_pos_label=prevalence_pos_label)
        disp.plot()
        plt.plot(class_thresh_recall, class_thresh_precision, "ko",
                 label=f'classification threshold = {classification_threshold}')
        plt.legend()
        plt.grid()
        plt.title(f'precision-recall curve - ave_precision_score = {round(ave_precision_score, 4)}\n'
                  f'data_set_name: {data_set_name}; model_selection_stage: {model_selection_stage}')
        plt.show()

    return precision_recall_df, ave_precision_score


def get_precision_recall_curves(trained_classifier, cap_x_df, y_df, data_set_name, model_selection_stage, binary=True,
                                prc_pos_label=None, sample_weight=None, drop_intermediate=False, average_precision=None,
                                estimator_name=None, prd_pos_label=None, prevalence_pos_label=None,
                                classification_threshold=0.5, print_prc=True, print_prd=True):

    if binary:
        precision_recall_df, ave_precision_score = \
            get_binary_precision_recall_curves(trained_classifier, cap_x_df, y_df, data_set_name, model_selection_stage,
                                               prc_pos_label=prc_pos_label, sample_weight=sample_weight,
                                               drop_intermediate=drop_intermediate, average_precision=average_precision,
                                               estimator_name=estimator_name, prd_pos_label=prd_pos_label,
                                               prevalence_pos_label=prevalence_pos_label,
                                               classification_threshold=classification_threshold, print_prc=print_prc,
                                               print_prd=print_prd)
    else:
        precision_recall_df, ave_precision_score = \
            get_multi_class_precision_recall_curves(trained_classifier, cap_x_df, y_df, data_set_name,
                                                    model_selection_stage, prc_pos_label=prc_pos_label,
                                                    sample_weight=sample_weight, drop_intermediate=drop_intermediate,
                                                    print_prd=print_prd)

    return precision_recall_df, ave_precision_score


def get_tpr_and_fpr_for_threshold(roc_df, threshold):
    closest_index = (roc_df.thresholds - threshold).abs().idxmin()
    thresh_tpr = roc_df.loc[closest_index, 'tpr']
    thresh_fpr = roc_df.loc[closest_index, 'fpr']
    return thresh_tpr, thresh_fpr


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


def get_binary_roc_curve(trained_classifier, cap_x_df, y_df, data_set_name, model_selection_stage, pos_label=None,
                         roc_curve_sample_weight=None, drop_intermediate=True, average='macro',
                         roc_score_sample_weight=None, max_fpr=None, multi_class='raise', labels=None,
                         classification_threshold=0.5, print_roc=True):

    roc_df, fpr, tpr = get_roc_df(trained_classifier, cap_x_df, y_df, pos_label=pos_label,
                                  roc_curve_sample_weight=roc_curve_sample_weight, drop_intermediate=drop_intermediate)

    class_thresh_tpr, class_thresh_fpr = \
        get_tpr_and_fpr_for_threshold(roc_df, classification_threshold)

    roc_auc_score_ = roc_auc_score(
        y_true=y_df,
        y_score=trained_classifier.predict_proba(cap_x_df)[:, 1],
        average=average,
        sample_weight=roc_score_sample_weight,
        max_fpr=max_fpr,
        multi_class=multi_class,
        labels=labels
    )

    if print_roc:
        plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
        plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
        plt.plot(class_thresh_fpr, class_thresh_tpr, "ko",
                 label=f'classification threshold = {classification_threshold}')
        plt.legend()
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate (recall)')
        plt.title(f'roc curve - roc_auc_score: {round(roc_auc_score_, 4)}\n'
                  f'data_set_name: {data_set_name}; model_selection_stage: {model_selection_stage}')
        plt.grid()
        plt.show()

    return roc_df, roc_auc_score_


def make_multi_class_roc_df(i, fpr, tpr, thresholds, roc_df):

    temp_roc_df = pd.DataFrame(
        {'fpr': fpr[i],
         'tpr': tpr[i],
         'thresholds': thresholds[i],
         'class': [i] * len(fpr[i])
         }
    )

    roc_df = pd.concat([roc_df, temp_roc_df], axis=0)

    return roc_df


def get_multi_class_roc_auc_score(i, binarize_y, y_score, average='macro', roc_score_sample_weight=None, max_fpr=None,
                                  multi_class='raise', labels=None):

    roc_auc_score_ = roc_auc_score(
        y_true=binarize_y[:, i],
        y_score=y_score[:, i],
        average=average,
        sample_weight=roc_score_sample_weight,
        max_fpr=max_fpr,
        multi_class=multi_class,
        labels=labels
    )

    return roc_auc_score_


def get_fpr_tpr(i, binarize_y, y_score, pos_label=None, roc_curve_sample_weight=None, drop_intermediate=True):

    fpr, tpr, thresholds = roc_curve(
        binarize_y[:, i],
        y_score[:, i],
        pos_label=pos_label,
        sample_weight=roc_curve_sample_weight,
        drop_intermediate=drop_intermediate
    )

    return fpr, tpr, thresholds


def get_multi_class_roc_curve(trained_classifier, cap_x_df, y_df, data_set_name, model_selection_stage, pos_label=None,
                              roc_curve_sample_weight=None, drop_intermediate=True, average='macro',
                              roc_score_sample_weight=None, max_fpr=None, multi_class='raise', labels=None,
                              print_roc=True):

    binarize_y, y_score = get_binarize_y_and_y_score(trained_classifier, cap_x_df, y_df)

    # roc curve
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc_score_ = dict()
    roc_df = pd.DataFrame()
    for i in range(len(np.unique(y_df))):

        fpr[i], tpr[i], thresholds[i] = get_fpr_tpr(i, binarize_y, y_score, pos_label=pos_label,
                                                    roc_curve_sample_weight=roc_curve_sample_weight,
                                                    drop_intermediate=drop_intermediate)
        if print_roc:
            plt.plot(fpr[i], tpr[i], lw=2, label='class {}'.format(i))

        roc_auc_score_[i] = \
            get_multi_class_roc_auc_score(i, binarize_y, y_score, average=average,
                                          roc_score_sample_weight=roc_score_sample_weight, max_fpr=max_fpr,
                                          multi_class=multi_class, labels=labels)

        roc_df = make_multi_class_roc_df(i, fpr, tpr, thresholds, roc_df)

    if print_roc:
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.legend(loc="best")
        plt.title(f'ROC curve\ndata_set_name: {data_set_name}; model_selection_stage: {model_selection_stage}')
        plt.grid()
        plt.show()

    return roc_df, roc_auc_score_


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


def get_roc_curve(trained_classifier, cap_x_df, y_df, data_set_name, model_selection_stage, binary=True, pos_label=None,
                  roc_curve_sample_weight=None, drop_intermediate=True, average='macro', roc_score_sample_weight=None,
                  max_fpr=None, multi_class='raise', labels=None, classification_threshold=0.5, print_roc=True):

    if binary:
        roc_df, roc_auc_score_ = \
            get_binary_roc_curve(trained_classifier, cap_x_df, y_df, data_set_name, model_selection_stage,
                                 pos_label=pos_label, roc_curve_sample_weight=roc_curve_sample_weight,
                                 drop_intermediate=drop_intermediate, average=average,
                                 roc_score_sample_weight=roc_score_sample_weight, max_fpr=max_fpr,
                                 multi_class=multi_class, labels=labels,
                                 classification_threshold=classification_threshold, print_roc=print_roc)
    else:
        roc_df, roc_auc_score_ = \
            get_multi_class_roc_curve(trained_classifier, cap_x_df, y_df, data_set_name,
                                      model_selection_stage, pos_label=None,
                                      roc_curve_sample_weight=roc_curve_sample_weight,
                                      drop_intermediate=drop_intermediate, average=average,
                                      roc_score_sample_weight=roc_score_sample_weight, max_fpr=max_fpr,
                                      multi_class=multi_class, labels=labels, print_roc=print_roc)

    return roc_df, roc_auc_score_


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


def plot_pred_proba_hist(trained_classifier, cap_x_df, title_note):

    class_1_pre_proba = trained_classifier.predict_proba(cap_x_df)[:, 1]

    plt.hist(class_1_pre_proba, range=(0, 1), bins=20)
    plt.xlabel('class 1 .predict_proba()')
    plt.ylabel('count')
    if len(title_note) > 0:
        plt.title(f'class 1 .predict_proba() histogram\n{title_note}')
    else:
        plt.title(f'class 1 .predict_proba() histogram')
    plt.grid()
    plt.show()

    print(f'\n\n')
    sns.kdeplot(x=class_1_pre_proba, clip=[0, 1])
    plt.xlabel('class 1 .predict_proba()')
    plt.ylabel('kernel density estimation')
    if len(title_note) > 0:
        plt.title(f'class 1 .predict_proba() kde plot\n{title_note}')
    else:
        plt.title(f'class 1 .predict_proba() kde plot')
    plt.grid()
    plt.show()


def plot_pred_proba_hist_annotated_with_actual_classes(trained_classifier, cap_x_df, y_df, classification_threshold,
                                                       title_note=None):

    prob_class_df = get_prob_class_df(trained_classifier, cap_x_df, y_df,
                                      classification_threshold=classification_threshold)

    sns.histplot(data=prob_class_df, x='class_1_proba_preds', hue='actual_class', bins=20)
    plt.grid()

    if len(title_note) > 0:
        plt.title(f'class 1 .predict_proba() histogram annotated with actual class\n{title_note}')
    else:
        plt.title(f'class 1 .predict_proba() histogram annotated with actual class')

    plt.show()

    return prob_class_df


def plot_pred_proba_hist_annotated_with_predicted_classes(trained_classifier, cap_x_df, y_df, classification_threshold,
                                                          title_note=None):

    prob_class_df = get_prob_class_df(trained_classifier, cap_x_df, y_df,
                                      classification_threshold=classification_threshold)

    fig, ax1 = plt.subplots()
    matplot_axes = sns.histplot(data=prob_class_df, x='class_1_proba_preds', hue='pred_class', bins=20, ax=ax1,
                                hue_order=[0, 1])
    y1 = matplot_axes.__dict__['dataLim'].y1
    ax1.vlines(classification_threshold, 0, y1, "k", "dotted")
    legend = ax1.get_legend()
    handles = legend.legend_handles
    legend.remove()
    ax1.legend(handles, ['0', '1'])
    plt.grid()

    if len(title_note) > 0:
        ax1.set_title(f'class 1 .predict_proba() histogram annotated with\npredicted class for classification '
                      f'threshold = {classification_threshold}\n{title_note}')
    else:
        ax1.set_title(f'class 1 .predict_proba() histogram annotated with\npredicted class for classification '
                      f'threshold = {classification_threshold}')

    plt.show()

    return prob_class_df


def get_calibration_curve_data(trained_estimator, cap_x_df, y_df, n_bins=20):

    y_prob = trained_estimator.predict_proba(cap_x_df)[:, 1]

    prob_true, prob_pred = calibration_curve(
        y_true=y_df,
        y_prob=y_prob,
        pos_label=None,
        n_bins=n_bins,
        strategy='uniform'  # 'uniform' or 'quantile'
    )

    return prob_true, prob_pred


def plot_calibration_display(trained_classifier, cap_x_df, y_df, title_note=None):

    if title_note is None:
        title_note = ''

    prob_true, prob_pred = get_calibration_curve_data(trained_classifier, cap_x_df, y_df)

    mean_cal_error = get_mean_cal_error(trained_classifier, cap_x_df, y_df)

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(prob_pred, prob_true, marker='.')
    plt.xlabel('predicted probability\nof positive class')
    plt.ylabel('observed probability\nof positive class')
    plt.grid()

    if len(title_note) > 0:
        plt.title(f'probability calibration curve\n{title_note}\nmean calibration error = {mean_cal_error:.3e}')
    else:
        plt.title(f'probability calibration curve\nmean calibration error = {mean_cal_error:.3e}')

    plt.show()


def get_title_note(data_set_name=None, model_selection_stage=None):

    title_note = ''

    if data_set_name is not None:
        title_note = f'data set: {data_set_name}'

    if model_selection_stage is not None:
        title_note = title_note + '; ' + f'model selection stage: {model_selection_stage}'

    return title_note


def probability_calibration_check(trained_classifier, cap_x_df, y_df, binary=True, classification_threshold=0.5,
                                  print_pred_proba=True, print_pred_proba_actual=True, print_pred_proba_pred=True,
                                  print_calibration=True, data_set_name=None, model_selection_stage=None):

    title_note = get_title_note(data_set_name, model_selection_stage)

    prob_class_df = None

    if print_pred_proba and binary:
        plot_pred_proba_hist(trained_classifier, cap_x_df, title_note)

    if print_pred_proba_actual and binary:
        prob_class_df = plot_pred_proba_hist_annotated_with_actual_classes(trained_classifier, cap_x_df, y_df,
                                                                           classification_threshold, title_note)

    if print_pred_proba_pred and binary:
        prob_class_df = plot_pred_proba_hist_annotated_with_predicted_classes(trained_classifier, cap_x_df, y_df,
                                                                              classification_threshold, title_note)

    if print_calibration and binary:
        plot_calibration_display(trained_classifier, cap_x_df, y_df, title_note)

    return prob_class_df


def classification_performance(trained_classifier, cap_x_df, y_df, classification_threshold=0.5, binary=True,

                               cr_labels=None, cr_target_names=None, cr_sample_weight=None, cr_digits=2, cr_print=True,

                               cm_labels=None, cm_sample_weight=None, cm_normalize=None, cm_print=True,

                               cvs_scoring_dict=None, cvs_cv=5, cvs_groups=None, cvs_n_jobs=-1, cvs_fit_params=None,
                               cvs_compute=True, cvs_print=True,

                               prc_pos_label=None, prc_sample_weight=None, prc_drop_intermediate=False, prc_print=True,

                               prd_average_precision=None, prd_estimator_name=None, prd_pos_label=None,
                               prd_prevalence_pos_label=None, prd_print=True,

                               roc_pos_label=None, roc_sample_weight=None, roc_drop_intermediate=True,
                               roc_score_average='macro', roc_score_sample_weight=None, roc_score_max_fpr=None,
                               roc_score_multi_class='ovr', roc_score_labels=None, roc_print=True,

                               print_pred_proba=True, print_pred_proba_actual=True, print_pred_proba_pred=True,
                               print_calibration=True,

                               data_set_name=None, model_selection_stage=None

                               ):

    class_report = get_classification_report(trained_classifier, cap_x_df, y_df,
                                             classification_threshold=classification_threshold, binary=binary,
                                             labels=cr_labels, target_names=cr_target_names,
                                             sample_weight=cr_sample_weight, digits=cr_digits,
                                             print_class_report=cr_print, data_set_name=data_set_name,
                                             model_selection_stage=model_selection_stage)

    conf_matrix = get_confusion_matrix(trained_classifier, cap_x_df, y_df, binary=binary,
                                       classification_threshold=classification_threshold,
                                       labels=cm_labels, sample_weight=cm_sample_weight, normalize=cm_normalize,
                                       print_confusion_matrix=cm_print, data_set_name=data_set_name,
                                       model_selection_stage=model_selection_stage)

    cross_val_score_dict = None
    if cvs_scoring_dict is not None and cvs_compute:
        cross_val_score_dict = get_cross_val_score_dict(cvs_scoring_dict, trained_classifier, cap_x_df, y_df, cv=cvs_cv,
                                                        groups=cvs_groups, n_jobs=cvs_n_jobs, fit_params=cvs_fit_params,
                                                        print_scores=cvs_print)

    precision_recall_df, ave_precision_score = \
        get_precision_recall_curves(trained_classifier, cap_x_df, y_df, binary=binary, prc_pos_label=prc_pos_label,
                                    sample_weight=prc_sample_weight, drop_intermediate=prc_drop_intermediate,
                                    average_precision=prd_average_precision, estimator_name=prd_estimator_name,
                                    prd_pos_label=prd_pos_label, prevalence_pos_label=prd_prevalence_pos_label,
                                    classification_threshold=classification_threshold, print_prc=prc_print,
                                    print_prd=prd_print, data_set_name=data_set_name,
                                    model_selection_stage=model_selection_stage)

    roc_df, roc_auc_score_ = get_roc_curve(trained_classifier, cap_x_df, y_df, binary=binary, pos_label=roc_pos_label,
                                           roc_curve_sample_weight=roc_sample_weight,
                                           drop_intermediate=roc_drop_intermediate, average=roc_score_average,
                                           roc_score_sample_weight=roc_score_sample_weight, max_fpr=roc_score_max_fpr,
                                           multi_class=roc_score_multi_class, labels=roc_score_labels,
                                           classification_threshold=classification_threshold, print_roc=roc_print,
                                           data_set_name=data_set_name, model_selection_stage=model_selection_stage)

    prob_class_df = probability_calibration_check(trained_classifier, cap_x_df, y_df, binary=binary,
                                                  classification_threshold=classification_threshold,
                                                  print_pred_proba=print_pred_proba,
                                                  print_pred_proba_actual=print_pred_proba_actual,
                                                  print_pred_proba_pred=print_pred_proba_pred,
                                                  print_calibration=print_calibration, data_set_name=data_set_name,
                                                  model_selection_stage=model_selection_stage)

    return {
        'data_set_name': data_set_name,
        'model_selection_stage': model_selection_stage,
        'classification_threshold': classification_threshold,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'cross_val_score_dict': cross_val_score_dict,
        'precision_recall_df': precision_recall_df,
        'ave_precision_score': ave_precision_score,
        'roc_df': roc_df,
        'roc_auc_score': roc_auc_score_,
        'prob_class_df': prob_class_df
    }


def decision_boundary_from_estimator(trained_estimator, cap_x_df, y_df, target_attr, mesh_grid_reso=100):

    # for two predictive attributes only

    feature_1, feature_2 = np.meshgrid(
        np.linspace(cap_x_df.iloc[:, 0].min(), cap_x_df.iloc[:, 0].max(), num=mesh_grid_reso),
        np.linspace(cap_x_df.iloc[:, 1].min(), cap_x_df.iloc[:, 1].max(), num=mesh_grid_reso)
    )

    grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T
    grid_df = pd.DataFrame(
        data=grid,
        columns=cap_x_df.columns,
    )

    y_pred = np.reshape(trained_estimator.predict(grid_df), feature_1.shape)

    display = DecisionBoundaryDisplay(
        xx0=feature_1,
        xx1=feature_2,
        response=y_pred,
        xlabel=cap_x_df.columns[0],
        ylabel=cap_x_df.columns[1]
    )
    display.plot()

    display.ax_.scatter(cap_x_df.iloc[:, 0], cap_x_df.iloc[:, 1], c=y_df[target_attr], edgecolor="black")
    plt.show()


def ranking_metrics_class_perf_assess_binary(estimator_names, grid_search_cv_results_df, cap_x_df, y_df,
                                             classification_threshold=0.50, cvs_compute=False, cvs_print=False,
                                             data_set_name=None, model_selection_stage=None):

    ranking_metrics = True
    class_thresh_metrics = False

    if model_selection_stage is None:
        model_selection_stage = ''
    if data_set_name is None:
        data_set_name = ''

    class_perf_dict = None
    for estimator_name in estimator_names:
        print(f'\n', '*' * 60, sep='')
        print(f'', '*' * 60, sep='')
        print(f'', '*' * 60, sep='')
        print(f'', '*' * 60, sep='')
        print(estimator_name)

        best_estimator = \
            grid_search_cv_results_df.loc[
                grid_search_cv_results_df.estimator == estimator_name, 'best_estimator'].iloc[0]

        class_perf_dict = classification_performance(
            best_estimator,
            cap_x_df,
            y_df.values.ravel(),
            classification_threshold=classification_threshold,
            binary=True,
            # https://scikit-learn.org/stable/modules/model_evaluation.html
            cvs_scoring_dict={
                'accuracy': 'accuracy',
                'precision': 'precision',
                'recall': 'recall',
                'f1': 'f1'
            },
            cr_digits=4,
            cr_print=class_thresh_metrics,  # print classification report
            cm_print=class_thresh_metrics,  # print confusion matrix
            cvs_compute=cvs_compute,  # compute cross_val_scores (classification threshold = 0.5 always)
            cvs_print=cvs_print,  # print cross_val_scores (classification threshold = 0.5 always) - ignored if
                                  # cvs_compute=False
            prc_print=ranking_metrics,  # print precision and recall curves as a function of classification threshold
            prd_print=ranking_metrics,  # print precision recall curves
            roc_print=ranking_metrics,  # print roc curve
            print_pred_proba=ranking_metrics,  # print pred proba hist
            print_pred_proba_actual=ranking_metrics,  # print pred proba hist annotated with actual classes
            print_pred_proba_pred=ranking_metrics,  # print pred proba hist annotated with predicted classes
            print_calibration=ranking_metrics,  # print probability calibration
            data_set_name=data_set_name,
            model_selection_stage=model_selection_stage
        )

    return class_perf_dict


def class_thresh_metrics_class_perf_assess_binary(best_model_name, estimator_names, grid_search_cv_results_df,
                                                  cap_x_df, y_df, class_threshold_list, cvs_compute=False,
                                                  cvs_print=False, data_set_name=None, model_selection_stage=None):
    ranking_metrics = False
    class_thresh_metrics = True

    if model_selection_stage is None:
        model_selection_stage = ''
    if data_set_name is None:
        data_set_name = ''

    for estimator_name in estimator_names:
        if estimator_name == best_model_name:

            print(estimator_name)

            best_estimator = \
                grid_search_cv_results_df.loc[
                    grid_search_cv_results_df.estimator == estimator_name, 'best_estimator'].iloc[0]

            for class_threshold in class_threshold_list:
                print(f'\n', '*' * 60, sep='')
                print(f'', '*' * 60, sep='')

                _ = classification_performance(
                    best_estimator,
                    cap_x_df,
                    y_df.values.ravel(),
                    classification_threshold=class_threshold,
                    binary=True,
                    # https://scikit-learn.org/stable/modules/model_evaluation.html
                    cvs_scoring_dict={
                        'accuracy': 'accuracy',
                        'precision': 'precision',
                        'recall': 'recall',
                        'f1': 'f1'
                    },
                    cr_digits=4,
                    cr_print=class_thresh_metrics,  # print classification report
                    cm_print=class_thresh_metrics,  # print confusion matrix
                    cvs_compute=cvs_compute,  # compute cross_val_scores (classification threshold = 0.5 always)
                    cvs_print=cvs_print,  # print cross_val_scores (classification threshold = 0.5 always)
                                          # - ignored if cvs_compute=False
                    prc_print=ranking_metrics,  # print precision and recall curves as a function of classification
                                                # threshold
                    prd_print=ranking_metrics,  # print precision recall curves
                    roc_print=ranking_metrics,  # print roc curve
                    print_pred_proba=ranking_metrics,  # print pred proba hist
                    print_pred_proba_actual=ranking_metrics,  # print pred proba hist annotated with actual classes
                    print_pred_proba_pred=ranking_metrics,  # print pred proba hist annotated with predicted classes
                    print_calibration=ranking_metrics,  # print probability calibration
                    data_set_name=data_set_name,
                    model_selection_stage=model_selection_stage
                )


def precision_recall_bootstrap_no_refit_binary(best_model_name, estimator_names, grid_search_cv_results_df,
                                               cap_x_df, y_df, n_bootstrap=10, prc_pos_label=None,
                                               prc_sample_weight=None, prc_drop_intermediate=False, data_set_name=None,
                                               model_selection_stage=None, classification_threshold=0.50):

    if model_selection_stage is None:
        model_selection_stage = ''
    if data_set_name is None:
        data_set_name = ''

    results_df = pd.DataFrame()
    for estimator_name in estimator_names:

        if estimator_name == best_model_name:

            print(f'\n', '*' * 60, sep='')
            print(f'', '*' * 60, sep='')
            print(estimator_name)

            estimator = \
                grid_search_cv_results_df.loc[
                    grid_search_cv_results_df.estimator == estimator_name, 'best_estimator'].iloc[0]

            prc_kwargs = {
                'prc_pos_label': prc_pos_label,
                'sample_weight': prc_sample_weight,
                'drop_intermediate': prc_drop_intermediate
            }
            temp_results_df = \
                precision_recall_bootstrap_no_refit_helper(estimator, cap_x_df, y_df, prc_kwargs, n_bootstrap,
                                                           data_set_name, model_selection_stage,
                                                           classification_threshold, estimator_name)

            results_df = pd.concat([results_df, temp_results_df], axis=0)

    print(f'\n variation of precision and recall across bootstrap (no refit) samples:')
    sns.catplot(data=results_df, x='estimator_name', y='precision', kind='box')
    plt.xticks(rotation=90)
    plt.grid()
    plt.title(f'{n_bootstrap} bootstrap (no refit) samples - variation in precision\n'
              f'classification threshold: {results_df.class_threshold.iloc[0]}')
    plt.show()

    sns.catplot(data=results_df, x='estimator_name', y='recall', kind='box')
    plt.xticks(rotation=90)
    plt.grid()
    plt.title(f'{n_bootstrap} bootstrap (no refit) samples - variation in recall\n'
              f'classification threshold: {results_df.class_threshold.iloc[0]}')
    plt.show()


def precision_recall_bootstrap_no_refit_helper(trained_classifier, cap_x_df, y_df, prc_kwargs, n_bootstrap,
                                               data_set_name, model_selection_stage, classification_threshold,
                                               estimator_name):

    ave_precision_score_list = []
    class_thresh_precision_list = []
    class_thresh_recall_list = []
    df_row_dict_list = []
    for bootstrap_ in range(0, n_bootstrap):

        bs_cap_x_df, bs_y_df = get_bootstrap_sample(cap_x_df, y_df, bootstrap_)

        precision_recall_df, _, _, _ = get_precision_recall_df(trained_classifier, bs_cap_x_df, bs_y_df, **prc_kwargs)
        class_thresh_precision, class_thresh_recall = \
            get_precision_and_recall_for_threshold(precision_recall_df, classification_threshold)
        class_thresh_precision_list.append(class_thresh_precision)
        class_thresh_recall_list.append(class_thresh_recall)
        # TODO: bring in the parameters for average_precision_score
        ave_precision_score_list.append(get_average_precision_score(trained_classifier, bs_cap_x_df, bs_y_df))

        sns.lineplot(data=precision_recall_df, x='recall', y='precision')

        if bootstrap_ == 0:  # only have label on graph once
            plt.plot(class_thresh_recall, class_thresh_precision, "ko",
                     label=f'classification threshold = {classification_threshold}')
        else:
            plt.plot(class_thresh_recall, class_thresh_precision, "ko")

        df_row_dict_list.append(
            {
               'estimator_name': estimator_name,
               'precision': class_thresh_precision,
               'recall': class_thresh_recall,
               'class_threshold': classification_threshold
            }
        )

    plt.legend()
    plt.grid()
    plt.title(f'{n_bootstrap} bootstrapped precision-recall curves\n'
              f'data_set_name: {data_set_name}; model_selection_stage: {model_selection_stage}')
    plt.show()

    # get the bootstrap conf intervals
    # bootstrap_statistics_helper(ave_precision_score_list, 'ave_precision_score')
    # bootstrap_statistics_helper(class_thresh_precision_list, 'precision')
    # bootstrap_statistics_helper(class_thresh_recall_list, 'recall')

    # get basic statistics
    statistics_helper(ave_precision_score_list, 'ave_precision_score')
    statistics_helper(class_thresh_precision_list, 'precision')
    statistics_helper(class_thresh_recall_list, 'recall')

    return pd.DataFrame(df_row_dict_list)


def statistics_helper(sample_list, sample_name, plot_hist=True):

    print(f'\n', '*' * 30, sep='')
    print(sample_name)

    if plot_hist:
        print()
        # print(np.round(sample_list, 3))
        sns.histplot(sample_list)
        plt.grid()
        plt.xlabel(sample_name)
        plt.title(f'sample of {sample_name} created with 20 bootstrap samples\n '
                  f'of the class_thresh_set_cap_x_df')
        plt.show()

    # print out basic statistics
    print(f'\nmin {sample_name}: {np.min(sample_list)}')
    print(f'mean {sample_name}: {np.mean(sample_list)}')
    print(f'max {sample_name}: {np.max(sample_list)}')
    print(f'{sample_name} percentile based 95% confidence interval ranges from '
          f'{np.percentile(sample_list, [5])[0]:.4f} to {np.percentile(sample_list, [95])[0]:.4f}')


def bootstrap_statistics_helper(sample_list, sample_name, statistic=np.mean, conf_level=0.95, plot_hist=False):

    print(f'\n', '*' * 30, sep='')
    print(sample_name)

    if plot_hist:
        print(np.round(sample_list, 3))
        sns.histplot(sample_list)
        plt.grid()
        plt.xlabel(sample_name)
        plt.title(f'sample of {sample_name} created with 20 bootstrap samples\n '
                  f'of the class_thresh_set_cap_x_df')
        plt.show()

    # get the bootstrap conf int
    # TODO: there is something wrong with tis approach - the 95% confidence interval is too small
    results = bootstrap(data=(sample_list,), statistic=statistic, confidence_level=conf_level, random_state=42,
                        n_resamples=20, method='BCa')
    print(f'\n{conf_level * 100}% confidence interval for {sample_name} bootstrap sample:')
    print(f'   sample conf int low: {results.confidence_interval.low}')
    print(f'   sample mean: {np.mean(sample_list)}')
    print(f'   sample conf int high: {results.confidence_interval.high}')
    print(f'   sample conf int margin of error: '
          f'{(results.confidence_interval.high - results.confidence_interval.low)/2}')

    # test if bootstrap sample distribution was sampled from a normal distribution
    _, p_value = normaltest(a=sample_list)
    print(f'\nH0: {sample_name} sample was sampled from a normal distribution')
    print(f'H1: {sample_name} sample was not sampled from a normal distribution')
    print(f'p_value: {p_value}')


def roc_curve_bootstrap_no_refit_binary(best_model_name, estimator_names, grid_search_cv_results_df, cap_x_df, y_df,
                                        n_bootstrap=10, roc_pos_label=None, roc_sample_weight=None,
                                        roc_drop_intermediate=True, roc_score_average='macro',
                                        roc_score_sample_weight=None, roc_score_max_fpr=None,
                                        roc_score_multi_class='ovr', roc_score_labels=None, data_set_name=None,
                                        model_selection_stage=None, classification_threshold=0.50):

    if model_selection_stage is None:
        model_selection_stage = ''
    if data_set_name is None:
        data_set_name = ''

    results_df = pd.DataFrame()
    for estimator_name in estimator_names:

        if estimator_name == best_model_name:

            print(f'\n', '*' * 60, sep='')
            print(f'', '*' * 60, sep='')
            print(estimator_name)

            estimator = \
                grid_search_cv_results_df.loc[
                    grid_search_cv_results_df.estimator == estimator_name, 'best_estimator'].iloc[0]

            roc_kwargs = {
                'pos_label': roc_pos_label,
                'roc_curve_sample_weight': roc_sample_weight,
                'drop_intermediate': roc_drop_intermediate
            }
            roc_score_kwargs = {
                'average': roc_score_average,
                'labels': roc_score_labels,
                'multi_class': roc_score_multi_class,
                'max_fpr': roc_score_max_fpr,
                'sample_weight': roc_score_sample_weight
            }
            temp_results_df = \
                roc_curve_bootstrap_no_refit_helper(estimator, cap_x_df, y_df, roc_kwargs, roc_score_kwargs,
                                                    n_bootstrap, data_set_name, model_selection_stage,
                                                    classification_threshold, estimator_name)

            results_df = pd.concat([results_df, temp_results_df], axis=0)

    print(f'\n variation of tpr (recall) and fpr across bootstrap (no refit) samples:')
    sns.catplot(data=results_df, x='estimator_name', y='tpr', kind='box')
    plt.ylabel('tpr (recall)')
    plt.xticks(rotation=90)
    plt.grid()
    plt.title(f'{n_bootstrap} bootstrap (no refit) samples - variation in tpr (recall)\n'
              f'classification threshold: {results_df.class_threshold.iloc[0]}')
    plt.show()

    sns.catplot(data=results_df, x='estimator_name', y='fpr', kind='box')
    plt.xticks(rotation=90)
    plt.grid()
    plt.title(f'{n_bootstrap} bootstrap (no refit) samples - variation in fpr\n'
              f'classification threshold: {results_df.class_threshold.iloc[0]}')
    plt.show()


def roc_curve_bootstrap_no_refit_helper(trained_classifier, cap_x_df, y_df, roc_kwargs, roc_score_kwargs, n_bootstrap,
                                        data_set_name, model_selection_stage, classification_threshold, estimator_name):
    roc_auc_score_list = []
    class_thresh_tpr_list = []
    class_thresh_fpr_list = []
    df_row_dict_list = []
    for bootstrap_ in range(0, n_bootstrap):

        bs_cap_x_df, bs_y_df = get_bootstrap_sample(cap_x_df, y_df, bootstrap_)

        roc_df, fpr, tpr = get_roc_df(trained_classifier, bs_cap_x_df, bs_y_df, **roc_kwargs)

        class_thresh_tpr, class_thresh_fpr = get_tpr_and_fpr_for_threshold(roc_df, classification_threshold)
        class_thresh_tpr_list.append(class_thresh_tpr)
        class_thresh_fpr_list.append(class_thresh_fpr)

        roc_auc_score_list.append(
            roc_auc_score(y_true=bs_y_df, y_score=trained_classifier.predict_proba(bs_cap_x_df)[:, 1],
                          **roc_score_kwargs)
        )

        plt.plot(fpr, tpr, linewidth=2)

        if bootstrap_ == 0:
            plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
            plt.plot(class_thresh_fpr, class_thresh_tpr, "ko",
                     label=f'classification threshold = {classification_threshold}')
        else:
            plt.plot(class_thresh_fpr, class_thresh_tpr, "ko")

        df_row_dict_list.append(
            {
                'estimator_name': estimator_name,
                'tpr': class_thresh_tpr,
                'fpr': class_thresh_fpr,
                'class_threshold': classification_threshold
            }
        )

    plt.legend()
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate (recall)')
    plt.title(f'{n_bootstrap} bootstrapped roc curves\n'
              f'data_set_name: {data_set_name}; model_selection_stage: {model_selection_stage}')
    plt.grid()
    plt.show()

    # get the bootstrap conf intervals
    # bootstrap_statistics_helper(roc_auc_score_list, 'roc_auc_score_')
    # bootstrap_statistics_helper(class_thresh_tpr_list, 'tpr')
    # bootstrap_statistics_helper(class_thresh_fpr_list, 'fpr')

    # get basic statistics
    statistics_helper(roc_auc_score_list, 'roc_auc_score_')
    statistics_helper(class_thresh_tpr_list, 'tpr')
    statistics_helper(class_thresh_fpr_list, 'fpr')

    return pd.DataFrame(df_row_dict_list)


def plot_errors_as_a_function_of_classification_threshold(best_model_name, estimator_names, grid_search_cv_results_df,
                                                          cap_x_df, y_df, class_threshold_list, data_set_name=None,
                                                          model_selection_stage=None, labels=None, sample_weight=None,
                                                          normalize=None):
    if model_selection_stage is None:
        model_selection_stage = ''
    if data_set_name is None:
        data_set_name = ''

    for estimator_name in estimator_names:
        if estimator_name == best_model_name:

            print(estimator_name)
            best_estimator = \
                grid_search_cv_results_df.loc[
                    grid_search_cv_results_df.estimator == estimator_name, 'best_estimator'].iloc[0]

            df_row_dict_list = []
            for class_threshold in class_threshold_list:

                class_preds, _ = get_pred_class_binary(best_estimator, cap_x_df, class_threshold)
                conf_matrix = confusion_matrix(
                    y_df,
                    class_preds,
                    labels=labels,
                    sample_weight=sample_weight,
                    normalize=normalize
                )

                df_row_dict_list.append(
                    {
                        'tn': conf_matrix[0, 0],
                        'fp': conf_matrix[0, 1],
                        'fn': conf_matrix[1, 0],
                        'tp': conf_matrix[1, 1],
                        'class_threshold': class_threshold
                    }
                )

            results_df = pd.DataFrame(df_row_dict_list)

            sns.lineplot(data=results_df, x='class_threshold', y='fn', label='fn')
            sns.lineplot(data=results_df, x='class_threshold', y='fp', label='fp')
            plt.ylabel('fp and fn')
            plt.legend()
            plt.title(f'data_set_name: {data_set_name}; model_selection_stage: {model_selection_stage}')
            plt.grid()
            plt.show()


def get_bootstrap_sample(cap_x_df, y_df, random_state):

    temp_df = pd.concat([cap_x_df, y_df], axis=1)
    bs_df = resample(
        temp_df,
        replace=True,
        n_samples=None,  # None this is automatically set to the first dimension of the arrays.
        random_state=random_state,
        stratify=temp_df.iloc[:, -1]
    )
    bs_cap_x_df, bs_y_df = bs_df.iloc[:, :-1], bs_df.iloc[:, -1]

    return bs_cap_x_df, bs_y_df


def get_randomized_target_sample(cap_x_df, y_df, random_state):

    rt_df = pd.concat([cap_x_df, y_df], axis=1)
    target_name = rt_df.columns[-1]
    np.random.seed(random_state)
    rt_df[target_name] = np.random.permutation(rt_df[target_name])
    rt_cap_x_df, rt_y_df = rt_df.iloc[:, :-1], rt_df.iloc[:, -1]

    return rt_cap_x_df, rt_y_df


def get_mean_cal_error(trained_estimator, validation_cap_x_df, validation_y_df, bins=20):

    # https://stats.stackexchange.com/questions/213292/performance-measure-for-calibration-for-binary-classification-
    # problems

    # set up labels based on number of bins
    labels = [round(i, 2) for i in np.arange(1/(2*bins), 1.0, 1/bins)]

    # get the class 1 pred_proba values and bin them - each data instance is placed in a probability bin based on its
    # pred_proba value
    class_1_pred_proba = trained_estimator.predict_proba(validation_cap_x_df)[:, 1]
    binned_pred_proba_class_1 = pd.cut(class_1_pred_proba, bins=bins, labels=labels)

    # construct a data frame with these data elements - use notation from the link above
    try:
        y = validation_y_df.values.ravel()
    except AttributeError:
        y = validation_y_df

    class_1_pred_proba_df = pd.DataFrame(
        {
            'p_i_hat': class_1_pred_proba,
            'class_1_pred_proba_bin': binned_pred_proba_class_1,
            'y': y
        },
        index=validation_cap_x_df.index
    )

    # compute the observed probability of class 1 for each probability bin
    binned_class_1_pred_proba_df = \
        class_1_pred_proba_df.groupby('class_1_pred_proba_bin', observed=False)['y'].\
        agg(["sum", "count"]).reset_index().rename(columns={'class_1_pred_proba_bin': 'B_k'})
    binned_class_1_pred_proba_df['p_k'] = binned_class_1_pred_proba_df['sum'] / binned_class_1_pred_proba_df['count']
    binned_class_1_pred_proba_df = binned_class_1_pred_proba_df.drop(columns=['sum', 'count'])

    # develop a mapping dictionary to be used to map each data instance to an observed probability of class 1 using its
    # probability bin assignment
    cap_b_k_dict = dict(zip(binned_class_1_pred_proba_df['B_k'], binned_class_1_pred_proba_df['p_k']))

    # map each data instance to an observed probability of class 1 using its probability bin assignment
    class_1_pred_proba_df['p_k'] = class_1_pred_proba_df['class_1_pred_proba_bin'].map(cap_b_k_dict).astype(float)

    # calculate the mean calibration error
    class_1_pred_proba_df['diff_squared'] = np.square(class_1_pred_proba_df['p_k'] - class_1_pred_proba_df['p_i_hat'])
    mean_cal_error = (1/class_1_pred_proba_df.shape[0]) * np.sqrt(class_1_pred_proba_df['diff_squared'].sum())

    return mean_cal_error


if __name__ == '__main__':
    pass
