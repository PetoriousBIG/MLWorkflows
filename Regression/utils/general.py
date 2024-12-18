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


def split_data(df, target_variable, test_ratio, split_random_state):

    data = df.drop(target_variable, inplace=False, axis=1)
    target = df[[target_variable]]

    training_data, testing_data, training_target, testing_target = \
        train_test_split(data, target, test_size=test_ratio, random_state=split_random_state, shuffle=True, stratify=None)

    return training_data, testing_data, training_target, testing_target