from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import time
import pandas as pd


def drop_obs_with_nans(a_df):
    """

    :param a_df:
    :return:
    """

    if a_df.isna().sum().sum() > 0:
        print(f'\nfound observations with nans - pre obs. drop a_df.shape: {a_df.shape}')
        a_df = a_df.dropna(axis=0, how='any')
        print(f'post obs. drop a_df.shape: {a_df.shape}')

    return a_df


def prep_data_for_vif_calc(a_df, a_num_attr_list):
    """

    :param a_df:
    :param a_num_attr_list:
    :return:
    """

    # drop observations with nans
    a_df = drop_obs_with_nans(a_df[a_num_attr_list])

    # prepare the data - make sure you perform the analysis on the design matrix
    design_matrix = None
    bias_attr = None
    for attr in a_df[a_num_attr_list]:
        if a_df[attr].nunique() == 1 and a_df[attr].iloc[0] == 1:  # found the bias attribute
            design_matrix = a_df[a_num_attr_list]
            bias_attr = attr
            print('found the bias term - no need to add one')
            break

    if design_matrix is None:
        design_matrix = sm.add_constant(a_df[a_num_attr_list])
        bias_attr = 'const'
        print('\nAdded a bias term to the data frame to construct the design matrix for assessment of vifs.', sep='')

    # if numerical attributes in the data frame are not scaled then scale them - don't scale the bias term
    if not (a_df[a_num_attr_list].mean() <= 1e-14).all():
        print('scale the attributes - but not the bias term')
        design_matrix[a_num_attr_list] = StandardScaler().fit_transform(design_matrix[a_num_attr_list])

    return design_matrix, bias_attr


def print_vifs(a_df, a_num_attr_list, vif_inspection_threshold=2, ols_large_vifs=True):
    """

    :param a_df:
    :param a_num_attr_list:
    :param vif_inspection_threshold:
    :param ols_large_vifs:
    :return:
    """

    # VIF determines the strength of the correlation between the independent variables. It is predicted by taking a
    # variable and regressing it against every other variable.
    # VIF score of an independent variable represents how well the variable is explained by other independent variables.
    # https://www.analyticsvidhya.com/blog/2020/03/what-is-multicollinearity/
    # https://www.statsmodels.org/v0.13.0/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html

    print('\n', 40 * '*', sep='')
    print('investigate multi co-linearity - calculate variance inflation factors:')

    design_matrix, bias_attr = prep_data_for_vif_calc(a_df, a_num_attr_list)

    # calculate the vifs
    vif_df = pd.DataFrame()
    vif_df['attribute'] = design_matrix.columns.tolist()
    vif_df['vif'] = [variance_inflation_factor(design_matrix.values, i) for i in range(design_matrix.shape[1])]
    vif_df['vif'] = vif_df['vif'].round(2)
    vif_df = vif_df.sort_values('vif')

    print('\n', vif_df, sep='')
    time.sleep(2)

    if (vif_df.vif.values > vif_inspection_threshold).any() and ols_large_vifs:
        check_out_large_vifs(vif_df, design_matrix, vif_inspection_threshold=vif_inspection_threshold)

    return vif_df


def check_out_large_vifs(vif_df, design_matrix, vif_inspection_threshold=2):

    vif_gt_threshold_list = vif_df.loc[vif_df.vif > vif_inspection_threshold, 'attribute'].tolist()

    if len(vif_gt_threshold_list) > 0:
        print(f'\nthe attributes {vif_gt_threshold_list} have vif values greater than {vif_inspection_threshold} - '
              f'let\'s look at the details of regressing them on the rest of the design matrix')

        for inf_vif_attr in sorted(vif_gt_threshold_list, reverse=True):
            predictors = design_matrix.columns.tolist()
            predictors.remove(inf_vif_attr)
            est = sm.OLS(design_matrix[inf_vif_attr], design_matrix[predictors]).fit()
            print(f'\n\n\n\n{inf_vif_attr}')
            print(est.summary())
    else:
        print(f'\nno attributes have vifs greater than {vif_inspection_threshold} - no further vif analysis required')


if __name__ == '__main__':
    pass
