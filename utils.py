import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from halo import Halo
from scipy.stats import skew

ERA_COL = "era"

TARGET_COLUMNS = {
    'target': 'target',
    'nomi20': 'target_nomi_20',
    'nomi60': 'target_nomi_60',
    'jerome20': 'target_jerome_20',
    'jerome60': 'target_jerome_60',
    'janet20': 'target_janet_20',
    'janet60': 'target_janet_60',
    'ben20': 'target_ben_20',
    'ben60': 'target_ben_60',
    'alan20': 'target_alan_20',
    'alan60': 'target_alan_60',
    'paul20': 'target_paul_20',
    'paul60': 'target_paul_60',
    'george20': 'target_george_20',
    'george60': 'target_george_60',
    'william20': 'target_william_20',
    'william60': 'target_william_60',
    'arthur20': 'target_arthur_20',
    'arthur60': 'target_arthur_60',
    'thomas20': 'target_thomas_20',
    'thomas60': 'target_thomas_60',
}

TARGET_OPTIMUM_PARAMS = {
    TARGET_COLUMNS['target']: {
        "boosting_type": "goss",
        "n_estimators": 1854,
        "learning_rate": 0.020430332790521113,
        "num_leaves": 7,
        "max_depth": 5,
        "feature_fraction": 0.09706028770241243,
        "bagging_fraction": 0.014553134875531988,
        "min_data_in_leaf": 142
    },
    TARGET_COLUMNS['nomi20']: {
        "boosting_type": "goss",
        "n_estimators": 1978,
        "learning_rate": 0.04378455468413633,
        "num_leaves": 5,
        "max_depth": 5,
        "feature_fraction": 0.11101377120763281,
        "bagging_fraction": 0.15678624159723772,
        "min_data_in_leaf": 1201
    },
    TARGET_COLUMNS['nomi60']: {
        "boosting_type": "goss",
        "n_estimators": 1978,
        "learning_rate": 0.04378455468413633,
        "num_leaves": 5,
        "max_depth": 5,
        "feature_fraction": 0.11101377120763281,
        "bagging_fraction": 0.15678624159723772,
        "min_data_in_leaf": 1201
    },
    TARGET_COLUMNS['jerome20']: {
        "boosting_type": "goss",
        "n_estimators": 1978,
        "learning_rate": 0.04378455468413633,
        "num_leaves": 5,
        "max_depth": 5,
        "feature_fraction": 0.11101377120763281,
        "bagging_fraction": 0.15678624159723772,
        "min_data_in_leaf": 1201
    },
    TARGET_COLUMNS['jerome60']: {
        "boosting_type": "goss",
        "n_estimators": 1978,
        "learning_rate": 0.04378455468413633,
        "num_leaves": 5,
        "max_depth": 5,
        "feature_fraction": 0.11101377120763281,
        "bagging_fraction": 0.15678624159723772,
        "min_data_in_leaf": 1201
    },
    TARGET_COLUMNS['janet20']: {
        "boosting_type": "goss",
        "n_estimators": 1978,
        "learning_rate": 0.04378455468413633,
        "num_leaves": 5,
        "max_depth": 5,
        "feature_fraction": 0.11101377120763281,
        "bagging_fraction": 0.15678624159723772,
        "min_data_in_leaf": 1201
    },
    TARGET_COLUMNS['janet60']: {
        "boosting_type": "goss",
        "n_estimators": 1842,
        "learning_rate": 0.026520287506337205,
        "num_leaves": 6,
        "max_depth": 5,
        "feature_fraction": 0.10866666339142045,
        "bagging_fraction": 0.0178426564762887,
        "min_data_in_leaf": 270
    },
    TARGET_COLUMNS['ben20']: {
        "boosting_type": "goss",
        "n_estimators": 1978,
        "learning_rate": 0.04378455468413633,
        "num_leaves": 5,
        "max_depth": 5,
        "feature_fraction": 0.11101377120763281,
        "bagging_fraction": 0.15678624159723772,
        "min_data_in_leaf": 1201
    },
    TARGET_COLUMNS['ben60']: {
        "boosting_type": "goss",
        "n_estimators": 1978,
        "learning_rate": 0.04378455468413633,
        "num_leaves": 5,
        "max_depth": 5,
        "feature_fraction": 0.11101377120763281,
        "bagging_fraction": 0.15678624159723772,
        "min_data_in_leaf": 1201
    },
    TARGET_COLUMNS['alan20']: {
        "boosting_type": "goss",
        "n_estimators": 1978,
        "learning_rate": 0.04378455468413633,
        "num_leaves": 5,
        "max_depth": 5,
        "feature_fraction": 0.11101377120763281,
        "bagging_fraction": 0.15678624159723772,
        "min_data_in_leaf": 1201
    },
    TARGET_COLUMNS['alan60']: {
        "boosting_type": "goss",
        "n_estimators": 1978,
        "learning_rate": 0.04378455468413633,
        "num_leaves": 5,
        "max_depth": 5,
        "feature_fraction": 0.11101377120763281,
        "bagging_fraction": 0.15678624159723772,
        "min_data_in_leaf": 1201
    },
    TARGET_COLUMNS['paul20']: {
        "boosting_type": "goss",
        "n_estimators": 1978,
        "learning_rate": 0.04378455468413633,
        "num_leaves": 5,
        "max_depth": 5,
        "feature_fraction": 0.11101377120763281,
        "bagging_fraction": 0.15678624159723772,
        "min_data_in_leaf": 1201
    },
    TARGET_COLUMNS['paul60']: {
        "boosting_type": "goss",
        "n_estimators": 1978,
        "learning_rate": 0.04378455468413633,
        "num_leaves": 5,
        "max_depth": 5,
        "feature_fraction": 0.11101377120763281,
        "bagging_fraction": 0.15678624159723772,
        "min_data_in_leaf": 1201
    },
    TARGET_COLUMNS['george20']: {
        "boosting_type": "goss",
        "n_estimators": 1978,
        "learning_rate": 0.04378455468413633,
        "num_leaves": 5,
        "max_depth": 5,
        "feature_fraction": 0.11101377120763281,
        "bagging_fraction": 0.15678624159723772,
        "min_data_in_leaf": 1201
    },
    TARGET_COLUMNS['george60']: {
        "boosting_type": "goss",
        "n_estimators": 1978,
        "learning_rate": 0.04378455468413633,
        "num_leaves": 5,
        "max_depth": 5,
        "feature_fraction": 0.11101377120763281,
        "bagging_fraction": 0.15678624159723772,
        "min_data_in_leaf": 1201
    },
    TARGET_COLUMNS['william20']: {
        "boosting_type": "goss",
        "n_estimators": 1978,
        "learning_rate": 0.04378455468413633,
        "num_leaves": 5,
        "max_depth": 5,
        "feature_fraction": 0.11101377120763281,
        "bagging_fraction": 0.15678624159723772,
        "min_data_in_leaf": 1201
    },
    TARGET_COLUMNS['william60']: {
        "boosting_type": "goss",
        "n_estimators": 1978,
        "learning_rate": 0.04378455468413633,
        "num_leaves": 5,
        "max_depth": 5,
        "feature_fraction": 0.11101377120763281,
        "bagging_fraction": 0.15678624159723772,
        "min_data_in_leaf": 1201
    },
    TARGET_COLUMNS['arthur20']: {
        "boosting_type": "goss",
        "objective": "regression",
        "n_estimators": 1975,
        "learning_rate": 0.08136245681492366,
        "num_leaves": 5,
        "max_depth": 10,
        "feature_fraction": 0.10285424138031136,
        "bagging_fraction": 0.02998509136892534,
        "min_data_in_leaf": 265,
        "lambda_l1": 7.606687251778586e-05,
        "lambda_l2": 0.05439351914172123
    },
    TARGET_COLUMNS['arthur60']: {
        "boosting_type": "goss",
        "objective": "regression",
        "n_estimators": 1986,
        "learning_rate": 0.015860118962803368,
        "num_leaves": 18,
        "max_depth": 46,
        "feature_fraction": 0.14782041331534704,
        "bagging_fraction": 0.011526112001252054,
        "min_data_in_leaf": 5165,
        "lambda_l1": 0.0353245861233883
    },
    TARGET_COLUMNS['thomas20']: {
        "boosting_type": "goss",
        "n_estimators": 1956,
        "learning_rate": 0.017314775722883955,
        "num_leaves": 34,
        "max_depth": 68,
        "feature_fraction": 0.19993147115415724,
        "bagging_fraction": 0.06766978813067094,
        "min_data_in_leaf": 672
    },
    TARGET_COLUMNS['thomas60']: {
        "boosting_type": "goss",
        "n_estimators": 1958,
        "learning_rate": 0.0905131449427027,
        "num_leaves": 7,
        "max_depth": 5,
        "feature_fraction": 0.09854767856610315,
        "bagging_fraction": 0.06908974187675336,
        "min_data_in_leaf": 154
    },
}

DATA_TYPE_COL = "data_type"

spinner = Halo(text='', spinner='dots')

MODEL_FOLDER = "models"
MODEL_CONFIGS_FOLDER = "model_configs"
PREDICTION_FILES_FOLDER = "prediction_files"


def save_model(model, name):
    try:
        Path(MODEL_FOLDER).mkdir(exist_ok=True, parents=True)
    except Exception as ex:
        pass
    pd.to_pickle(model, f"{MODEL_FOLDER}/{name}.pkl")


def load_model(name):
    path = Path(f"{MODEL_FOLDER}/{name}.pkl")
    if path.is_file():
        model = pd.read_pickle(f"{MODEL_FOLDER}/{name}.pkl")
    else:
        model = False
    return model


def save_model_config(model_config, model_name):
    try:
        Path(MODEL_CONFIGS_FOLDER).mkdir(exist_ok=True, parents=True)
    except Exception as ex:
        pass
    with open(f"{MODEL_CONFIGS_FOLDER}/{model_name}.json", 'w') as fp:
        json.dump(model_config, fp)


def load_model_config(model_name):
    path_str = f"{MODEL_CONFIGS_FOLDER}/{model_name}.json"
    path = Path(path_str)
    if path.is_file():
        with open(path_str, 'r') as fp:
            model_config = json.load(fp)
    else:
        model_config = False
    return model_config


def get_biggest_change_features(corrs, n):
    all_eras = corrs.index.sort_values()
    h1_eras = all_eras[:len(all_eras) // 2]
    h2_eras = all_eras[len(all_eras) // 2:]

    h1_corr_means = corrs.loc[h1_eras, :].mean()
    h2_corr_means = corrs.loc[h2_eras, :].mean()

    corr_diffs = h2_corr_means - h1_corr_means
    worst_n = corr_diffs.abs().sort_values(ascending=False).head(n).index.tolist()
    return worst_n


def get_time_series_cross_val_splits(data, cv=3, embargo=12):
    all_train_eras = data[ERA_COL].unique()
    len_split = len(all_train_eras) // cv
    test_splits = [all_train_eras[i * len_split:(i + 1) * len_split] for i in range(cv)]
    # fix the last test split to have all the last eras, in case the number of eras wasn't divisible by cv
    test_splits[-1] = np.append(test_splits[-1], all_train_eras[-1])

    train_splits = []
    for test_split in test_splits:
        test_split_max = int(np.max(test_split))
        test_split_min = int(np.min(test_split))
        # get all of the eras that aren't in the test split
        train_split_not_embargoed = [e for e in all_train_eras if not (test_split_min <= int(e) <= test_split_max)]
        # embargo the train split so we have no leakage.
        # one era is length 5, so we need to embargo by target_length/5 eras.
        # To be consistent for all targets, let's embargo everything by 60/5 == 12 eras.
        train_split = [e for e in train_split_not_embargoed if
                       abs(int(e) - test_split_max) > embargo and abs(int(e) - test_split_min) > embargo]
        train_splits.append(train_split)

    # convenient way to iterate over train and test splits
    train_test_zip = zip(train_splits, test_splits)
    return train_test_zip


def neutralize(df,
               columns,
               neutralizers=None,
               proportion=1.0,
               normalize=True,
               era_col="era"):
    if neutralizers is None:
        neutralizers = []
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        df_era = df[df[era_col] == u]
        scores = df_era[columns].values
        if normalize:
            scores2 = []
            for x in scores.T:
                x = (scipy.stats.rankdata(x, method='ordinal') - .5) / len(x)
                x = scipy.stats.norm.ppf(x)
                scores2.append(x)
            scores = np.array(scores2).T
        exposures = df_era[neutralizers].values

        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32)).dot(scores.astype(np.float32)))

        scores /= scores.std(ddof=0)

        computed.append(scores)

    return pd.DataFrame(np.concatenate(computed),
                        columns=columns,
                        index=df.index)


def unif(df):
    x = (df.rank(method="first") - 0.5) / len(df)
    return pd.Series(x, index=df.index)


def validation_metrics(validation_data, pred_cols, target_col):
    validation_stats = pd.DataFrame()
    for pred_col in pred_cols:
        validation_correlations = validation_data.groupby(ERA_COL).apply(
            lambda d: unif(d[pred_col]).corr(d[target_col]))

        mean = validation_correlations.mean()
        std = validation_correlations.std(ddof=0)
        sharpe = mean / std

        validation_stats.loc["mean", pred_col] = mean
        validation_stats.loc["std", pred_col] = std
        validation_stats.loc["sharpe", pred_col] = sharpe

    # .transpose so that stats are columns and the model_name is the row
    return validation_stats.transpose()


def download_data(napi, filename, dest_path):
    spinner.start(f'Downloading {dest_path}')
    napi.download_dataset(filename, dest_path)
    spinner.succeed()
