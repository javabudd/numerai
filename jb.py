import gc
import json

import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from numerapi import NumerAPI

from utils import (
    neutralize,
    get_biggest_change_features,
    validation_metrics,
    ERA_COL,
    DATA_TYPE_COL,
    TARGET_COLUMNS
)

TARGET_COL = TARGET_COLUMNS['arthur20']

napi = NumerAPI()

current_round = napi.get_current_round(tournament=8)  # tournament 8 is the primary Numerai Tournament

# Tournament data changes every week so we specify the round in their name. Training
# and validation data only change periodically, so no need to download them every time.
print('Downloading dataset files...')
napi.download_dataset("numerai_training_data.parquet", "training_data.parquet")
napi.download_dataset("numerai_validation_data.parquet", f"validation_data.parquet")
napi.download_dataset("features.json", "features.json")

print('Reading minimal training data')
# read the feature metadata amd get the "small" feature set
with open("features.json", "r") as f:
    feature_metadata = json.load(f)
features = feature_metadata["feature_sets"]["small"]
# read in just those features along with era and target columns
read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
training_data = pd.read_parquet('training_data.parquet', columns=read_columns)

# pare down the number of eras to every 4th era
# every_4th_era = training_data[ERA_COL].unique()[::4]
# training_data = training_data[training_data[ERA_COL].isin(every_4th_era)]

# getting the per era correlation of each feature vs the target
all_feature_corrs = training_data.groupby(ERA_COL).apply(
    lambda era: era[features].corrwith(era[TARGET_COL])
)

# find the riskiest features by comparing their correlation vs
# the target in each half of training data; we'll use these later
riskiest_features = get_biggest_change_features(all_feature_corrs, 50)

# "garbage collection" (gc) gets rid of unused data and frees up memory
gc.collect()

model_name = f"model_target"


def objective(optuna_trial: optuna.trial):
    max_depth = optuna_trial.suggest_int("max_depth", 2, 50, log=True)

    params = {
        "objective": optuna_trial.suggest_categorical('objective', ['regression', 'binary']),
        "boosting_type": "goss",
        "n_estimators": optuna_trial.suggest_int("n_estimators", 1800, 2000, log=True),
        "learning_rate": optuna_trial.suggest_float("learning_rate", .01, .1, log=True),
        "max_depth": max_depth,
        "num_leaves": optuna_trial.suggest_int("num_leaves", 2, 2 ^ max_depth, log=True),
        "feature_fraction": optuna_trial.suggest_float("feature_fraction ", .001, 1, log=True),
        "bagging_fraction": optuna_trial.suggest_float("bagging_fraction  ", .001, 1, log=True),
        "min_data_in_leaf": optuna_trial.suggest_int("min_data_in_leaf", 100, 9999, log=True),
        "lambda_l1": optuna_trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
    }

    model = LGBMRegressor(**params)

    # train on all of train and save the model so we don't have to train next time
    model.fit(
        training_data.filter(like='feature_', axis='columns'),
        training_data[TARGET_COL]
    )

    gc.collect()

    validation_data = pd.read_parquet('validation_data.parquet', columns=read_columns)

    # double check the feature that the model expects vs what is available to prevent our
    # pipeline from failing if Numerai adds more data and we don't have time to retrain!
    model_expected_features = model.booster_.feature_name()
    if set(model_expected_features) != set(features):
        print(f"New features are available! Might want to retrain model {model_name}.")
    validation_data.loc[:, f"preds_{model_name}"] = model.predict(
        validation_data.loc[:, model_expected_features])

    gc.collect()

    # neutralize our predictions to the riskiest features
    validation_data[f"preds_{model_name}_neutral_riskiest_50"] = neutralize(
        df=validation_data,
        columns=[f"preds_{model_name}"],
        neutralizers=riskiest_features,
        proportion=1.0,
        normalize=True,
        era_col=ERA_COL
    )

    model_to_submit = f"preds_{model_name}_neutral_riskiest_50"

    # rename best model to "prediction" and rank from 0 to 1 to meet upload requirements
    validation_data["prediction"] = validation_data[model_to_submit].rank(pct=True)

    validation_stats = validation_metrics(
        validation_data,
        [model_to_submit],
        TARGET_COL
    )

    print(validation_stats["sharpe"].iloc[-1])

    return validation_stats["sharpe"].iloc[-1]


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
