import os

from numerapi import NumerAPI

KEY = os.environ['NUMERAI_PUBLIC_KEY']
SECRET = os.environ['NUMERAI_SECRET_KEY']

desired_model = 'jbenaducci_heavy'
api = NumerAPI(public_id=KEY, secret_key=SECRET)

models = api.get_models()

model_id = None
for k, v in models:
    if k == desired_model:
        model_id = v

        break

if model_id is None:
    print('Could not find model in Numerai')
else:
    tournament_round = api.get_current_round()

    api.upload_predictions(
        file_path=f"prediction_files/tournament_predictions_{tournament_round}.csv",
        model_id=model_id,
        version=2
    )
