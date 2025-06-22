import optuna
import pandas as pd
import os

DB_PATH = "sqlite:///gan_optuna_study_IS.db"

OUTPUT_DIR = "results/gan/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

study_summaries = optuna.get_all_study_summaries(storage=DB_PATH)

for summary in study_summaries:
    study_name = summary.study_name
    print(f"Exporting study: {study_name}")

    study = optuna.load_study(study_name=study_name, storage=DB_PATH)

    trials_data = []
    for trial in study.trials:
        flat_params = {f"param_{k}": v for k, v in trial.params.items()}

        if trial.values is not None:
            if len(trial.values) == 1:
                value_data = {"value": trial.values[0]}
            else:
                value_data = {f"value_{i}": val for i, val in enumerate(trial.values)}
        else:
            value_data = {"value": None}

        row = {
            "number": trial.number,
            "state": trial.state.name,
            **value_data,
            **flat_params,
        }
        trials_data.append(row)

    df = pd.DataFrame(trials_data)
    csv_path = os.path.join(OUTPUT_DIR, f"{study_name}.csv")
    df.to_csv(csv_path, index=False)

    print("Best trials:")
    for trial in study.best_trials:
        print(f"Trial #{trial.number}")
        print(f"  Time: {trial.datetime_start} - {trial.datetime_complete}")
        print(f"  Values (IS ↑, FID ↓): {trial.values}")
        print(f"  Params: {trial.params}")

    print("\n")
