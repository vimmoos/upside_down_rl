from pathlib import Path
import numpy as np
import json
import csv


def convert_json_to_pgfplots(json_file, output_file):
    """
    Convert JSON feature data to pgfplots-compatible format
    """
    # Read JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    # Extract feature values
    features = data["feature"]

    # Create pgfplots data
    # Each line will be "x y" format
    plot_data = []
    for i, value in features.items():
        plot_data.append(f"{i} {float(value)}")

    # Write to file
    with open(output_file, "w") as f:
        f.write("\n".join(plot_data))


def convert_all_viz(
    base=Path("data") / "viz_examples", algo="RandomForestClassifier"
):

    for env_p in base.iterdir():
        path = env_p / algo
        for fil in path.iterdir():
            if "info" not in fil.name:
                continue
            convert_json_to_pgfplots(
                str(fil), f"{str(fil.parent)}/{fil.stem}.dat"
            )


def smooth_curve(
    df,
    column_name="mean_reward",
    window_size=10,
    method="exponential",
    alpha=0.1,
):
    """
    Smooth a column in a pandas DataFrame using different methods.

    Parameters:
    - df: pandas DataFrame containing the data
    - column_name: name of the column to smooth
    - window_size: size of the rolling window for simple moving average
    - method: 'simple' for simple moving average, 'exponential' for exponential moving average
    - alpha: smoothing factor for exponential moving average (0 < alpha < 1)

    Returns:
    - DataFrame with both original and smoothed data
    """

    # Create a copy to avoid modifying the original DataFrame
    df_smoothed = df.copy()

    if method == "simple":
        # Simple Moving Average
        df_smoothed[f"{column_name}_smoothed"] = (
            df[column_name]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
        )

        # # Handle NaN values at the beginning and end
        df_smoothed[f"{column_name}_smoothed"].fillna(
            df[column_name], inplace=True
        )

    elif method == "exponential":
        # Exponential Moving Average
        df_smoothed[f"{column_name}_smoothed"] = (
            df[column_name].ewm(alpha=alpha, adjust=False).mean()
        )

    return df_smoothed


# import pandas as pd

# path = "data/csvs/Acrobot-v1.csv"
# path = "data/csvs/CartPole-v0.csv"
# path = "data/csvs/LunarLander-v2.csv"

# dat = pd.read_csv(path)


# for name in ["NN", "ET", "RF", "KNN", "SVM", "AdaBoost", "XGBoost"]:
#     dat = smooth_curve(dat, name + "_mean", window_size=20, method="simple")

# dat.to_csv(path)


naming = {
    "neural": "NN",
    "ensemble.ExtraTreesClassifier": "ET",
    "ensemble.RandomForestClassifier": "RF",
    "neighbors.KNeighborsClassifier": "KNN",
    "svm.SVC": "SVM",
    "ensemble.AdaBoostClassifier": "AdaBoost",
    "ensemble.GradientBoostingClassifier": "XGBoost",
}

if __name__ == "__main__":
    path = Path("data")
    csvs_path = path / "csvs"
    csvs_path.mkdir(parents=True, exist_ok=True)
    for env in path.iterdir():
        all_paths = list(set([p.parent for p in env.rglob("*.npy")]))
        if not all_paths:
            continue
        toy_rewards = np.load(all_paths[0] / "train_rewards.npy")
        data = {"episode": list(range(len(toy_rewards)))}
        estimators = {
            "neural": ([], [], [], []),
            "ensemble.ExtraTreesClassifier": ([], [], [], []),
            "ensemble.RandomForestClassifier": ([], [], [], []),
            "neighbors.KNeighborsClassifier": ([], [], [], []),
            "svm.SVC": ([], [], [], []),
            "ensemble.AdaBoostClassifier": ([], [], [], []),
            "ensemble.GradientBoostingClassifier": ([], [], [], []),
        }
        for exp in all_paths:
            print(exp)
            rewards = np.load(exp / "train_rewards.npy")

            with open((exp / "conf.json"), "r") as f:
                conf = json.load(f)
            if conf["estimator_name"] not in list(estimators.keys()):
                continue

            estimators[conf["estimator_name"]][0].append(list(rewards[:, 0]))
            estimators[conf["estimator_name"]][1].append(list(rewards[:, 1]))
            estimators[conf["estimator_name"]][2].append(conf["test_mean"])
            estimators[conf["estimator_name"]][3].append(conf["test_std"])

        for k, v in estimators.items():
            data[naming[k] + "_mean"] = [
                "{:.2f}".format(np.mean(x)) for x in zip(*v[0])
            ]
            data[naming[k] + "_std"] = [
                "{:.2f}".format(np.std(x)) for x in zip(*v[0])
            ]
            print(f"{k}:{env.name}-> {np.median(v[2])} +- {np.median(v[3])}")

        with open(csvs_path / f"{env.name}.csv", "w") as f:
            w = csv.writer(f)
            w.writerow(data.keys())
            w.writerows(zip(*data.values()))
