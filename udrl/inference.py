import numpy as np
from udrl.policies import SklearnPolicy, NeuralPolicy
from udrl.agent import UpsideDownAgent, AgentHyper
from pathlib import Path
from collections import Counter
from pprint import pprint
import re
from dataclasses import dataclass, field
from typing import Dict, Any

# from tqdm import trange, tqdm


def get_common(base="", env="", conf="", seed="", path=None, algo_name=None):
    if path is None:
        path = base / env / conf / seed

    if not path.exists():
        print("Cannot find path")
        return None, None
    if algo_name is None:
        algo_name = "UNKNOWN"

    des_ret = np.load(str(path / "desired_returns.npy")).astype(int)
    des_hor = np.load(str(path / "desired_horizons.npy")).astype(int)
    # rew = np.load(str(path / "train_rewards.npy")).astype(int)[:, 0]

    te = []
    prev = -np.inf
    for i, x in enumerate(des_hor):
        if prev < x:
            te.append(i)
        prev = x

    init_des_ret = des_ret[te]
    init_des_hor = des_hor[te]

    mean_des_ret = []
    mean_des_hor = []
    tmp_r = []

    tmp_h = []
    for i, (ret, hor) in enumerate(zip(init_des_ret, init_des_hor)):
        tmp_r.append(ret)
        tmp_h.append(hor)
        if i % 15 == 0:
            mean_des_hor.append(np.mean(tmp_h))
            mean_des_ret.append(np.mean(tmp_r))
            tmp_r = []
            tmp_h = []

    common_hor = Counter(init_des_hor[-1500:]).most_common()[0][0]
    common_ret = Counter(init_des_ret[-1500:]).most_common()[0][0]
    print(f"{env}:{algo_name}.horizon-> {common_hor}")
    print(f"{env}:{algo_name}.return-> {common_ret}")
    return common_ret, common_hor


def test_desired(base, env, conf, algo_name, des_ret, des_hor):

    if des_hor is None or des_ret is None:
        print(f"Invalid desired for {env}:{algo_name}")
        return
    ret = []
    for path in (base / env / conf).iterdir():
        if "neural" in conf:
            policy = NeuralPolicy.load(str(path / "policy"))
        else:
            policy = SklearnPolicy.load(str(path / "policy"))

        hyper = AgentHyper(env, warm_up=0)

        agent = UpsideDownAgent(hyper, policy)

        final_r = [
            agent.collect_episode(
                des_ret,
                des_hor,
                test=True,
                store_episode=False,
            )[0]
            for _ in range(100)
        ]
        ret.append(
            {
                "env": env,
                "algo": algo_name,
                "seed": path.name,
                "des_ret": des_ret,
                "des_hor": des_hor,
                "final_r": np.median(final_r),
                "final_r_std": np.std(final_r),
                "final_r_max": np.max(final_r),
                "final_r_min": np.min(final_r),
                "final_raw": final_r,
            }
        )
        print(
            f"{env}:{algo_name}:{path.name}:r.{des_ret}:h.{des_hor}"
            f" -> {np.median(final_r):.2f} +- {np.std(final_r):.2f}"
            f",max {np.max(final_r):.2f},min {np.min(final_r):.2f}"
        )
    return ret


@dataclass
class RunStats:
    median: float
    mean: float
    std: float
    min_val: float
    max_val: float
    infos: Dict[str, Any] = field(repr=False)
    weights: Dict[str, float] = field(
        repr=False,
        default_factory=lambda: {
            "median": 0.5,
            "mean": 0.1,
            "std_penalty": 0.3,
            "range_penalty": 0.1,
        },
    )
    score: float = field(init=False, default=-np.inf)

    def __post_init__(self):
        self.score = self.calculate_score()

    def calculate_score(self):
        """
        Calculate a composite score for a run.

        The score is calculated as:
        score = (w1 * median + w2 * mean) * stability_factor

        where stability_factor penalizes high std and wide ranges
        """
        shift = 1000 if self.median < 0 else 0

        base_score = self.weights["median"] * (
            self.median + shift
        ) + self.weights["mean"] * (self.mean + shift)

        std_factor = 1 / (1 + self.weights["std_penalty"] * self.std)
        range_factor = 1 / (
            1 + self.weights["range_penalty"] * (self.max_val - self.min_val)
        )

        return base_score + std_factor + range_factor


def extract_statistics(run):
    return RunStats(
        median=run["final_r"],
        mean=np.mean(run["final_raw"]),
        std=run["final_r_std"],
        min_val=run["final_r_min"],
        max_val=run["final_r_max"],
        infos=run,
    )


base = Path("data")


envs = ["LunarLander-v2", "Acrobot-v1"]
algo_name_extract = r"(.*/)+(estimator_name(.+?)_train.*save_desired.*)$"
available_algo = []
special_confs = {
    "NN": "estimator_nameneural_batch_size256_warm_up260_save_desiredTrue",
    "RT": "train_per_iter1_save_desiredTrue",
}
wanted_algo = [
    "ensemble.ExtraTreesClassifier",
    "neighbors.KNeighborsClassifier",
    "NN",
    "RT",
    "svm.SVC",
    "ensemble.AdaBoostClassifier",
    "ensemble.GradientBoostingClassifier",
]

res = {}


for env in envs:
    res[env] = {}
    available_algo = [
        re.match(algo_name_extract, str(x))
        for x in (base / env).iterdir()
        if re.match(algo_name_extract, str(x))
    ]
    for algo_match in available_algo:
        conf = algo_match.group(2)
        algo_name = algo_match.group(3)
        res[env][(algo_name, conf)] = {}
        for seed_path in Path(algo_match.string).iterdir():
            seed = seed_path.stem
            ret, hor = get_common(path=seed_path, algo_name=algo_name)
            res[env][(algo_name, conf)][seed] = (ret, hor)
    for algo_name, conf in special_confs.items():
        res[env][(algo_name, conf)] = {}
        for seed_path in (base / env / conf).iterdir():
            seed = seed_path.stem
            ret, hor = get_common(path=seed_path, algo_name=algo_name)
            res[env][(algo_name, conf)][seed] = (ret, hor)


pprint(res)
data = []

for env, algos in res.items():
    for algo, seeds in algos.items():
        for _, vals in seeds.items():
            data.append(test_desired(base, env, algo[1], algo[0], *vals))


best_res = {env: {algo: None for algo in wanted_algo} for env in envs}

for runs in data:
    if runs[0]["algo"] not in wanted_algo:
        continue
    for run in runs:
        stats = extract_statistics(run)
        print(
            f"{run['env']}:{run['algo']}:{run['seed']}:r.{run['des_ret']}:h.{run['des_hor']}"
            f" -> SCORE {stats.score} \t {run['final_r']:.2f} +- {run['final_r_std']:.2f}"
            f",max {run['final_r_max']:.2f},min {run['final_r_min']:.2f}"
        )
        current_best = best_res[run["env"]][run["algo"]]
        if current_best is None:
            best_res[run["env"]][run["algo"]] = stats
            continue

        if current_best.score < stats.score:
            best_res[run["env"]][run["algo"]] = stats


pprint(best_res)

best_res["LunarLander-v2"]["svm.SVC"].infos
for env, vs in best_res.items():
    for algo, v in vs.items():
        run = v.infos
        print(
            f"{run['env']}:{run['algo']}: r. {run['des_ret']} : h. {run['des_hor']}"
        )
