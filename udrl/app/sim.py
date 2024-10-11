import streamlit as st
import gymnasium as gym
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, Callable, List
from typing_extensions import Self
from types import MethodType
from pathlib import Path
import re
from streamlit_javascript import st_javascript
from skimage.transform import resize
from udrl.agent import UpsideDownAgent, AgentHyper
from udrl.policies import SklearnPolicy, NeuralPolicy
import altair as alt
import pandas as pd
import json

# from pylatexenc.latex2text import LatexNodes2Text


@dataclass
class BaseState:
    callbacks: List[Callable[[Self, ...], None]]

    def __post_init__(self):
        for k, v in self.callbacks.items():
            setattr(self, k, MethodType(v, self))


@dataclass
class State(BaseState):
    paused: bool = True
    next: bool = False
    reward: float = 0
    epoch: int = 0

    env_name: str = field(init=False)
    algo_name: str = field(init=False)
    init_desired_return: int = field(init=False)
    init_desired_horizon: int = field(init=False)
    desired_return: int = field(init=False)
    desired_horizon: int = field(init=False)
    agent: UpsideDownAgent = field(init=False)
    feature_importances: List[float] = field(init=False, default_factory=list)
    env: gym.Env = field(init=False)
    frame: np.array = field(init=False, default=None)
    prev_frame: np.array = field(init=False, default=None)
    obs: np.array = field(init=False)

    def __post_init__(self):
        super().__post_init__()


# Function to toggle pause state
def toggle_pause(state: State):
    state.paused = not state.paused


def next_epoch(state: State):
    state.next = True


def update_frame(state: State):
    state.prev_frame = state.frame
    state.frame = resize(state.env.render(), (400, 600, 3))
    while len(state.frame) < 1:
        state.frame = resize(state.env.render(), (400, 600, 3))


def reset_env(state: State):
    state.obs, _ = state.env.reset()
    state.desired_horizon = state.init_desired_horizon
    state.desired_return = state.init_desired_return
    state.reward = 0
    state.epoch = 0
    update_frame(state)


def sync_state(state: State):
    state.env_name = st.session_state.get("env_name", state.env_name)
    state.algo_name = st.session_state.get("algo_name", state.algo_name)
    state.init_desired_return = st.session_state.get(
        "desired_return", state.init_desired_return
    )
    state.init_desired_horizon = st.session_state.get(
        "desired_horizon", state.init_desired_horizon
    )


def make_env(state: State):
    state.env = gym.make(state.env_name, render_mode="rgb_array")
    reset_env(state)


def make_agent(state: State):
    hyper = AgentHyper(state.env_name, warm_up=0)
    policy_path = str(
        Path("resources") / state.env_name / state.algo_name / "policy"
    )
    if state.algo_name == "Neural":
        policy = NeuralPolicy.load(policy_path)
    else:
        policy = SklearnPolicy.load(policy_path)

    state.agent = UpsideDownAgent(hyper, policy)


def merge_callbacks(list_fun):
    def callback():
        for f in list_fun:
            f()

    return callback


CALLBACKS = [
    toggle_pause,
    update_frame,
    reset_env,
    make_env,
    sync_state,
    next_epoch,
    make_agent,
]


def init():
    state = State({c.__name__: c for c in CALLBACKS})
    return state


def get_envs():
    return [
        p.name
        for p in Path("resources").iterdir()
        if re.match(r".*-v.*", p.name)
    ]


def get_algo(env):

    return [
        p.name
        for p in (Path("resources") / env).iterdir()
        if p.name != "info.json"
    ]


def make_exp_parameters(state):
    with st.form("exp_param"):
        col1, col2, col3 = st.columns(3)
        with col1:
            state.env_name = st.radio(
                "Available Environments", sorted(get_envs()), key="env_name"
            )
        with col2:
            state.algo_name = st.radio(
                "Available Estimators",
                sorted(get_algo(state.env_name)),
                key="algo_name",
            )
        with col3:
            state.init_desired_return = st.number_input(
                "Desired return:", step=1, key="desired_return", value=200
            )
            state.init_desired_horizon = st.number_input(
                "Desired horizon:", step=1, key="desired_horizon", value=200
            )

        st.form_submit_button(
            "Update Experiment",
            on_click=merge_callbacks(
                [state.sync_state, state.make_env, state.make_agent]
            ),
        )
    return state


@st.cache_data
def load_env_info(env_name):
    info_path = Path("resources") / env_name / "info.json"
    with open(str(info_path), "r") as f:
        info = json.load(f)
    return info


def make_viz(state):
    with st.container(key="viz"):
        sim_col, featur_col = st.columns([3, 2], vertical_alignment="center")

        with sim_col:
            st.header("Environment Visualization", divider=True)
            image_placeholder = st.empty()
            image_placeholder.image(
                state.frame if len(state.frame) > 2 else state.prev_frame,
                caption=f"Environment Visualization {'(Paused)' if state.paused else ''}",
                use_column_width=True,
            )

        val = st_javascript(
            'window.parent.document.getElementsByClassName("st-key-viz")[0].querySelector("img").height'
        )

        with featur_col:
            st.header("Feature Importance", divider=True)
            if state.algo_name == "Neural":
                st.write(
                    "Unable to show Feature Importance for this estimator"
                )
            elif not state.feature_importances:
                st.write(
                    "Run the simulation to calculate the feature importance"
                )
            elif (
                len(state.feature_importances)
                != len(list(load_env_info(state.env_name)["state"].values()))
                + 2
            ):
                st.write(
                    "Run the simulation to calculate the feature importance"
                )

            else:
                data = pd.DataFrame(
                    {
                        "f": state.feature_importances,
                        "index": [
                            # LatexNodes2Text().latex_to_text(v[0])
                            v[1][:17]
                            for v in load_env_info(state.env_name)[
                                "state"
                            ].values()
                        ]
                        + ["d_r", "d_t"],
                    }
                )
                base = (
                    alt.Chart(data)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "f",
                            scale=alt.Scale(domain=[-10, 30]),
                        ),
                        y="index:O",
                        color=alt.Color("index:N", legend=None),
                    )
                    .properties(height=int(val))
                )

                st.altair_chart(
                    base + base.mark_text(align="left", dx=2),
                    use_container_width=True,
                )

    return state


def make_commands(state):
    # Add control buttons in a horizontal layout
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.button("Reset Environment", on_click=state.reset_env)
    with col2:
        st.button(
            "Resume" if state.paused else "Pause", on_click=state.toggle_pause
        )
    with col3:
        st.button("Next", on_click=state.next_epoch)
    with col4:
        st.button("Save")
    return state


def calculate_feat_imp(state: State, ext_state):
    if state.algo_name == "Neural":
        state.feature_importances = []
        return

    feature_importances = {idx: [] for idx in range(ext_state.shape[1])}

    for t in state.agent.policy.estimator.estimators_:
        branch = np.array(t.decision_path(ext_state).todense(), dtype=bool)
        imp = t.tree_.impurity[branch[0]]

        for f, i in zip(t.tree_.feature[branch[0]][:-1], imp[:-1] - imp[1:]):
            feature_importances.setdefault(f, []).append(i)

    state.feature_importances = [
        sum(feature_importances.get(k, [0.001]))
        for k in range(len(feature_importances.keys()))
    ]


@st.cache_data
def env_info_md(env_name):
    info = load_env_info(env_name)
    str_info = f"Environment name: {info['env_name']}\n"
    str_info += "State dimensions: \n"
    str_info += "&nbsp;&nbsp;&nbsp;&nbsp;" + "\n&nbsp;&nbsp;&nbsp;&nbsp;".join(
        [f"{v[0]} -> {v[1]}" for k, v in info["state"].items()]
    )
    return str_info


@st.cache_data
def load_algo_info(env_name, algo_name):
    info_path = Path("resources") / env_name / algo_name / "conf.json"
    with open(str(info_path), "r") as f:
        info = json.load(f)
    return info


@st.cache_data
def algo_info_md(env_name, algo_name):
    info = load_algo_info(env_name, algo_name)
    str_info = f"Algorithm: {info['estimator_name']}\n"
    str_info += f"Suggested Desired Return: {info['final_desired_return']}\n"
    str_info += f"Suggested Desired Horizon: {info['final_desired_horizon']}\n"
    full_conf = "&nbsp;&nbsp;&nbsp;&nbsp;" + "\n&nbsp;&nbsp;&nbsp;&nbsp;".join(
        [f"{k} -> {v}" for k, v in info.items()]
    )

    return str_info, full_conf


def make_metrics(state: State):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Epoch", state.epoch)
    with col2:
        st.metric("Cum Reward", f"{state.reward:.2f}")
    with col3:
        st.metric("Desired Return", f"{state.desired_return:.2f}")
    with col4:
        st.metric("Desired Horizon", state.desired_horizon)


def run(state: State):
    make_exp_parameters(state)
    if state.frame is None:
        state.sync_state()
        state.make_env()
        state.make_agent()
    make_metrics(state)
    make_viz(state)
    make_commands(state)

    if not state.paused or state.next:
        command = np.array(
            [
                state.desired_return * state.agent.conf.return_scale,
                state.desired_horizon * state.agent.conf.horizon_scale,
            ]
        )
        command = np.expand_dims(command, axis=0)
        obs = np.expand_dims(state.obs, axis=0)

        action = state.agent.policy(obs, command, True)

        ext_state = np.concatenate((obs, command), axis=1)

        state.obs, reward, ter, tru, info = state.env.step(action)

        calculate_feat_imp(state, ext_state)

        # Line 8 Algorithm 2
        state.desired_return -= reward
        # Line 9 Algorithm 2
        state.desired_horizon = max(state.desired_horizon - 1, 1)

        state.update_frame()
        state.next = False
        state.epoch += 1
        state.reward += reward
        if ter or tru:
            state.reset_env()

    # Create a container for environment info
    sidebar_container = st.sidebar.container()
    with sidebar_container:
        st.slider(
            "Refresh Rate", min_value=1, max_value=20, key="refresh_rate"
        )
        st.header("Environment Info")
        for line in env_info_md(state.env_name).split("\n"):
            st.markdown(line)
        st.header("Algorithm Info")
        algo_info, conf_info = algo_info_md(state.env_name, state.algo_name)
        for line in algo_info.split("\n"):
            st.markdown(line)
        with st.expander("Full Configuration"):
            for line in conf_info.split("\n"):
                st.markdown(line)

    # Add auto-refresh logic
    if not state.paused:
        sleep_time = 1 - (st.session_state.refresh_rate - 1) * (1 - 0.05) / 19
        time.sleep(sleep_time)  # Add a small delay to control refresh rate
        st.rerun()

    return state


if "sim" not in st.session_state:
    st.session_state.sim = init()

run(st.session_state.sim)
