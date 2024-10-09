import streamlit as st
import gymnasium as gym
import numpy as np
from PIL import Image
import time

# Initialize session state variables if they don't exist
if "env" not in st.session_state:
    st.session_state.env = gym.make("LunarLander-v2", render_mode="rgb_array")
    st.session_state.env.reset()
    st.session_state.frame = st.session_state.env.render()
if "paused" not in st.session_state:
    st.session_state.paused = False


# Function to reset the environment
def reset_environment():
    st.session_state.env.reset()


# Function to toggle pause state
def toggle_pause():
    st.session_state.paused = not st.session_state.paused


# Create the Streamlit app
st.title("Gymnasium Environment Viewer")

# Add control buttons in a horizontal layout
col1, col2 = st.columns(2)
with col1:
    st.button("Reset Environment", on_click=reset_environment)
with col2:
    if st.session_state.paused:
        st.button("Resume", on_click=toggle_pause)
    else:
        st.button("Pause", on_click=toggle_pause)

# Create a placeholder for the image
image_placeholder = st.empty()

# Create a container for environment info
sidebar_container = st.sidebar.container()

# Main simulation loop using rerun
if not st.session_state.paused:
    # Take a random action
    action = st.session_state.env.action_space.sample()
    observation, reward, terminated, truncated, info = (
        st.session_state.env.step(action)
    )

    # Render the environment
    st.session_state.frame = st.session_state.env.render()

    # Reset if the episode is done
    if terminated or truncated:
        st.session_state.env.reset()
# Display the frame
if st.session_state.paused:
    image_placeholder.image(
        st.session_state.frame,
        caption="Environment Visualization (Paused)",
        use_column_width=True,
    )
else:
    image_placeholder.image(
        st.session_state.frame,
        caption="Environment Visualization",
        use_column_width=True,
    )

# Display some information about the environment
with sidebar_container:
    st.header("Environment Info")
    st.write(f"Action Space: {st.session_state.env.action_space}")
    st.write(f"Observation Space: {st.session_state.env.observation_space}")

# Add auto-refresh logic
if not st.session_state.paused:
    time.sleep(0.1)  # Add a small delay to control refresh rate
    st.rerun()

# fig, ax = plt.subplots()
# ax.imshow(env.render())
# st.pyplot(fig)
# st.image(env.render())


# import gymnasium as gym
# import streamlit as st
# import numpy as np
# from udrl.policies import SklearnPolicy
# from udrl.agent import UpsideDownAgent, AgentHyper
# from pathlib import Path

# # import json


# def normalize_value(value, is_bounded, low=None, high=None):
#     return (value - low) / (high - low)


# def visualize_environment(
#     state,
#     env,
#     # paused,
#     feature_importances,
#     epoch,
#     max_epoch=200,
# ):

#     st.image(env.render())
#     st.image(e)
#     # Render the Gym environment
#     # env_render = env.render()

#     # # Display the rendered image using Streamlit
#     # st.image(env_render, caption=f"Epoch {epoch}", use_column_width=True)

#     # Display feature importances using Streamlit metrics
#     # cols = st.columns(len(feature_importances))
#     # for i, col in enumerate(cols):
#     #     col.metric(
#     #         label=f"Importance {i}", value=f"{feature_importances[i]:.2f}"
#     #     )

#     # Create buttons using Streamlit
#     # reset_button = st.button("Reset")
#     # pause_play_button = st.button("Pause" if not paused else "Play")
#     # next_button = st.button("Next")
#     # save_button = st.button("Save")

#     # return reset_button, pause_play_button, next_button, save_button


# def run_visualization(
#     env_name,
#     agent,
#     init_desired_return,
#     init_desired_horizon,
#     max_epoch,
#     base_path,
# ):
#     # base_path = (
#     #     Path(base_path) / env_name / agent.policy.estimator.__str__()[:-2]
#     # )
#     # base_path.mkdir(parents=True, exist_ok=True)
#     desired_return = init_desired_return
#     desired_horizon = init_desired_horizon

#     # Initialize the Gym environment
#     env = gym.make(env_name, render_mode="rgb_array")
#     state, _ = env.reset()

#     epoch = 0
#     # save_index = 0

#     # paused = False
#     # step = False

#     # # Use Streamlit session state to manage paused state
#     # if "paused" not in st.session_state:
#     #     st.session_state.paused = False

#     while True:
#         # Render and display the environment
#         env_render = env.render()
#         # if not st.session_state.pausedor step:
#         command = np.array(
#             [
#                 desired_return * agent.conf.return_scale,
#                 desired_horizon * agent.conf.horizon_scale,
#             ]
#         )
#         command = np.expand_dims(command, axis=0)
#         state = np.expand_dims(state, axis=0)

#         action = agent.policy(state, command, True)

#         ext_state = np.concatenate((state, command), axis=1)

#         state, reward, done, truncated, info = env.step(action)

#         feature_importances = {idx: [] for idx in range(ext_state.shape[1])}

#         for t in agent.policy.estimator.estimators_:
#             branch = np.array(t.decision_path(ext_state).todense(), dtype=bool)
#             imp = t.tree_.impurity[branch[0]]

#             for f, i in zip(
#                 t.tree_.feature[branch[0]][:-1], imp[:-1] - imp[1:]
#             ):
#                 feature_importances.setdefault(f, []).append(i)

#         # Line 8 Algorithm 2
#         desired_return -= reward
#         # Line 9 Algorithm 2
#         desired_horizon = max(desired_horizon - 1, 1)

#         summed_importances = [
#             sum(feature_importances.get(k, [0.001]))
#             for k in range(len(feature_importances.keys()))
#         ]

#         epoch += 1
#         visualize_environment(
#             state,
#             env,
#             # st.session_state.paused,  # Use session state
#             summed_importances,
#             epoch,
#             max_epoch,
#         )
#         # reset_button, pause_play_button, next_button, save_button = (

#         # )

#         if done or truncated:
#             state, _ = env.reset()
#             desired_horizon = init_desired_horizon
#             desired_return = init_desired_return
#             epoch = 0

#         # step = False

#         # Handle button clicks
#         # if reset_button:
#         #     state, _ = env.reset()
#         #     desired_horizon = init_desired_horizon
#         #     desired_return = init_desired_return
#         #     epoch = 0
#         # elif pause_play_button:
#         #     st.session_state.paused = (
#         #         not st.session_state.paused
#         #     )  # Toggle paused state
#         # elif next_button and st.session_state.paused:
#         #     step = True
#         # elif save_button:
#         #     # Save image and info using Streamlit
#         #     st.image(
#         #         env_render, caption=f"Epoch {epoch}", use_column_width=True
#         #     )
#         #     st.write(
#         #         {
#         #             "state": {i: str(val) for i, val in enumerate(state)},
#         #             "feature": {
#         #                 i: str(val) for i, val in enumerate(summed_importances)
#         #             },
#         #             "action": str(action),
#         #             "reward": str(reward),
#         #             "desired_return": str(desired_return + reward),
#         #             "desired_horizon": str(desired_horizon + 1),
#         #         }
#         #     )

#     env.close()


# env = "Acrobot-v1"
# desired_return = -79
# desired_horizon = 82
# max_epoch = 500


# policy = SklearnPolicy.load("policy")
# hyper = AgentHyper(
#     env,
#     warm_up=0,
# )

# agent = UpsideDownAgent(hyper, policy)

# run_visualization(
#     env, agent, desired_return, desired_horizon, max_epoch, "data/viz_examples"
# )
