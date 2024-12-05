import streamlit as st


st.image("logo.jpg")

st.html(
    """
<body>
    <div class="container">
            <h1 >Upside-Down Reinforcement Learning for More Interpretable Optimal Control</h1>
            <div class="authors" style="text-align: center;">
                <p class="author-names">Juan Cardenas-Cartagena, Massimiliano Falzari, Marco Zullich, Matthia Sabatelli</p>
                <p class="institution">Bernoulli Institute, University of Groningen, The Netherlands</p>
            </div>
<h2><a href="https://arxiv.org/abs/2411.11457" target="_blank">Read the full paper on arXiv</a></h2>

        <section class="motivation">
            <h2>Research Motivation</h2>
            <p>The dramatic growth in adoption of Neural Networks (NNs) within the last 15 years has sparked a crucial need for increased transparency, especially in high-stake applications. While NNs have demonstrated remarkable performance across various domains, they are essentially black boxes whose decision-making processes remain opaque to human understanding. This research addresses this fundamental challenge by exploring alternative approaches that maintain performance while dramatically improving interpretability.</p>

            <div class="key-challenges">
                <h3>Current Challenges in Reinforcement Learning</h3>
                <p>Traditional approaches to Reinforcement Learning (RL) face several key limitations:</p>
                <ul>
                    <li>Complex neural network policies are difficult to interpret and explain</li>
                    <li>Lack of transparency in decision-making processes poses risks in critical applications</li>
                    <li>Traditional RL approaches either focus on predicting rewards or learning environment models, making interpretation challenging</li>
                    <li>The gap between performance and interpretability has been difficult to bridge</li>
                </ul>
            </div>
        </section>

        <section class="udrl-framework">
            <h2>The UDRL Framework: A Novel Approach</h2>
            <p>Upside-Down Reinforcement Learning represents a fundamental shift in how we approach reinforcement learning problems. Instead of traditional methods that focus on predicting rewards or learning environment models, UDRL transforms the reinforcement learning problem into a supervised learning task.</p>

            <div class="framework-details">
                <h3>Key Components</h3>
                <p>The UDRL approach centers around learning a behavior function f(st, dr, dt) = at where:</p>
                <ul>
                    <li><strong>st</strong>: The current state of the environment</li>
                    <li><strong>dr</strong>: The desired reward the agent aims to achieve</li>
                    <li><strong>dt</strong>: The time horizon within which to achieve the reward</li>
                    <li><strong>at</strong>: The action to take to achieve the desired reward</li>
                </ul>
            </div>

            <div class="mathematical-framework">
                <h3>Mathematical Foundation</h3>
                <p>The framework is built on a Markov Decision Process (MDP) defined as a tuple M = ⟨S,A,P,R⟩ where:</p>
                <ul>
                    <li>S: The state space of the environment</li>
                    <li>A: The action space modeling all possible actions</li>
                    <li>P: The transition function P : S × A × S → [0,1]</li>
                    <li>R: The reward function R : S × A × S → R</li>
                </ul>
            </div>
        </section>

        <section class="implementation">
            <h2>Implementation and Methodology</h2>
            <div class="algorithms-detailed">
                <h3>Studied Algorithms</h3>
                <div class="tree-based">
                    <h4>Tree-Based Methods</h4>
                    <p>We extensively evaluated two primary tree-based approaches:</p>
                    <ul>
                        <li><strong>Random Forests (RF):</strong> An ensemble method building multiple decision trees and merging their predictions</li>
                        <li><strong>Extremely Randomized Trees (ET):</strong> A variation that adds additional randomization in the tree-building process</li>
                    </ul>
                </div>

                <div class="boosting">
                    <h4>Boosting Algorithms</h4>
                    <p>We also investigated sequential ensemble methods:</p>
                    <ul>
                        <li><strong>AdaBoost:</strong> Adaptive Boosting for sequential tree construction</li>
                        <li><strong>XGBoost:</strong> A more advanced implementation of gradient boosting</li>
                    </ul>
                </div>

                <div class="baseline">
                    <h4>Baseline Methods</h4>
                    <ul>
                        <li><strong>Neural Networks:</strong> Traditional multi-layer perceptron architecture</li>
                        <li><strong>K-Nearest Neighbours:</strong> Non-parametric baseline for comparison</li>
                    </ul>
                </div>
            </div>

            <div class="experimental-environments">
                <h3>Test Environments</h3>
                <div class="cartpole">
                    <h4>CartPole</h4>
                    <p>A 4-dimensional continuous state space including:</p>
                    <ul>
                        <li>Cart Position (x)</li>
                        <li>Cart Velocity (ẋ)</li>
                        <li>Pole Angle (θ)</li>
                        <li>Pole Angular Velocity (θ̇)</li>
                    </ul>
                </div>

                <div class="acrobot">
                    <h4>Acrobot</h4>
                    <p>A 6-dimensional state space representing:</p>
                    <ul>
                        <li>First Link: sin(θ1), cos(θ1), θ̇1</li>
                        <li>Second Link: sin(θ2), cos(θ2), θ̇2</li>
                    </ul>
                </div>

                <div class="lunar-lander">
                    <h4>Lunar Lander</h4>
                    <p>An 8-dimensional state space including:</p>
                    <ul>
                        <li>Position (x, y)</li>
                        <li>Velocity (ẋ, ẏ)</li>
                        <li>Angle (θ) and Angular velocity (θ̇)</li>
                        <li>Left and right leg contact points</li>
                    </ul>
                </div>
            </div>
        </section>

        <section class="results">
            <h2>Comprehensive Results and Analysis</h2>
            <div class="performance-analysis">
                <h3>Performance Metrics</h3>
                <p>Our experiments revealed surprising competitiveness of tree-based methods:</p>
                <ul>
                    <li><strong>CartPole Environment:</strong>
                        <ul>
                            <li>Neural Networks: 199.93 ± 0.255</li>
                            <li>Random Forests: 188.25 ± 13.82</li>
                            <li>XGBoost: 199.27 ± 4.06</li>
                        </ul>
                    </li>
                    <li><strong>Acrobot Environment:</strong>
                        <ul>
                            <li>Neural Networks: -75.00 ± 15.36</li>
                            <li>Random Forests: -100.05 ± 62.80</li>
                            <li>Extra Trees: -100.00 ± 93.72</li>
                        </ul>
                    </li>
                    <li><strong>Lunar Lander:</strong>
                        <ul>
                            <li>Random Forests: -54.74 ± 96.22</li>
                            <li>XGBoost: -76.96 ± 89.69</li>
                            <li>Neural Networks: -157.04 ± 71.26</li>
                        </ul>
                    </li>
                </ul>
            </div>

            <div class="interpretability-analysis">
                <h3>Interpretability Insights</h3>
                <p>The tree-based methods provided unprecedented insights into decision-making:</p>
                <ul>
                    <li><strong>CartPole:</strong> Pole angular velocity emerged as the most crucial feature for balancing</li>
                    <li><strong>Acrobot:</strong> Angular velocities of both links proved essential for control</li>
                    <li><strong>Lunar Lander:</strong> Vertical position showed highest importance for landing decisions</li>
                </ul>
            </div>
        </section>

        <section class="future-directions">
            <h2>Future Research Directions</h2>
            <p>Our findings open several promising avenues for future research:</p>
            <ul>
                <li>Scaling to High-Dimensional Spaces:
                    <p>Investigating the applicability of tree-based UDRL to more complex environments with higher-dimensional state spaces</p>
                </li>
                <li>Enhanced Interpretation Tools:
                    <p>Development of specialized tools for analyzing and visualizing decision processes in tree-based UDRL systems</p>
                </li>
                <li>Real-World Applications:
                    <p>Exploring applications in safety-critical domains where interpretability is crucial</p>
                </li>
                <li>Theoretical Analysis:
                    <p>Deeper investigation of the theoretical foundations underlying the success of tree-based methods in UDRL</p>
                </li>
            </ul>
        </section>

    </div>
</body>
    """
)
st.html(
    """
   <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #e0e0e0;
            background-color: #1a1a1a;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h3 {
            text-align: center;
        }
        h1, h2 {
            color: #81a1c1;
            text-align: center;
        }
        .abstract {
            background-color: #2e3440;
            padding: 20px;
            border-left: 4px solid #88c0d0;
            margin: 20px 0;
            border-radius: 5px;
        }
        .results {
            margin-top: 30px;
        }
        .chart-container {
            margin-top: 20px;
            height: 400px;
            background-color: #2e3440;
            padding: 20px;
            border-radius: 5px;
        }
        .highlight {
            background-color: #4c566a;
            padding: 2px 5px;
            border-radius: 3px;
        }
        code {
            background-color: #3b4252;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', Courier, monospace;
        }
    </style>
    """
)


#     """<div>
# <h1>Upside-Down Reinforcement Learning for More Interpretable Optimal Control</h1>
#     <div class="abstract">
#         <h2>Abstract</h2>
#         <p>This research introduces a novel approach to reinforcement learning that emphasizes interpretability and explainability. By leveraging tree-based methods within the Upside-Down Reinforcement Learning (UDRL) framework, we demonstrate that it's possible to achieve performance comparable to neural networks while gaining significant advantages in terms of interpretability.</p>
#     </div>

#     <h2>What is Upside-Down Reinforcement Learning?</h2>
#     <p>UDRL is an innovative paradigm that transforms reinforcement learning problems into supervised learning tasks. Unlike traditional approaches that focus on predicting rewards or learning environment models, UDRL learns to predict actions based on:</p>
#     <ul>
#         <li>Current state (s<sub>t</sub>)</li>
#         <li>Desired reward (d<sub>r</sub>)</li>
#         <li>Time horizon (d<sub>t</sub>)</li>
#     </ul>

#     <h2>Motivation</h2>
#     <p>While neural networks have been the go-to choice for implementing UDRL, they lack interpretability. Our research explores whether other supervised learning algorithms, particularly tree-based methods, can:</p>
#     <ul>
#         <li>Match the performance of neural networks</li>
#         <li>Provide more interpretable policies</li>
#         <li>Enhance the explainability of reinforcement learning systems</li>
#     </ul>

#     <div class="results">
#         <h2>Results</h2>
#         <p>We tested three different implementations of the Behaviour Function:</p>
#         <ul>
#             <li>Neural Networks (NN)</li>
#             <li>Random Forests (RF)</li>
#             <li>Extremely Randomized Trees (ET)</li>
#         </ul>
#         <p>Tests were conducted on three popular OpenAI Gym environments:</p>
#         <ul>
#             <li>CartPole</li>
#             <li>Acrobot</li>
#             <li>Lunar-Lander</li>
#         </ul>

#     </div>

#     <h2>Key Findings</h2>
#     <ul>
#         <li>Tree-based methods performed comparably to neural networks</li>
#         <li>Random Forests and Extremely Randomized Trees provided fully interpretable policies</li>
#         <li>Feature importance analysis revealed insights into decision-making processes</li>
#     </ul>

#     <h2>Implications</h2>
#     <p>This research opens new avenues for:</p>
#     <ul>
#         <li>More explainable reinforcement learning systems</li>
#         <li>Enhanced safety in AI decision-making</li>
#         <li>Better understanding of agent behavior in complex environments</li>
#     </ul>
#     </div>
# """
