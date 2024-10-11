import streamlit as st


st.image("logo.jpg")

st.html(
    """<div>
<h1>Upside-Down Reinforcement Learning for More Interpretable Optimal Control</h1>

    <div class="abstract">
        <h2>Abstract</h2>
        <p>This research introduces a novel approach to reinforcement learning that emphasizes interpretability and explainability. By leveraging tree-based methods within the Upside-Down Reinforcement Learning (UDRL) framework, we demonstrate that it's possible to achieve performance comparable to neural networks while gaining significant advantages in terms of interpretability.</p>
    </div>

    <h2>What is Upside-Down Reinforcement Learning?</h2>
    <p>UDRL is an innovative paradigm that transforms reinforcement learning problems into supervised learning tasks. Unlike traditional approaches that focus on predicting rewards or learning environment models, UDRL learns to predict actions based on:</p>
    <ul>
        <li>Current state (s<sub>t</sub>)</li>
        <li>Desired reward (d<sub>r</sub>)</li>
        <li>Time horizon (d<sub>t</sub>)</li>
    </ul>

    <h2>Motivation</h2>
    <p>While neural networks have been the go-to choice for implementing UDRL, they lack interpretability. Our research explores whether other supervised learning algorithms, particularly tree-based methods, can:</p>
    <ul>
        <li>Match the performance of neural networks</li>
        <li>Provide more interpretable policies</li>
        <li>Enhance the explainability of reinforcement learning systems</li>
    </ul>

    <div class="results">
        <h2>Results</h2>
        <p>We tested three different implementations of the Behaviour Function:</p>
        <ul>
            <li>Neural Networks (NN)</li>
            <li>Random Forests (RF)</li>
            <li>Extremely Randomized Trees (ET)</li>
        </ul>
        <p>Tests were conducted on three popular OpenAI Gym environments:</p>
        <ul>
            <li>CartPole</li>
            <li>Acrobot</li>
            <li>Lunar-Lander</li>
        </ul>

    </div>

    <h2>Key Findings</h2>
    <ul>
        <li>Tree-based methods performed comparably to neural networks</li>
        <li>Random Forests and Extremely Randomized Trees provided fully interpretable policies</li>
        <li>Feature importance analysis revealed insights into decision-making processes</li>
    </ul>

    <h2>Implications</h2>
    <p>This research opens new avenues for:</p>
    <ul>
        <li>More explainable reinforcement learning systems</li>
        <li>Enhanced safety in AI decision-making</li>
        <li>Better understanding of agent behavior in complex environments</li>
    </ul>
    </div>
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
        h1, h2 {
            color: #81a1c1;
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
