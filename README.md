# ROBa_Project

## Introduction

The goal of this project is to implement Monte Carlo Localization (MCL) to enable a robot to estimate its position within a given environment by integrating sensors.

## Features

- **Agent Control**: Uses Q-Learning algorithm to make decisions and learn which action is the most beneficial. (see `agent.py`).
- **Helper Utilities**: Display the agent's performance in real time, including the training progress, score and mean scores over multiple games. (`helper.py`).
- **Model Management**: Defines the neural network model and the training process for the reinforcement learning agent which uses Q-Learning (`model.py`, `model` directory).
- **Robotics Motion Simulation**: Simulate and control the robot movments, interaction with obstacles and readings of its surrounding. (`robot_motions.py`).

## Installation

1. Clone this repository:
    git clone https://github.com/nicolas-bock/ROBa_Project.git
2. Navigate to the project directory:
    cd ROBa_Project
3. Install the required dependencies:
    pip install -r requirements.txt
   *Note*: Ensure you have Python 3.8 or higher installed.

## Usage

1. Run the main agent script:
    python agent.py

## Contributing

To contribute to the project, here's the following instructions:

1. Fork this repository.
2. Create a new branch:
    git checkout -b feature-name
3. Commit your changes:
    git commit -m "Add a descriptive message"
4. Push your branch and submit a pull request.

