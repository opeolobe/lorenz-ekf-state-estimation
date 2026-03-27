# State Estimation of the Lorenz Attractor Using an Extended Kalman Filter

## Overview
This repository implements a state estimation framework for the Lorenz attractor model using the Extended Kalman Filter (EKF).

The project includes simulation of the nonlinear Lorenz system and estimation of its states from noisy measurements.

---

## Objectives
- Simulate the dynamics of the Lorenz attractor
- Perform nonlinear state estimation using the EKF
- Evaluate the statistical consistency of the EKF (e.g., NIS, NEES)

---

## Features
- Nonlinear Lorenz attractor model
- Generic Extended Kalman Filter implementation
- State estimation under measurement noise
- Statistical consistency analysis

---

## Project Structure
ekf.py                # Generic Extended Kalman Filter implementation
lorenz.py             # Lorenz system dynamics
run_lorenz_ekf.py     # Script to run simulation and state estimation

---


## Author
Opeoluwa Adebayo
GitHub: https://github.com/opeolobe


## Usage

Run the EKF state estimation:

```bash
python run_lorenz_ekf.py


