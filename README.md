# Tandem Queueing System Simulation

This repository contains a Python implementation of a tandem queueing system simulation, designed for an Operations Research (OR) course. The simulation models the behavior of two single-server queues in series, where customers arrive at the first queue, get served, move to the second queue, get served again, and finally leave the system.

## Project Overview

The project aims to:

* **Simulate a tandem queueing system:** Implement a SimPy-based simulation to model the arrival, service, and departure of customers in a two-queue system.
* **Analyze system performance:** Calculate the time average number of customers in the system and compare the results with theoretical predictions from queueing theory.
* **Visualize queue dynamics:** Generate plots of queue lengths over time to visualize the behavior of the system.

## Code Structure

The code is organized into the following classes:

* **`Queue1`:** Represents the first queue in the tandem system.
* **`Queue2`:** Represents the second queue in the tandem system.

The main simulation logic is implemented in the `run` function, which:

* Sets up the simulation environment.
* Creates the two queues with specified parameters.
* Runs the simulation for a specified time horizon.
* Calculates the time average number of customers in the system.
* Generates plots of queue lengths over time (for specific simulation scenarios).

## Usage

To run the simulation:

1. **Install SimPy:** `pip install simpy`
2. **Execute the script:** `python project.py`

The code will run multiple simulations with different parameter combinations and print the results. It will also generate plots for the scenario with 1000 initial customers and a simulation time of 2000.

## Parameters

The simulation can be configured with the following parameters:

* **`lambda_`:** Arrival rate of customers (λ).
* **`mu1`:** Service rate of the server in the first queue (μ1).
* **`mu2`:** Service rate of the server in the second queue (μ2).
* **`capacity1`:** Capacity of the server in the first queue (number of customers it can serve simultaneously).
* **`capacity2`:** Capacity of the server in the second queue (number of customers it can serve simultaneously).
* **`q`:** Initial number of customers in the first queue.
* **`T`:** Simulation time horizon.
* **`N`:** Number of simulation runs to perform.

## Results

The simulation results include:

* **Average time average customers:** The average number of customers in the system over all simulation runs.
* **Standard deviation:** The standard deviation of the time average customer counts.
* **Queue length plots:** Plots of queue lengths over time for specific simulation scenarios.

## Future Work

* **Steady-State Analysis:** Implement the steady-state product form formulas for the expected number of customers in the system (Eπ[L1 + L2]) and compare the results with the simulation.
* **More Complex Queueing Systems:** Extend the simulation to handle more complex queueing systems, such as multi-server queues, queues with different arrival and service distributions, or queues with feedback loops.
* **Performance Optimization:** Explore techniques to optimize the simulation performance, such as using different random number generators or reducing the number of events tracked.

## Contributing

Contributions to this project are welcome! Feel free to submit pull requests with bug fixes, improvements, or new features.
