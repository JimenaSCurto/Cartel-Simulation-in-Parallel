# Cartel-Simulation-in-Parallel
 simulates a drug cartel's operations using multithreaded Python, SQLite for event logging, and pandas/Seaborn for analysis. It models production, processing, delivery, and law enforcement via concurrent threads, stores events in a normalized database, and generates visual insights into system dynamics

## Disclaimer

This project is intended solely for technical, educational, and research purposes. It does not promote or condone any illegal activity.

# Cartel Simulation System

This project is a multithreaded Python simulation that models the operational structure of a fictional drug cartel, covering activities from raw material production to delivery logistics under the threat of law enforcement. Events are logged to a SQLite database and later analyzed through a dedicated data analysis module that produces statistical summaries and visualizations.

## Features

### Core Components

* **Production Threads**: Simulate harvesting events with probabilistic outcomes (successes and failures).
* **Processing Threads**: Model drug conversion processes with success probabilities and potential failures (e.g., lab explosions).
* **Delivery Threads**: Simulate shipments with variable success depending on law enforcement presence and logistical constraints.
* **Law Enforcement Events**: Represent disruptions caused by periodic police actions, seizures, and raids.
* **Event Logging**: Each event is timestamped, assigned a simulation ID, and stored in a relational database for post-simulation analysis.

### Concurrency and System Architecture

* Built using Python’s threading module to simulate concurrent operations.
* Event execution is managed through a queue system to ensure orderly processing and avoid race conditions.
* All threads operate independently while pushing results to a centralized logger and database handler.

## Database Structure

The SQLite database (`cartel_simulation.db`) includes the following normalized tables:

* `SimulationRuns`: Metadata for each simulation session.
* `ProductionEvents`: Records of harvesting attempts, including outcome, quantity, and location.
* `ProcessingEvents`: Lab processing events with status indicators.
* `DeliveryEvents`: Shipment records with cargo amount, route information, and success/failure details.
* `LawEnforcementEvents`: Details on busts, disruptions, and their geographic impact.

Each event is linked to a unique simulation ID to facilitate segmented analysis.

## Analysis Module

The companion script `analysis.py` is used to query and analyze the simulation data. It uses the pandas, matplotlib, and seaborn libraries to provide detailed statistical reports and save visualizations to disk.

### Analysis Functions

* **Production Analysis**: Evaluates success/failure rates and production volume by region.
* **Processing Analysis**: Assesses conversion rates and distribution of processing failures.
* **Delivery Analysis**: Summarizes shipment success, loss rates, and delivery times.
* **Law Enforcement Analysis**: Measures the frequency and impact of police actions on operations.

Visual outputs are stored in the `charts/` directory.

## Workflow

1. Execute the simulation to generate and log events.
2. Use the analysis module to extract insights from the stored data.
3. Review generated visualizations and summary statistics for operational evaluation.

## Dependencies

* Python 3.7+
* pandas
* matplotlib
* seaborn
* scipy (optional, used for statistical modeling)

## Design Principles

* Emphasis on modular, testable simulation code.
* Realistic behavior modeled via randomness and concurrency.
* Persistent data storage enables reproducible analysis.
* Extensible architecture for future enhancements (e.g., economic modeling, AI agents, rival cartels).

## Future Enhancements

* Introduce agent-based behavior with individual actor decision-making.
* Add real-time visualization through an interactive dashboard.
* Implement economic dynamics such as pricing, demand, and supply shocks.
* Expand law enforcement logic to include patrol zones and intelligence.
