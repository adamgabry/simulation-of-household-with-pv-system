# PV Household Powerplant Simulation

## Description
This project is a simulation of a photovoltaic (PV) powerplant developed for a family house. It aims to model the behavior and performance of a PV powerplant under different conditions. The framework integrates functionalities to handle data extraction, work with threephase system, solar production calculations with providing a comprehensive toolset for analyzing and optimizing energy systems.

## Installation

### Prerequisites
Ensure you have Python 3.12 or higher installed on your system.

### Setup
Navigate to the project source directory:
```bash
cd source
```

### Installing Dependencies
Install all the necessary packages using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Usage
To run the simulation, execute the following command:
```bash
python simulation.py
```

## Simulation Start

When you run the simulation, you will be asked to provide some parameters:

1. **Battery**: You will be asked if you want to run the simulation with a battery. Enter 'yes' if you want to include a battery in the simulation, or 'no' if you don't.

2. **Off-Grid**: You will be asked if you want to run the simulation off-grid. Enter 'yes' if you want the simulation to be off-grid, or 'no' if you want it to be connected to the grid.

3. **Selling Back to the Grid**: You will be asked if you want to sell excess energy back to the grid. Enter 'yes' if you want to sell excess energy back to the grid, or 'no' if you don't.

Detailed workflow is described here:
[Go to Simulation Workflow](#simulation-workflow)

For simulation is also crucial setting parameters that fit your needs.
Here are the configs that can be edited

### Configuration Files Details

#### battery_config.json
- **battery_info**: General characteristics for different battery types.
  - `min_capacity` (float): Minimum capacity a battery can hold, expressed in kWh.
  - `max_capacity` (float): Maximum capacity for the battery, expressed in kWh.
  - `lifespan_years` (int): Estimated lifespan of the battery in years.
  - `cost_per_kWh_capacity` (float): Cost per kilowatt-hour capacity of the battery.
  - `annual_degradation` (float): Yearly percentage rate at which the battery degrades. Not used in the final experiments, but is ready for future development
- **battery_configs**: Specific configurations for batteries, useful for simulation parameters.
  - `type` (str): Type of the battery (e.g., LiFePO4, Lead-acid).
  - `size` (int or float): Capacity of the battery in Wh.
  - `cost` (int or float): Purchase cost of the battery.
  - `default` (bool): Whether this configuration is the default setting for simulation without optimizing.

#### database_config.json
- **database_config**: Contains parameters for database connectivity.
  - `consumption_db_path` (str): File path to the database storing consumption data.
  - `WeatherUnderground_path` (str or None): Path to the WeatherUnderground folder with weather data CSVs.
  - `get_new_consumption_data` (bool): Indicates whether to fetch new consumption data.
  - `sensor_ids` (dict): Mapping of sensor IDs to their respective phases.
    - `first_phase` (int): Sensor ID for the first phase.
    - `second_phase` (int): Sensor ID for the second phase.
    - `third_phase` (int): Sensor ID for the third phase.
  - `data_in_Watts` (bool): Indicates if the data is stored in Watts. If False it is stored in mA.

#### inverter_config.json
- **invertor_configs**: Configuration parameters for inverters, each detailing specific operational and cost-related properties.
  - `size` (int or float): Maximum output capacity of the inverter in watts, indicating how much power it can handle.
  - `efficiency` (float): Operational efficiency of the inverter, expressed as a percentage. This reflects how effectively the inverter converts DC to AC power.
  - `cost` (int): Purchase cost of the inverter, expressed in your local currency.
  - `default` (bool): Specifies whether this inverter configuration is the default setting.
  - `hybrid` (bool): Indicates whether the inverter is capable of both connecting to the grid and operating with battery. Only these inverters can be used for simulation with battery in this work.
  - `battery_loader_efficiency` (float): Efficiency with which the inverter charges batteries, <0,1>.
  - `asymetric` (bool): Whether the inverter can handle asymmetrical loading across its phases.
  - `max_single_phase_ratio` (float): The maximum ratio of total power that can be loaded onto a single phase without causing issues, expressed as a fraction of total capacity.

#### location_config.json
- **location**: Geographical settings that might influence the simulation.
  - `latitude` (float): The geographic latitude of the location.
  - `longitude` (float): The geographic longitude of the location.

#### low_tariff_periods_config.json
- **low_tariff_periods**: Times during which lower tariffs are applied.
  - `start` (str): Start time of the low tariff period, formatted as `HH:MM`.
  - `end` (str): End time of the low tariff period, formatted as `HH:MM`.

#### panel_techsheet_config.json
- **panel_parameters**: Technical specifications for solar panels, detailing performance characteristics for different cell types.
  - `celltype` (str): Type of solar cells used in the panels, such as monoSi, polySi, etc.
  - `pmax` (int or float): Maximum power output of the panels in watts, indicating the peak power that the panel can produce under ideal conditions.
  - `v_mp` (int or float): Voltage at maximum power, which is the voltage at which the panel produces maximum power output.
  - `i_mp` (int or float): Current at maximum power, the current at which maximum power is delivered.
  - `v_oc` (int or float): Open circuit voltage, which is the maximum voltage the panel can produce when not connected to an electrical circuit.
  - `i_sc` (int or float): Short circuit current, the current that flows when the panel's terminals are shorted.
  - `alpha_sc` (float): Temperature coefficient of the short-circuit current, showing how the current changes with temperature.
  - `beta_voc` (float): Temperature coefficient of the open-circuit voltage, showing how the voltage changes with temperature.
  - `gamma_pmax` (float): Power temperature coefficient, indicating how the maximum power output changes with temperature.
  - `cells_in_series` (int): Number of cells wired in series in the panel.
  - `temp_ref` (int or float): Reference temperature at which the panel's specifications are rated.
  - `price` (int or float): Retail price of the solar panel.
- **panel_degredation_yearly**: Annual degradation rates for each type of solar panel material, detailing the percentage decrease in efficiency per year.


#### panels_config.json
- Configurations for solar panel arrays.
  - `surface_tilt` (int or float): Tilt angle at which the panels are mounted.
  - `surface_azimuth` (int or float): Azimuth angle of the panels.
  - `strings` (int): Number of strings in the array.
  - `modules_per_string` (int): Number of modules per string.

#### prizing_energy_and_cost_config.json
- **prizing_energy_and_cost**: Details on energy pricing and associated costs.
  - `selling_cost_to_distributor_per_kwh` (float): Cost per kWh for selling energy to a distributor.
  - `buying_cost_from_distributor_per_kwh` (float): Cost per kWh for buying energy from a distributor.
  - `installation_cost` (int or float): Cost of installing the system.
  - `annual_expenses` (int or float): Annual operational and maintenance expenses.

## Simulation Workflow

### Initialization
1. **Configuration Loading**: The simulation starts by loading all necessary configurations from JSON files. These configurations include settings for batteries, inverters, solar panels, and other described above.
2. **Parameter Setting**: Default parameters for the inverter and battery are set based on the loaded configurations.

### Simulation Setup
3. **User Input**: The user is prompted to input simulation options such as including a battery, running in off-grid mode, handling low tariff periods, and enabling energy selling.
4. **Initial Status Display**: Before proceeding, the simulation prints the initial configuration based on the user's inputs.
5. **Data Handling**:
   - **Data Extraction**: If new consumption data is required, it is extracted from the specified databases.
   - **Data Resampling**: The consumption data is resampled to align with the simulation’s time steps.

### Simulation Execution
6. **Simulation Start**: The simulation process begins, using the setup parameters and user inputs.
7. **Solar Energy Calculation**: Solar energy production is calculated based on the location and panel configurations.
8. **Energy Processing**:
   - If a battery is included, battery dynamics are simulated considering the configured parameters.
   - Energy transactions (consumption and sales to the grid) are computed, especially focusing on tariff variations if enabled.

### Optimization (Optional)
9. **Optimization Prompt**: The user is asked whether to proceed with optimization, which aims to improve the efficiency and cost-effectiveness of the system.
    **NOTE** -always try to do optimization of ROI with real yearly data. Extrapolation has high impact on the consumption during various months.  

10. **Genetic Algorithm**: If proceeded with optimization, a genetic algorithm is run to find the best configuration of panels, battery, and inverter that yields the highest return on investment (ROI).
11. **Optimization Results**: The results, including the best configuration and the improved ROI, are displayed.

### Finalization

12. **Results Saving**: All final configurations and simulation results are saved in a JSON file for later review or documentation.



## Modules
Here is an overview of the key modules in the simulation:
- **Database Extraction**: Manages large datasets from databases.
- **Phases CSV Merging**: Merges various CSV files related to different simulation phases.
- **Phase Grouping**: Organizes data into phases for further analysis.
- **Data Extrapolation**: Extends consumption data -- used mainly for testing purposes of optimization, if possible always use just real data. But for optimization of ROI we need whole year data. 
- **Solar Production**: Calculates solar energy based on specific parameters.
- **PV Optimizer**: Optimizes PV system configurations.
- **Energy Processing**: Processes energy data for simulation purposes.

## License
This project is released under the MIT License.

