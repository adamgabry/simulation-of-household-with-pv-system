"""
@author: Adam Gabrys
@Description: This file contains the main simulation logic of running the photovoltaics for household microgrid simulation.
"""

from functools import partial
import os
import pandas as pd
import time

from database_extraction.large_database_extractor import LargeDatabaseExtractor
from phases_csv_merging.phases_csv_merger import CSVMerger
from phase_grouping.group_phases import PhaseGrouper
from data_extrapolation.data_extrapolation_phases import DataExtrapolation
from solar_production.solar_energy_from_spec_panel import SolarEnergyCalculator
from pv_optimizer.pv_optimizer import PVOptimizer
from energy_processing.energy_data_processor import EnergyDataProcessor
from parameter_handling.param_setter import ParamSetter

class Simulation:
    """
    Class created for running simulation logic as a whole.

    Attributes:
    - param_setter (ParamSetter): An instance of the ParamSetter class.
    - configs (dict): A dictionary containing the loaded configurations.
    - current_inverter (dict): The current inverter configuration.
    - current_battery (dict): The current battery configuration.
    - db_extractor (LargeDatabaseExtractor): An instance of the LargeDatabaseExtractor class.
    - with_battery (bool): Indicates whether the simulation includes a battery.
    - off_grid (bool): Indicates whether the simulation is run in off-grid mode.
    - handle_low_tariff (bool): Indicates whether the simulation handles low tariff periods.
    - selling_enabled (bool): Indicates whether selling energy back to the grid is enabled.
    - max_blackout_days (int): The maximum number of blackout days.
    - battery_configs (dict): The battery configurations.
    - battery_info (dict): The battery information.
    - low_tariff_periods (list): A list of low tariff periods.
    - invertor_configs (dict): The inverter configurations.
    - prizing_energy_and_cost (dict): The prizing energy and cost configurations.
    - location (dict): The location configuration.
    - database_config (dict): The database configuration.
    - consumption_db_path (str): The path to the consumption database.
    - WeatherUnderground_path (str): The path to the WeatherUnderground database.
    - consumption_sensors (dict): The consumption sensors.
    - data_in_Watts (bool): Indicates whether the data is in Watts.
    - get_new_consumption_data (bool): Indicates whether to get new consumption data.
    - panel_degredation_yearly (float): The panel degradation rate per year.
    - current_panel_params (dict): The current panel parameters.
    - panel_array_configs (dict): The panel array configurations.
    - battery_loader_efficiency (float): The battery loader efficiency.
    - max_single_phase_ratio (float): The maximum single phase ratio.
    - initial_configuration (dict): The initial configuration.
    - debug (bool): Indicates whether debug mode is enabled.
    """

    def __init__(self):
        self.param_setter = ParamSetter()
        self.configs = self.load_configs()
        self.setup_configurations()
        self.current_inverter = self.param_setter.set_default_inverter(self.invertor_configs)
        self.current_battery = self.param_setter.set_default_battery(self.battery_configs)
        self.db_extractor = LargeDatabaseExtractor(self.consumption_db_path)

    def print_initial_status(self):
        """
        Prints the initial status of the simulation.

        Parameters:
        - args (dict): The simulation arguments.
        """
        print("Running simulation with these settings: \n")
        print(f"Running with battery: {self.with_battery}")
        if self.with_battery:
            print(f"Off-grid mode: {self.off_grid}")
            if self.off_grid:
                print(f"Max blackout days: {self.max_blackout_days}")
        print(f"Handle low tariff: {self.handle_low_tariff}")
        print(f"Selling enabled: {self.selling_enabled}")

    def load_configs(self):
        file_paths = [
            'parameter_handling/configs/battery_config.json',
            'parameter_handling/configs/low_tariff_periods_config.json',
            'parameter_handling/configs/panel_techsheet_config.json',
            'parameter_handling/configs/panels_config.json',
            'parameter_handling/configs/inverter_config.json',
            'parameter_handling/configs/location_config.json',
            'parameter_handling/configs/database_config.json',
            'parameter_handling/configs/prizing_energy_and_cost_config.json'
        ]
        return self.param_setter.load_multiple_json_configs(file_paths)

    def set_simulation_arguments(self, args):
        # This method was used for testing with hardcoded values
        self.with_battery = args['with_battery']
        self.off_grid = args['off_grid']
        self.handle_low_tariff = args['handle_low_tariff']
        self.selling_enabled = args['selling_enabled']
        self.max_blackout_days = args['max_blackout_days']

    def get_user_input(self):
        print("---\nStarting the run of photovoltaics for household microgrid simulation\n---")
        with_battery = input("Include a battery in the simulation? (y/n): ").lower() == 'y'
        off_grid = False  # Default to false
        max_blackout_days = 365 
        selling_enabled = False  # Default to false
        
        if with_battery:
            off_grid = input("Run the simulation in off-grid mode? (y/n): ").lower() == 'y'
            if off_grid:
                max_blackout_days = int(input("Set the maximum number of blackout days (default 365): ") or "365")
        handle_low_tariff = input("Handle computing low tariff periods? (NOTE: not used in optimalization) (y/n): ").lower() == 'y'
        if not off_grid:
            selling_enabled = input("Enable selling energy back to the grid? (y/n): ").lower() == 'y'

        self.with_battery = with_battery
        self.off_grid = off_grid
        self.handle_low_tariff = handle_low_tariff
        self.selling_enabled = selling_enabled
        self.max_blackout_days = max_blackout_days
        return {
            "with_battery": with_battery,
            "off_grid": off_grid,
            "handle_low_tariff": handle_low_tariff,
            "selling_enabled": selling_enabled,
            "max_blackout_days": max_blackout_days
        }

    def setup_configurations(self):
        self.battery_configs = self.configs['battery_config']['battery_configs']
        self.battery_info = self.configs['battery_config']['battery_info']
        self.low_tariff_periods = self.param_setter.convert_periods_to_tuples(self.configs['low_tariff_periods_config']['low_tariff_periods'])
        self.invertor_configs = self.configs['inverter_config']['invertor_configs']
        
        self.prizing_energy_and_cost = self.configs['prizing_energy_and_cost_config']['prizing_energy_and_cost']
        self.location = self.configs['location_config']['location']
        
        self.database_config = self.configs['database_config']['database_config']
        self.query = self.database_config['db_query']
        self.consumption_db_path = self.database_config['consumption_db_path']
        self.WeatherUnderground_path = self.database_config['WeatherUnderground_path']
        self.consumption_sensors = self.database_config['sensor_ids']
        self.data_in_Watts = self.database_config['data_in_Watts']
        self.get_new_consumption_data = self.database_config['get_new_consumption_data']

        self.panel_degredation_yearly = self.configs['panel_techsheet_config']['panel_degredation_yearly']
        self.current_panel_params = self.configs['panel_techsheet_config']['panel_parameters']['current_config']
        self.panel_array_configs = self.configs['panels_config']

        # Initialize default inverter and battery
        self.current_inverter = self.param_setter.set_default_inverter(self.invertor_configs)
        self.current_battery = self.param_setter.set_default_battery(self.battery_configs)

        self.battery_loader_efficiency = self.current_inverter.get('battery_loader_efficiency', 0.985)
        self.max_single_phase_ratio = self.current_inverter.get('max_single_phase_ratio', 0.5)

        self.debug = False

    def setup_initial_configuration(self, args):
        self.initial_configuration = {
            "arguments": args,
            "panel_configuration": self.panel_array_configs,
            "battery_configuration": self.current_battery,
            "inverter_configuration": self.current_inverter,
            "other_settings": {
                "battery_included": args['with_battery'],
                "off_grid": args['off_grid'],
                "handle_low_tariff": args['handle_low_tariff'],
                "selling_enabled": args['selling_enabled'],
                "max_blackout_days": args['max_blackout_days']
            },
            "prizing_energy_and_cost": self.prizing_energy_and_cost,
            "panel_cost": self.current_panel_params['price']
        }
        self.param_setter.save_configuration(self.initial_configuration)

    def handle_data_extraction(self):
        try:
            # Attempt to extract data for all phases at once
            self.db_extractor.extract_data(
                (self.consumption_sensors['first_phase'], 
                    self.consumption_sensors['second_phase'], 
                    self.consumption_sensors['third_phase']),
                query=self.query,
                output_file='other/extracted_data_utc_all.csv'
            )
            print('Data extracted successfully')
        except Exception as e:
            print(e)
            # Extract data for each phase individually if the combined extraction fails
            for phase in ['first_phase', 'second_phase', 'third_phase']:
                self.db_extractor.extract_data(
                    self.consumption_sensors[phase],
                    query=self.query,
                    output_file=f'other/extracted_data_utc_{phase}.csv'
                )

            # Merge the extracted data into one CSV file
            csv_files = ['other/extracted_data_utc_first_phase.csv', 
                            'other/extracted_data_utc_second_phase.csv', 
                            'other/extracted_data_utc_third_phase.csv']
            output_csv_path = 'other/extracted_data_utc_all.csv'
            merger = CSVMerger(csv_files, output_csv_path)
            merger.merge_and_sort_csv()

        # Process the extracted or merged data
        self.process_and_resample_data(output_csv_path)

    def process_and_resample_data(self, data_file_path):
        grouper = PhaseGrouper(
            sensor_id_first_phase=self.consumption_sensors['first_phase'],
            sensor_id_second_phase=self.consumption_sensors['second_phase'],
            sensor_id_third_phase=self.consumption_sensors['third_phase'],
            file_path=data_file_path
        )
        grouped_data = grouper.group_phases()
        consumption_data_df = grouper.process_grouped_data(grouped_data, phases_in_Watts=self.data_in_Watts)
        freq = grouper.get_freq(consumption_data_df)

        if self.debug:
            print(freq)

        df_resampled = consumption_data_df.resample(freq).mean()
        df_resampled.interpolate(method='linear', inplace=True)

        if self.debug:
            print(df_resampled.head(20))
            grouper.plot_phases(df_resampled)

        df_resampled.to_pickle('consumption_data/df_resampled.pkl')
        df_resampled.to_csv('consumption_data/consumption_data.csv')

        data_extrapolator = DataExtrapolation()
        full_year_df = data_extrapolator.extrapolate_last_month(df_resampled, ['phase_1', 'phase_2', 'phase_3'])
        full_year_df.to_pickle('consumption_data/full_year_df.pkl')
        full_year_df.to_csv('consumption_data/full_year_df.csv')

        if self.debug:
            data_extrapolator.plot_extrapolated_data(full_year_df, 3)

    def validate_inverter(self):
        if self.with_battery and not ('hybrid' in self.current_inverter and self.current_inverter['hybrid']):
            raise ValueError("The current default inverter is not hybrid. Cannot run the simulation with battery.")

    def load_full_year_data(self):
        """
        Loads the full year data from a pickle file or falls back to a CSV file if necessary.
        Returns the DataFrame if successful, or None if not.
        """
        file_path_pickle = 'consumption_data/full_year_df.pkl'
        file_path_csv = 'consumption_data/full_year_df.csv'
        try:
            if os.path.exists(file_path_pickle):
                return pd.read_pickle(file_path_pickle)
            elif os.path.exists(file_path_csv):
                return pd.read_csv(file_path_csv, index_col=0, parse_dates=True)
            else:
                print("No data file found.")
                return None
        except Exception as e:
            print(f"Failed to load data: {e}")
            return None

    def get_consumption_year_number(self, full_year_df):
        return full_year_df.index[1000].year

    def print_simulation_results(self, processor, solar_calculator):
        print("Generated DC energy in kWh: ")
        print((solar_calculator.total_dc_output.sum()/1000))
        print("Generated AC energy in kWh: ")
        print((solar_calculator.total_ac_output.sum()/1000))
        print("Consumed energy from grid without use of photovoltaics in kWh: ")
        print((processor.total_consumed_energy_without_PV/1000))
        print("Consumed energy from grid with use of photovoltaics system in kWh: ")
        print((processor.total_consumed_energy_with_PV/1000))
        print("Energy sold to grid in kWh: ")
        print((processor.energy_sold/1000))
        
    def initialize_solar_energy_calculator(self, consumption_data_year):

        solar_calculator = SolarEnergyCalculator(
            latitude=self.location['latitude'],
            longitude=self.location['longitude'],
            array_configs = self.panel_array_configs,  
            start=f"{consumption_data_year}-01-01 00:00",
            end=f"{consumption_data_year}-12-31 23:00",
            input_csvs_file_path=self.WeatherUnderground_path,
            panel_info=self.current_panel_params,
            invertor_threshold=self.current_inverter['nominal_power'],
            invertor_efficiency=self.current_inverter['efficiency'],
        )
        return solar_calculator

    def print_low_tarif_energy_amounts(self,processor):
        if self.handle_low_tariff:
            tariff_consumption = processor.calculate_tariff_consumption(self.low_tariff_periods)
            low_tariff_consumption = tariff_consumption.get('low', 0)
            high_tariff_consumption = tariff_consumption.get('high', 0)
            print("Low tariff energy consumption:", low_tariff_consumption/1000, "kWh")
            print("High tariff energy consumption:", high_tariff_consumption/1000, "kWh")

    def get_sampling_frequency(self, df):
        """
        Determine the most common sampling frequency in a DataFrame based on the first 1000 time deltas.
        
        Args:
        df (pd.DataFrame): DataFrame with a datetime index.
        
        Returns:
        str: A string representation of the frequency.
        """
        # Calculate the differences between successive timestamps
        time_deltas = df.index.to_series().diff().iloc[1:1001]  # Skip the first NaN, calculate for first 1000
        
        # Find the most common time delta
        mode_delta = time_deltas.mode()[0]  # mode() returns a Series, take the first
        
        # Convert the time delta to a frequency string recognizable by pandas
        freq = pd.tseries.frequencies.to_offset(mode_delta)
        
        if freq is None:
            return ValueError("Could not determine the frequency.")
        return freq.freqstr

    def set_days_without_blackout_if_off_grid(self, off_grid, pv_optimizer):
        if off_grid:
            pv_optimizer.days_without_blackout = self.max_blackout_days

    def visualize_solar_energy_data(self, solar_calculator):
        print("\nYou have two options for visualizing the solar energy data: \n")
        print("1. 'all' - This will display all graphs and print stats related to solar energy data.")
        print("2. 'power' - This will display only the power output graphs.")
        print("If you don't enter anything, the default will be 'power'. \n")
        visualize_solar_all = input("Please enter your choice (all/power): ").lower() == 'all'
        if visualize_solar_all:
            solar_calculator.plot_dhi_dni_ghi() if self.debug else None # To show the difference between DHI, DNI and GHI from WeatherUnderground and PVGIS
            solar_calculator.plot_eff_irradiation()
            solar_calculator.plot_energy_output(plot_months=True)
            solar_calculator.print_monthly_output()
            solar_calculator.print_annual_output()
        solar_calculator.plot_configurations()
        solar_calculator.plot_output()
    
    def count_blackout_days(self, processor):
        if self.off_grid:
            blackout_days = processor.blackout_days_count()
            print(f"Blackout days: {blackout_days}") if self.debug else None

    def visualize_energy_processor_data(self, processor):
        visualize_energy = input("Visualize graphs of energy data? (y/n): ").lower() == 'y'
        if visualize_energy:
            processor.statistics_show(off_grid=self.off_grid)
            processor.demo_plot_battery_state()
            processor.plot_phases_data()
            processor.plot_monthly_and_hourly_consumption(hours_only=True)
            processor.plot_monthly_and_hourly_consumption(months_only=True)
            if self.off_grid:
                processor.plot_blackout_duration()
            visualize_energy = input("Visualize all other graphs of energy data? (y/n): ").lower() == 'y'
            if visualize_energy:
                processor.plot_generated_power_per_phase_allocation()
                processor.plot_power_needs()
                processor.plot_energy_delta()

    def resample_data_to_new_frequency(self, df, new_frequency):
        """
        Resample a DataFrame to a new frequency.
        
        Args:
        df (pd.DataFrame): The DataFrame to resample.
        new_frequency (str): The new frequency as a string.
        
        Returns:
        pd.DataFrame: The resampled DataFrame.
        """
        if new_frequency == df.index.freqstr:
            return df
        return df.resample(new_frequency).mean()
    
    def set_simulation_consumption_data_precision(self, consumption_data_frequency):
        """
        Adjust the frequency of simulation data based on user-selected precision,
        without exceeding a maximum frequency of 30 minutes.

        Args:
        base_frequency (str): The baseline frequency to adjust from.

        Returns:
        str: The adjusted frequency string.
        """
        # Convert the base frequency to a pandas Timedelta for easy manipulation
        base_timedelta = pd.to_timedelta(pd.tseries.frequencies.to_offset(consumption_data_frequency))

        # User selects the precision level
        print("\n\n")
        print("Please set data sampling precision:\nNOTE: The higher precision level will slower the speed and increase accuracy. \n")
        print("high: The highest(original) sampling possible from your data")
        print("medium: upsampled with multiply of 2 of the original sampling")
        print("low: upsampled with multiply of 10 of the original sampling")
        print("minimum: upsampled to 30 minutes")
        print("default: minimum \n")
        precision = input("Choose the data sampling precision (high/medium/low/minimum): ").lower()

        # Calculate new frequencies based on the precision level
        if precision == "most":
            # Most detailed level, use the base frequency
            new_frequency = base_timedelta
        elif precision == "medium":
            # Slightly less detailed, double the interval
            new_frequency = base_timedelta * 2
        elif precision == "low":
            # Normal detail, multiply the interval by 10
            new_frequency = base_timedelta * 10
        else:
            # Least detail
            new_frequency = pd.Timedelta(minutes=30)  # Default to 30 minutes
        
        # Convert the new frequency to a string and check if it exceeds 30 minutes
        max_frequency = pd.to_timedelta('30min')
        if new_frequency > max_frequency:
            new_frequency = max_frequency  # Ensure it does not exceed 30 minutes

        # Return the new frequency as a string suitable for resampling
        return pd.tseries.frequencies.to_offset(new_frequency).freqstr
    
    def initialize_energy_data_processor(self, consumption_data_df, solar_calculator):
        processor = EnergyDataProcessor(
                                    self.current_battery['size'],
                                    self.current_battery['type'],
                                    buying_cost_per_kWh=self.prizing_energy_and_cost['buying_cost_from_distributor_per_kwh'],
                                    selling_cost_per_kWh=self.prizing_energy_and_cost['selling_cost_to_distributor_per_kwh'],
                                    invertor_threshold=self.current_inverter['nominal_power'],
                                    battery_info=self.battery_info,
                                    inverter_efficiency=self.current_inverter['efficiency'],
                                    phases_data_frame=consumption_data_df,
                                    dc_generation_data_frame=solar_calculator.total_dc_output,
                                    ac_generation_data_frame=solar_calculator.total_ac_output,
                                    handle_low_tariff=self.handle_low_tariff,
                                    asymetric_inverter=self.current_inverter['asymetric'],
                                    selling_enabled=self.selling_enabled,
                                    with_battery=self.with_battery,
                                    battery_loader_efficiency=self.battery_loader_efficiency
                                    )
        return processor

    def initialize_pv_optimizer(self, solar_calculator, processor):
        pv_optimizer = PVOptimizer(
            solar_calculator=solar_calculator,
            processor=processor,
            cost_per_kwh=self.prizing_energy_and_cost['buying_cost_from_distributor_per_kwh'],
            selling_cost_per_kwh=self.prizing_energy_and_cost['selling_cost_to_distributor_per_kwh'],
            battery_configs=self.battery_configs,
            battery_cost=self.current_battery['cost'],
            battery_type=self.current_battery['type'],
            battery_info=self.battery_info,
            panel_configurations_list=self.panel_array_configs,
            panel_degradation=self.panel_degredation_yearly[self.current_panel_params['celltype']],
            one_panel_cost=self.current_panel_params['price'],
            invertor_configs=self.invertor_configs,
            invertor_cost=self.current_inverter['cost'],
            installation_cost=self.prizing_energy_and_cost['installation_cost'],
            annual_expenses=self.prizing_energy_and_cost['annual_expenses'],
            with_battery=self.with_battery,
            current_battery=self.current_battery,
            current_inverter=self.current_inverter,
            off_grid=self.off_grid,
        )
        return pv_optimizer

    def show_first_iteration_stats(self, roi):
        print("ROI in years:", roi)

    def get_configuration(self, individual):
        """for saving the best configuration of panels, batteries and inverters to json file"""
        configuration = {}

        configuration['panels_after_optimization'] = []
        panel_configs = individual[0]  # Get the list of panel configurations
        for config in panel_configs:
            configuration['panels_after_optimization'].append({
                'surface_tilt': config['surface_tilt'],
                'surface_azimuth': config['surface_azimuth'],
                'strings': config['strings'],
                'modules_per_string': config['modules_per_string']
            })

        if self.with_battery:
            configuration['battery_after_optimization'] = individual[-2]
        configuration['inverter_after_optimization'] = individual[-1]

        return configuration

    ## Genetic algorithm methods 
    def print_best_configuration(self, best_configuration, fitness):
        print("Best Configuration of panels:", best_configuration[0])
        index = 1
        if self.with_battery:
            print("Best Configuration of battery:", best_configuration[1])
            index += 1
        print("Best Configuration of inverter:", best_configuration[index])
        print("Fitness (ROI in years):", fitness)

    def save_all_genetic_alg_runs(self, pv_optimizer):
        if self.with_battery:
            if pv_optimizer.battery_configs is not None:
                ddf = pd.DataFrame(pv_optimizer.params_roi_lists, columns=['Panel configuration', 'Battery Size','Battery exchanges done', 'Years of ROI', 'Inverter config', 'Blackout Days'])
                if self.off_grid:
                    ddf.to_csv('output_of_simulations/GA_configurations_with_battery_off_grid.csv', index=False)
                else:
                    ddf.to_csv('output_of_simulations/GA_configurations_with_battery.csv', index=False)
            else:
                ddf = pd.DataFrame(pv_optimizer.params_roi_lists, columns=['Panel configuration','Battery Config', 'Battery exchanges done', 'Years of ROI', 'Inverter config', 'Blackout Days'])
                ddf.to_csv('output_of_simulations/GA_configurations_with_battery.csv', index=False)
        else:
            ddf = pd.DataFrame(pv_optimizer.params_roi_lists, columns=['Panel configuration', 'Inverter config', 'Years of ROI'])
            ddf.to_csv('output_of_simulations/GA_configurations.csv', index=False)
        return ddf

    def set_optimization_precision(self):
        """
        Set the values of ngen and pop_size based on the user's desired simulation precision.
        """
        while True:
            print("\nPlease select the level of genetic algorithm optimization precision:\n")
            print("1. Detailed: 20 generations, 40 population size")
            print("2. Middle: 15 generations, 30 population size")
            print("3. Little: 10 generations, 20 population size")
            print("4. Test (default) : 2 generations, 3 population size")
            precision = input("Enter your choice (1-4): ").lower()

            # Map the user's choice to the corresponding precision
            precision_map = {'1': 'detailed', '2': 'middle', '3': 'little', '4': 'test', '': 'test'}

            if precision in precision_map:
                precision = precision_map[precision]
                if precision == "detailed":
                    ngen = 20
                    pop_size = 40
                elif precision == "middle":
                    ngen = 15
                    pop_size = 30
                elif precision == "little":
                    ngen = 10
                    pop_size = 20
                else:
                    ngen = 1
                    pop_size = 3

                return ngen, pop_size
            else:
                print("Invalid input. Please enter a number between 1 and 4.")

    def set_off_grid_best_configuration(self, pv_optimizer, ga_iterations_df, current_best_configuration):
        if self.off_grid:
            off_grid_df = pv_optimizer.print_best_config_off_grid(ga_iterations_df, self.max_blackout_days)
            return pv_optimizer.set_config_to_off_grid_config(off_grid_df)
        return current_best_configuration

    def display_optimization_results(self, pv_optimizer, new_roi_value, original_roi, best_configuration, ddf, max_blackout_days):
        """
        Display the results of the PV optimization.

        Args:
        off_grid (bool): Whether the simulation is run in off-grid mode.
        pv_optimizer (PVOptimizer): The PV optimizer instance used in the simulation.
        new_roi_value (float): The new ROI value computed.
        original_roi (float): The original ROI value before optimization.
        best_configuration (list): The list containing the best configuration of panels, batteries, and inverter.
        index (int): The index in the best_configuration list where the inverter configuration is found.
        ddf (DataFrame): The DataFrame containing all configurations and their respective ROIs.
        max_blackout_days (int): Maximum number of blackout days allowed in the simulation.
        """
        if self.off_grid and not pv_optimizer.was_new_configuration_better(new_roi_value, original_roi):
            print("\n\n\n")
            pv_optimizer.print_best_config_off_grid(ddf, max_blackout_days)
        
        print("Total DC energy production in kWh: ", pv_optimizer.solar_calculator.total_dc_output.sum()/1000)
        print("Total AC energy production in kWh: ", pv_optimizer.solar_calculator.total_ac_output.sum()/1000)
        print("New ROI:", new_roi_value)
        print("ROI improvement:", original_roi - new_roi_value, "years")
        print("Best Configuration of panels:", best_configuration[0])
        
        if self.with_battery:
            print("Best Configuration of battery:", best_configuration[1])
        
        index = 2 if self.with_battery else 1
        print("Best Configuration of inverter:", best_configuration[index])

    def run(self):
        args = self.get_user_input()
        self.print_initial_status()
        self.validate_inverter()
        self.setup_initial_configuration(args)
        if self.get_new_consumption_data or not os.path.exists('consumption_data/full_year_df.pkl'):
            self.handle_data_extraction()
        full_year_df = self.load_full_year_data()
        if full_year_df is None: 
            print("Failed to load data. Exiting.")
            return
            
        consumption_data_year = self.get_consumption_year_number(full_year_df)
        solar_calculator = self.initialize_solar_energy_calculator(consumption_data_year)

        visualize_solar = input("\nVisualize graphs of solar energy data? (y/n): ").lower() == 'y'
        if visualize_solar:
            self.visualize_solar_energy_data(solar_calculator)
        
        frequency_of_consumption_data = self.get_sampling_frequency(full_year_df)
        new_frequency = self.set_simulation_consumption_data_precision(frequency_of_consumption_data)
        consumption_data_df = self.resample_data_to_new_frequency(full_year_df, new_frequency)
        frequency_of_consumption_data = new_frequency # Update the frequency to the new one for later use, but before resampling method
        
        processor = self.initialize_energy_data_processor(consumption_data_df, solar_calculator)
        if self.with_battery:
            partial_f = partial(processor.processing_records_with_battery, off_grid=self.off_grid, asymetric_inverter=processor.asymetric_inverter, selling_enabled=processor.selling_enabled, max_single_phase_ratio=self.max_single_phase_ratio)
        else:
            partial_f = partial(processor.process_energy_3_phases_on_grid, selling_enabled=processor.selling_enabled, asymetric_inverter=processor.asymetric_inverter, max_single_phase_ratio=self.max_single_phase_ratio)
        partial_f()

        self.print_low_tarif_energy_amounts(processor)
        self.print_simulation_results(processor, solar_calculator)
        self.count_blackout_days(processor)
        self.visualize_energy_processor_data(processor)

        pv_optimizer = self.initialize_pv_optimizer(solar_calculator, processor)
        self.set_days_without_blackout_if_off_grid(self.off_grid, pv_optimizer)

        original_roi = pv_optimizer.compute_original_configuration(partial_f)
        self.show_first_iteration_stats(original_roi)

        continue_with_optimalization = input("Do you want to proceed with optimalization? (y/n): ").lower() == 'y'
        if not continue_with_optimalization:
            print("End of simulation.")
            return

        # Downsampling the data to 30 minutes for optimization
        resampled_consumption_data = consumption_data_df.resample('30min').mean()
        processor.phases_data_frame = resampled_consumption_data

        start_of_optimalization = time.time()

        ngen, pop_size = self.set_optimization_precision()
        
        best_configuration, fitness = pv_optimizer.run_genetic_algorithm(ngen, pop_size)
        self.print_best_configuration(best_configuration, fitness=fitness)
        ga_iterations_df = self.save_all_genetic_alg_runs(pv_optimizer)

        best_configuration = self.set_off_grid_best_configuration(pv_optimizer=pv_optimizer,ga_iterations_df=ga_iterations_df, current_best_configuration=best_configuration)

        # Run the simulation to check if the optimalization was better with original sampling frequency
        consumption_data_df = consumption_data_df.resample(frequency_of_consumption_data).mean()
        processor.phases_data_frame = consumption_data_df
        new_roi = pv_optimizer.run_best_config(best_configuration)
        pv_optimizer.check_new_roi(new_roi, original_roi)

        self.display_optimization_results(pv_optimizer, new_roi, original_roi, best_configuration, ga_iterations_df, self.max_blackout_days)

        end_of_optimalization = time.time()
        print("Time of optimalization: ", end_of_optimalization - start_of_optimalization)
        data_to_save_optimized_config = self.get_configuration(best_configuration)

        # Add the optimized configurations to the data_to_save dictionary
        data_to_save = {
            "new_ROI": new_roi,
            "original_ROI": original_roi,
            "ROI_change": original_roi - new_roi,
            "inverter_after_optimization": data_to_save_optimized_config['inverter_after_optimization'],
            'battery_after_optimization' : data_to_save_optimized_config['battery_after_optimization'] if 'battery_after_optimization' in data_to_save_optimized_config else None,
            "panels_after_optimization": data_to_save_optimized_config['panels_after_optimization'],
            "time_of_optimization": end_of_optimalization - start_of_optimalization,
            "total_DC_energy_production": pv_optimizer.solar_calculator.total_dc_output.sum() / 1000,
            "total_energy_production": pv_optimizer.solar_calculator.total_ac_output.sum() / 1000,
            "consumed_energy_from_grid_without_PV": pv_optimizer.consumed_energy_without_PV / 1000,
            "consumed_energy_from_grid_with_PV": pv_optimizer.consumed_energy_with_PV / 1000,
            "energy_sold_to_grid": pv_optimizer.energy_sold_to_grid / 1000,
            "blackout_days": pv_optimizer.blackout_days if self.off_grid else "N/A",
        }

        self.param_setter.save_configuration(data_to_save, 'output_of_simulations/optimalization_results.json')
        print("Optimalization results saved to output_of_simulations/optimalization_results.json\n")
        print("End of optimalization and simulation.")

if __name__ == "__main__":
    sim = Simulation()
    sim.run()
