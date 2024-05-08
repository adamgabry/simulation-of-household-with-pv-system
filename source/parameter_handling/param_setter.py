import json
import os
import pandas as pd

class ParamSetter:
    """
    A class that provides methods for setting default values and loading JSON configurations.
    """
    def __init__(self):
        pass

    def set_default_inverter(self, invertor_configs):
        """
        Sets the default inverter configuration from a list of inverter configurations.

        Args:
            invertor_configs (list): A list of inverter configurations.

        Returns:
            dict: The default inverter configuration.
        """
        for invertor in invertor_configs:
            if invertor['default']:
                return invertor
            
    def set_default_battery(self, battery_configs):
        """
        Sets the default battery configuration from a list of battery configurations.

        Args:
            battery_configs (list): A list of battery configurations.

        Returns:
            dict: The default battery configuration.
        """
        for battery in battery_configs:
            if battery['default']:
                return battery
            
    def load_json_config(self, file_path):
        """
        Loads a JSON configuration from a file.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            dict: The loaded JSON configuration.
        """
        with open(file_path, 'r') as file:
            config = json.load(file)
        return config
    
    def convert_periods_to_tuples(self,periods_dict):
        """
        Convert a dictionary of periods to a list of tuples.

        Args:
            periods_dict (dict): A dictionary containing periods with 'start' and 'end' keys.

        Returns:
            list: A list of tuples, where each tuple contains the start and end values of a period.
        """
        return [(period['start'], period['end']) for period in periods_dict]
    
    def load_multiple_json_configs(self, file_paths):
        configs = {}
        for file_path in file_paths:
            file_name = os.path.basename(file_path).split('.')[0]  # get file name without extension
            configs[file_name] = self.load_json_config(file_path)
        return configs

    def save_configuration(self, configuration, filepath="output_of_simulations/initial_config.json"):
        """ Save the initial configuration to a JSON file before running the genetic algorithm. """
        with open(filepath, "w") as file:
            json.dump(configuration, file, indent=4)

    def save_results_to_csv_nicer(self, data, filename='output_of_simulations/simulation_results_nicer.csv'):
        # Modify the dictionary entries to be JSON strings
        for key, value in data.items():
            if isinstance(value, dict):
                data[key] = json.dumps(value, indent=4) 
            elif isinstance(value, list):
                # Assume list of dicts
                data[key] = "[\n" + ",\n".join(json.dumps(v, indent=4) for v in value) + "\n]"
        
        # Convert dictionary to DataFrame
        df = pd.DataFrame([data])
        df.to_csv(filename, index=False)
        print(f"Data successfully saved to {filename}")