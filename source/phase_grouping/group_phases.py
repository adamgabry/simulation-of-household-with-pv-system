"""
@Author: Adam Gabrys
@Date: 2024-02-15
Description: This script contains a class PhaseGrouper, which is used to group and process phase data from a CSV file to a dataframe. 
The class takes sensor IDs for each phase and a file path during initialization. It has several methods to handle unavailable 
values, get frequency of data, group phase data based on a time interval, process grouped data into a DataFrame, resample and 
interpolate the DataFrame, and plot the phase states over time.
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone

class PhaseGrouper:
    def __init__(self, sensor_id_first_phase, sensor_id_second_phase, sensor_id_third_phase, file_path=None):
        """
        Initialize the GroupPhases object.

        Args:
            sensor_id_first_phase (int): The sensor ID for the first phase.
            sensor_id_second_phase (int): The sensor ID for the second phase.
            sensor_id_third_phase (int): The sensor ID for the third phase.
            file_path (str, optional): The file path. Defaults to None.
        """
        self.file_path = file_path
        self.sensor_id_first_phase = sensor_id_first_phase
        self.sensor_id_second_phase = sensor_id_second_phase
        self.sensor_id_third_phase = sensor_id_third_phase
        
    @staticmethod
    def handle_unavailable(val):
        return 0 if val == 'unavailable' else val
    
    def get_freq(self, df):
        """
        Calculate the most common time interval between timestamps in a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the timestamps.

        Returns:
            str: The most common time interval in seconds, represented as a string to pass to the resample method.
        Raises:
            ValueError: If the index of the DataFrame is not a DatetimeIndex.
        """
        df = df.head(1000)

        # check if the index is in datetime format
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be a DatetimeIndex")

        # Transpose the DataFrame so that sensors are rows
        df_transposed = df.transpose()

        # Calculate the differences within each sensor across timestamps
        time_diffs = df.index.to_series().diff().dt.total_seconds().dropna()

        # Find the most common time difference
        if time_diffs.empty:
            print("No time differences found or insufficient data.")
            return None
        else:
            overall_common_interval = time_diffs.mode()[0]
            print(f"Most common interval across sensors (in seconds): {overall_common_interval}")
            freq = f"{int(overall_common_interval)}s"
        return freq
    
    def handle_timestamp(self, timestamp):
        """
        Handles the given timestamp by converting it to a datetime object.

        Args:
            timestamp (float or str): The timestamp to be handled. It can be either a float representing a Unix timestamp
                or a string in the format 'YYYY-MM-DD HH:MM:SS'.

        Returns:
            datetime: The converted datetime object.

        Raises:
            ValueError: If the timestamp is not in a valid format.

        """
        if isinstance(timestamp, float):
            # If timestamp is a float, convert it from Unix timestamp to datetime
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        else:
            # If timestamp is a string, parse it into a datetime object
            return datetime.strptime(timestamp.split('.')[0], '%Y-%m-%d %H:%M:%S')

    def group_phases(self, time_interval=3):
        """
        Groups the data based on the time interval and returns the state of each phase at a given time.

        Parameters:
        - time_interval (int): The time interval in seconds used to group the data. Default is 3 seconds.

        Returns:
        - grouped_data (dict): A dictionary where the keys are formatted timestamps and the values are dictionaries containing the state of each phase at that time.
        """
        # Time interval is set to 3 seconds by default to group the data that are close to each other, if needed it can be adjusted
        # We want to group the data based on the time_interval, so that we can get the state of each phase at a given time, larger time_interval means bigger difference between the capturing of states and sure missleading data
        df = pd.read_csv(self.file_path, header=None, 
                 names=['id', 'current_phase_id', 'state_mA', 'UTC_timestamp'],
                 dtype={'id': int, 'current_phase_id': int },
                 converters={'state_mA': self.handle_unavailable})
        df.sort_values(by='UTC_timestamp', inplace=True)

        grouped_data = {}
        start_time = None
        current_group = {}

        for _, row in df.iterrows():
            timestamp = self.handle_timestamp(row['UTC_timestamp'])
            if start_time is None or (timestamp - start_time).total_seconds() > time_interval:
                if start_time is not None:
                    formatted_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
                    grouped_data[formatted_time] = current_group
                start_time = timestamp
                current_group = {}
            current_group[row['current_phase_id']] = row['state_mA']

        # Adding last group
        formatted_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
        grouped_data[formatted_time] = current_group

        return grouped_data
    
    def process_grouped_data(self, grouped_data, phases_in_Watts=False, current_voltage=235):
        """
        Process the grouped data and convert it into a DataFrame.

        Args:
            grouped_data (dict): A dictionary containing the grouped data, where the keys are timestamps and the values are dictionaries of phase data.
            phases_in_Watts (bool, optional): Indicates whether the phase values are in Watts. Defaults to False.
            current_voltage (int, optional): The current voltage in Volts. Defaults to 235.

        Returns:
            pandas.DataFrame: The processed data as a DataFrame, with timestamps as the index and phase values in separate columns.
        """
        timestamps = []
        phase_1 = []
        phase_2 = []
        phase_3 = []

        for timestamp, phases in grouped_data.items():
            timestamps.append(datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S"))
            # Convert 'unavailable' to 0, ensure it's numeric
            phase_1_val = 0 if phases.get(self.sensor_id_first_phase, 'unavailable') == 'unavailable' else phases.get(self.sensor_id_first_phase, 0)
            phase_2_val = 0 if phases.get(self.sensor_id_second_phase, 'unavailable') == 'unavailable' else phases.get(self.sensor_id_second_phase, 0)
            phase_3_val = 0 if phases.get(self.sensor_id_third_phase, 'unavailable') == 'unavailable' else phases.get(self.sensor_id_third_phase, 0)

            # If phases are in Watts, just append the values
            if phases_in_Watts:
                phase_1.append(pd.to_numeric(phase_1_val, errors='coerce'))
                phase_2.append(pd.to_numeric(phase_2_val, errors='coerce'))
                phase_3.append(pd.to_numeric(phase_3_val, errors='coerce'))
            else:
                # Convert mA to Watts - so multiply by current_voltage ~235(mW) and divide by 1000 (W)
                mA_to_Watts_multiplier_with_Voltage = current_voltage / 1000

                phase_1.append(pd.to_numeric(phase_1_val, errors='coerce') * mA_to_Watts_multiplier_with_Voltage)
                phase_2.append(pd.to_numeric(phase_2_val, errors='coerce') * mA_to_Watts_multiplier_with_Voltage)
                phase_3.append(pd.to_numeric(phase_3_val, errors='coerce') * mA_to_Watts_multiplier_with_Voltage)

        df = pd.DataFrame({
            'timestamp': timestamps,
            'phase_1': phase_1,
            'phase_2': phase_2,
            'phase_3': phase_3
        }).set_index('timestamp')

        # Fill NaN values resulted from conversion with 0
        # Shouldnt be necessary, but to ensure the data is clean and ready to use further
        df.fillna(0, inplace=True)

        return df
    
    def resample_and_interpolate(self, freq=None):
        """
        Resamples and interpolates the grouped data.

        Args:
            freq (str, optional): The frequency to resample the data to. Defaults to None.

        Returns:
            pandas.DataFrame: The resampled and interpolated data.
        """
        grouped_data = self.group_phases()
        df = self.process_grouped_data(grouped_data)
        if freq is None:
            freq = self.get_freq(df)
        df = df.resample(freq).mean()
        df.interpolate(method='linear', inplace=True)

        return df

    def plot_phases(self, df):
        """
        Plots the phase states over time.

        Parameters:
        - df (DataFrame): The DataFrame containing the phase states.

        Returns:
        - None
        """
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['phase_1'], label='Phase 1')
        plt.plot(df.index, df['phase_2'], label='Phase 2', linestyle='--')
        plt.plot(df.index, df['phase_3'], label='Phase 3', linestyle='-.')
        plt.xlabel('Time')
        plt.ylabel('State (Watts)')
        plt.title('Phase States Over Time')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
