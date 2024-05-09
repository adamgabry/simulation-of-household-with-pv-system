'''
@author: Adam Gabrys
@date: 2023-12-20
@description: This module contains the EnergyDataProcessor class, which processes energy data from consumption and generation sources, and simulates energy storage and usage.
'''

import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, time

class EnergyDataProcessor:
    def __init__(self, battery_capacity,
            battery_type,
            battery_info,
            buying_cost_per_kWh, selling_cost_per_kWh, 
            inverter_efficiency, invertor_threshold,
            with_battery,
            handle_low_tariff=False,
            asymetric_inverter=False,
            selling_enabled=False,
            consumption_file=None,
            battery_loader_efficiency=None,
            dc_generation_file = None,
            ac_generation_file = None,
            phases_data_frame=None,
            dc_generation_data_frame = None,
            ac_generation_data_frame = None
            ):
        """
        This class processes energy data from consumption and generation sources, and simulates energy storage(if with accumulative items) and usage.
        
        Parameters:
        - battery_capacity (float): The capacity of the battery in kWh.
        - battery_type (str): The type of the battery.
        - consumption_file (str): The file path of the consumption data.
        - battery_info (dict): The properties of the battery.
        - buying_cost_per_kWh (float): The cost of buying electricity per kWh.
        - selling_cost_per_kWh (float): The cost of selling electricity per kWh.
        - inverter_efficiency (float): The efficiency of the inverter.
        - invertor_threshold (float): The nominal threshold for inverter operation.
        - dc_generation_file (str, optional): The file path of the DC generation data. Defaults to None.
        - ac_generation_file (str, optional): The file path of the AC generation data. Defaults to None.
        - phases_data_frame (DataFrame): The data frame containing phase information. Defaults to None.
        - dc_generation_data_frame (DataFrame): The data frame containing DC generation data. Defaults to None.
        - ac_generation_data_frame (DataFrame): The data frame containing AC generation data. Defaults to None.
        """
        
        self.with_battery = with_battery
        self.battery_capacity = battery_capacity
        self.battery_type = battery_type
        self.battery_properties = battery_info
        self.consumption_file = consumption_file
        self.inv_eff = inverter_efficiency
        self.invertor_threshold = invertor_threshold
        self.dc_generation_file = dc_generation_file
        self.ac_generation_file = ac_generation_file
        self.phases_data_frame = phases_data_frame
        self.cost_per_kWh = buying_cost_per_kWh
        self.selling_cost_per_kWh = selling_cost_per_kWh
        self.dc_generation_df = dc_generation_data_frame
        self.ac_generation_df = ac_generation_data_frame
        self.handle_low_tariff = handle_low_tariff
        self.asymetric_inverter = asymetric_inverter
        self.selling_enabled = selling_enabled
        self.max_single_phase_ratio = 0.333

        # If the battery loader efficiency is not provided, assume it's the same as the inverter efficiency - this is a common scenario
        if battery_loader_efficiency is None:
            self.battery_loader_efficiency = inverter_efficiency
        else:
            self.battery_loader_efficiency = battery_loader_efficiency    
        
        self.battery_current_storage = 0.7 * self.battery_capacity # Start with 70% of the battery capacity
        self.total_consumed_energy_with_PV = 0
        self.energy_sold = 0
        self.total_consumed_energy_without_PV = 0
        self.wasted_excess_energy = 0
        self.wasted_excess_energy_phase = [[0], [0], [0]]
        self.phases_count = 3
        
        self.consumed_energy_list = []
        self.battery_storage_list = []
        self.momentary_energy_used_from_grid_list = []
        self.momentary_energy_consumption_without_PV = []
        self.total_energy_used_from_grid_list = []
        self.wasted_excess_energy_list = []
        self.wasted_excess_energy_points_list = []
        self.energy_data_df = None
        self.blackout_data_df = None
        self.data_to_plot = None

    def reset_data(self):
        self.battery_current_storage = 0
        self.total_consumed_energy_with_PV = 0
        self.energy_sold = 0
        self.total_consumed_energy_without_PV = 0
        self.wasted_excess_energy = 0
        self.wasted_excess_energy_phase = [[0], [0], [0]]
        self.bad_data_count = 0
        
        self.consumed_energy_list = []
        self.battery_storage_list = []
        self.momentary_energy_consumption_without_PV = []
        self.momentary_energy_used_from_grid_list = []
        self.total_energy_used_from_grid_list = []
        self.wasted_excess_energy_list = []
        self.wasted_excess_energy_points_list = []
        self.energy_data_df = None
        self.blackout_data_df = None
        self.data_to_plot = None

    def set_battery_operational_range(self):
        """
        Sets the operational range of the battery based on the battery type.

        If the battery type is recognized, the method sets the minimum and maximum capacity coefficients
        and returns the calculated minimum and maximum operational range of the battery.
        If the battery type is not recognized, it sets the operational range to default values and prints a warning message.

        Returns:
            Tuple: A tuple containing the minimum and maximum operational range of the battery.
        """
        # Check if the battery type provided is in the battery_properties dictionary
        if self.battery_type in self.battery_properties:
            # If so, set the min and max capacity coefficients
            self.min_capacity_coef = self.battery_properties[self.battery_type]['min_capacity']
            self.max_capacity_coef = self.battery_properties[self.battery_type]['max_capacity']
            return self.min_capacity_coef * self.battery_capacity, self.battery_capacity * self.max_capacity_coef
        else:
            print(f'Battery type {self.battery_type} not recognized. Setting to default. Check the battery_info.py file.')
            return 0.2 * self.battery_capacity, self.battery_capacity
    
    def get_processed_simulation_data(self):
            """
            Returns the processed simulation data .

            Returns:
                tuple: A tuple in this order containing the total 
                        - consumed energy with PV, 
                        - total consumed energy without PV,
                        - energy sold
                        - wasted excess energy.
            """
            return self.total_consumed_energy_with_PV, self.total_consumed_energy_without_PV, self.energy_sold, self.wasted_excess_energy

    def statistics_show(self, with_battery=True, off_grid = False):	                
        """
        Display statistics related to energy consumption and production.

        prints:
        - Total consumed energy without PV
        - Total consumed energy with PV
        - Total energy sold
        - Total wasted excess energy
        - Total cost of consumed energy without PV
        - Total cost of consumed energy with PV
        - Total cost of sold energy
        - Total saved money

        Shows plots for:
        - Energy Used from Grid vs Consumed Energy
        - Battery Storage vs Energy Used from Grid
        - Wasted Excess Energy per Phase (accumulated)
        
        Parameters:
        - with_battery (bool): Whether to include battery storage data in the plot. Default is True.
        """
        print("Values are set for the given computed time-array:")
        print(f"Total consumption from grid with use of the PV system: {self.total_consumed_energy_without_PV/1000} kWh")
        print(f"Energy sold yearly: {self.energy_sold/1000} kWh")
        print(f"Total consumption from grid with use of the PV system: {self.total_consumed_energy_with_PV/1000} kWh")
        print("Money saved: ", ((self.total_consumed_energy_without_PV - self.total_consumed_energy_with_PV)/1000 * self.cost_per_kWh + (self.energy_sold/1000 * self.selling_cost_per_kWh)))

        # Plot the data
        plt.figure(figsize=(16, 12))

        # Subplot 1: Energy Used from Grid vs Consumed Energy
        ax1 = plt.subplot(3, 1, 1)
        timestamps, energy_values = zip(*self.total_energy_used_from_grid_list)
        energy_values = [value / 1000 for value in energy_values]
        if off_grid:
            ax1.plot(timestamps, energy_values, label='Grid Energy Equivalent Required to Support the PV System in Off-grid Mode')
        else:
            ax1.plot(timestamps, energy_values, label='Energy used from grid with the use of PV system')
        ax1.set_ylabel('Energy (kWh)') 
        if off_grid:
            ax1.set_title('Energy consumption not covered from PV at Off-Grid system', fontsize=16)
        else:
            ax1.set_title('Energy Used from grid with PV system', fontsize=16) 
        ax1.legend(loc='upper right')  

        # Subplot 2: Battery Storage
        ax2 = plt.subplot(3, 1, 2)
        if with_battery:
            try:
                timestamps, battery_values = zip(*self.battery_storage_list)
                ax2.plot(timestamps, battery_values, label='Battery state of charge')
                ax2.legend(loc='upper right')
            except ValueError:
                print('No battery data to plot.')
        ax2.set_ylabel('Energy (Wh)')
        ax2.set_title('')

        # Subplot 3: Wasted Excess Energy per Phase (accumulated)
        plt.subplot(3, 1, 3)
        plt.plot(range(len(self.wasted_excess_energy_phase[0])), self.wasted_excess_energy_phase[0], label='Phase 1')
        plt.plot(range(len(self.wasted_excess_energy_phase[1])), self.wasted_excess_energy_phase[1], label='Phase 2')
        plt.plot(range(len(self.wasted_excess_energy_phase[2])), self.wasted_excess_energy_phase[2], label='Phase 3')
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Wasted generated energy (kWh)')
        plt.title('Accumulated Wasted Excess Energy per Phase, that was already cut by inverter inverter threshold', fontsize=16)
        plt.legend(loc='upper right')

        # Adjust the spacing between subplots
        plt.subplots_adjust(hspace=0.5)

        plt.show()

    def plot_energy_delta(self):
        """
        Plots the power delta for each phase over time.

        Returns:
            None
        """
        linestyles = ['-', '--', '-.']

        # Plot power delta for each phase
        fig, ax2 = plt.subplots(figsize=(10, 6))
        for phase in range(1, self.phases_count + 1):
            ax2.plot(self.energy_data_df['timestamp'], self.energy_data_df[f'power_delta_phase_{phase}'],
                    label=f'Phase {phase} Delta', linestyle=linestyles[(phase - 1) % len(linestyles)])
        ax2.set_title('Power Delta per Phase')
        ax2.set_ylabel('Power (W)')
        ax2.legend(loc='upper right')

    def demo_plot_battery_state(self):
        """
        Plot the battery storage status over time for the entire year.

        This function requires that the battery data has been accumulated.

        The plot includes:
        - Battery Utilization: Shows the battery storage status over time.

        Note: If there is no battery data to plot, a corresponding message will be printed.

        Returns:
        None
        """
        if not self.with_battery:
            return
        
        plt.figure(figsize=(8, 4))

        # Increase the font size
        plt.rcParams.update({'font.size': 14})

        # Plot battery storage status if applicable
        if self.with_battery:
            try:
                timestamps, battery_values = zip(*self.battery_storage_list)  # This unpacks the list of tuples

                # Convert to pandas Series for easy filtering
                battery_series = pd.Series(battery_values, index=timestamps)

                plt.plot(battery_series.index, battery_series.values, label='Battery Status')
                plt.legend(loc='upper right')
            except ValueError:
                print('No battery data available for plotting.')
            plt.xlabel('Time')
            plt.ylabel('Energy (Wh)')
            plt.title('Battery Charge Status')

            # Format the x-axis to only show the month and day
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

            plt.tight_layout()
        else:
            print("No battery data available.")
        plt.show()
        
    def plot_generated_power_per_phase_allocation(self):
        linestyles = ['-', '--', '-.']  # different line styles for different phases

        # Create a single plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot generated power for each phase
        for phase in range(1, self.phases_count + 1):
            ax.plot(self.energy_data_df['timestamp'], self.energy_data_df[f'power_generated_phase_{phase}'],
                    label=f'Allocated power for phase_{phase}', linestyle=linestyles[(phase - 1) % len(linestyles)])
        if self.asymetric_inverter:
            ax.set_title('Generated power allocated to individual phases with ASYMMETRIC inverter', fontsize=16)
        else:
            ax.set_title('Generated power allocated to individual phases with SYMMETRIC inverter', fontsize=16)
        ax.set_ylabel('Power (W)', fontsize=14)
        ax.set_xlabel('Time', fontsize=14)
        ax.legend(loc='upper right')

        plt.show()

    def plot_phases_data(self):
            """
            Plot the power needs, power delta, generated power, battery storage status, and wasted excess energy points
            for each phase of the power plant.

            This function requires that the energy data has been accumulated in one of proccessing functions.

            The plots include:
            - Power Need per Phase: Shows the power needs for each phase over time.
            - Power Delta per Phase: Shows the change in power for each phase over time.
            - Generated Power per Phase: Shows the power generated for each phase over time.
            - Battery Utilization: Shows the battery storage status over time (if applicable).
            - Wasted Excess Energy Points: Shows the wasted excess energy points over time.

            Note: If there is no battery data or wasted excess energy points data to plot, a corresponding message will be printed.

            Returns:
            None
            """
            plt.figure(figsize=(14, 10))

            num_of_plots = 3
            if self.with_battery:
                num_of_plots = 4

            linestyles = ['-', '--', '-.']  # different line styles for different phases

            # Plot power needs for each phase
            ax1 = plt.subplot(num_of_plots, 1, 1)
            for phase in range(1, self.phases_count + 1):
                ax1.plot(self.energy_data_df['timestamp'], self.energy_data_df[f'power_need_phase_{phase}'],
                        label=f'Power consumption phase {phase}', linestyle=linestyles[(phase - 1) % len(linestyles)])
            ax1.set_title('Power need for each phase', fontsize=14)
            ax1.set_ylabel('Power (W)', fontsize=12)
            ax1.legend(loc='upper right')

            # Plot power delta for each phase
            ax2 = plt.subplot(num_of_plots, 1, 2)
            for phase in range(1, self.phases_count + 1):
                ax2.plot(self.energy_data_df['timestamp'], self.energy_data_df[f'power_delta_phase_{phase}'],
                        label=f'Difference on phase {phase} ', linestyle=linestyles[(phase - 1) % len(linestyles)])
            ax2.set_title('Difference between generated and consumed power per phase', fontsize=14)
            ax2.set_ylabel('Power (W)', fontsize=12)
            ax2.legend(loc='upper right')

            # Plot generated power for each phase
            ax3 = plt.subplot(num_of_plots, 1, 3)
            for phase in range(1, self.phases_count + 1):
                ax3.plot(self.energy_data_df['timestamp'], self.energy_data_df[f'power_generated_phase_{phase}'],
                        label=f'Generated power delivered to phase {phase}', linestyle=linestyles[(phase - 1) % len(linestyles)])
            ax3.set_title('Generated power distributed to each phase with ASYMMETRIC inverter', fontsize=14)
            ax3.set_ylabel('Power (W)', fontsize=12)   
            ax3.legend(loc='upper right')

            # Plot battery storage status if applicable
            if self.with_battery:
                ax4 = plt.subplot(4, 1, 4)
                try:
                    timestamps, battery_values = zip(*self.battery_storage_list)  # This unpacks the list of tuples
                    ax4.plot(timestamps, battery_values, label='Battery charge status')
                    ax4.legend(loc='upper right')
                except ValueError:
                    print('No battery data to plot.')
                ax4.set_xlabel('Time')
                ax4.set_ylabel('DC Energy (Wh)', fontsize=12)
                ax4.set_title('Battery charge status', fontsize=14)

            # Format the x-axis to only show the month and day
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            if self.with_battery:
                ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.tight_layout()
            plt.show()

    def plot_power_needs(self):
        """
        Plot the power needs for each phase and the total power need comparison.

        This method creates a figure with three subplots, each showing the power needs for a specific phase
        (Phase 1, Phase 2, and Phase 3) as well as the total power need. The power needs are plotted against
        the timestamp.

        Returns:
            None
        """
        # Set the overall figure size
        plt.figure(figsize=(10, 18))

        # Subplot 1: Phase 1 Need + Total Power Need
        plt.subplot(3, 1, 1)  # 3 rows, 1 column, subplot 1
        plt.plot(self.energy_data_df['timestamp'], self.energy_data_df['power_need_all_phases'], label='Total Power Need', linestyle='-.')
        plt.plot(self.energy_data_df['timestamp'], self.energy_data_df['power_need_phase_1'], label='Phase 1 Need')
        plt.title('Phase 1 Power Need vs. Total')
        plt.ylabel('Power (W)')
        plt.legend(loc='upper right')

        # Subplot 2: Phase 2 Need + Total Power Need
        plt.subplot(3, 1, 2)  # 3 rows, 1 column, subplot 2
        plt.plot(self.energy_data_df['timestamp'], self.energy_data_df['power_need_all_phases'], label='Total Power Need', linestyle='-.')
        plt.plot(self.energy_data_df['timestamp'], self.energy_data_df['power_need_phase_2'], label='Phase 2 Need', linestyle='--')
        plt.title('Phase 2 Power Need vs. Total')
        plt.ylabel('Power (W)')
        plt.legend(loc='upper right')

        # Subplot 3: Phase 3 Need + Total Power Need
        plt.subplot(3, 1, 3)  # 3 rows, 1 column, subplot 3
        plt.plot(self.energy_data_df['timestamp'], self.energy_data_df['power_need_all_phases'], label='Total Power Need', linestyle='-.')
        plt.plot(self.energy_data_df['timestamp'], self.energy_data_df['power_need_phase_3'], label='Phase 3 Need', linestyle='-.')
        plt.title('Phase 3 Power Need vs. Total')
        plt.ylabel('Power (W)')
        plt.legend(loc='upper right')

        plt.tight_layout()  # Adjust the layout to make sure everything fits without overlap
        plt.show()

    def plot_monthly_and_hourly_consumption(self, hours_only=False, months_only=False, czech=False):
        """
        Plots the monthly and hourly energy consumption with and without PV.

        Parameters:
        - hours_only (bool): If True, only plots the hourly energy consumption. Default is False.
        - months_only (bool): If True, only plots the monthly energy consumption. Default is False.
        """
        data = {
            'timestamp': [x[0] for x in self.momentary_energy_used_from_grid_list],  # Extract timestamps
            'consumed_with_PV': [x[1] for x in self.momentary_energy_used_from_grid_list],
            'consumed_without_PV': [x[1] for x in self.momentary_energy_consumption_without_PV]
        }

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp']) 

        df.set_index('timestamp', inplace=True)

        if not hours_only and not months_only:
            number_of_plots = 2
        else:
            number_of_plots = 1

        plt.figure(figsize=(12, 6))
        if not hours_only:
            monthly_data = df.resample('ME').sum()/1000 # Convert to kWh
            plt.subplot(number_of_plots, 1, 1)
            if not czech:
                plt.plot(monthly_data.index, monthly_data['consumed_with_PV'], label='Energy Consumed with PV', marker='o')
                plt.plot(monthly_data.index, monthly_data['consumed_without_PV'], label='Energy Consumed without PV', marker='o')
                plt.title('Monthly Energy Consumption', fontsize=16)
                plt.xlabel('Month')
                plt.ylabel('Energy (kWh)', fontsize=14)
                plt.tick_params(axis='both', which='major', labelsize=12)
            else:
                plt.plot(monthly_data.index, monthly_data['consumed_with_PV'], label='Spotřeba z veřejné sítě s FVE', marker='o')
                plt.plot(monthly_data.index, monthly_data['consumed_without_PV'], label='Spotřeba z veřejné sítě bez FVE', marker='o')
                plt.title('Měsíční spotřeba energie z veřejné sítě', fontsize=16)
                plt.xlabel('Měsíc', fontsize=14)
                plt.ylabel('Energie (kWh)', fontsize=14)
                plt.tick_params(axis='both', which='major', labelsize=12)
            plt.legend(loc='upper left')

        if not months_only:
            hourly_sum = df.groupby(df.index.hour).sum()/1000 # Convert to kWh
            plt.subplot(number_of_plots, 1, number_of_plots)
            if not czech:
                plt.plot(hourly_sum.index, hourly_sum['consumed_with_PV'], label='Energy Consumed with PV system', marker='o')
                plt.plot(hourly_sum.index, hourly_sum['consumed_without_PV'], label='Energy Consumed without PV system', marker='o')
                plt.title('Yearly energy consumption in hours of the day')
                plt.xlabel('Time')
                plt.ylabel('Energy (kWh)')
                plt.tick_params(axis='both', which='major', labelsize=12)
            else:
                plt.plot(hourly_sum.index, hourly_sum['consumed_with_PV'], label='Spotřeba z veřejné sítě s FVE', marker='o')
                plt.plot(hourly_sum.index, hourly_sum['consumed_without_PV'], label='Spotřeba z veřejné sítě bez FVE', marker='o')
                plt.title('Roční spotřeba energie z veřejné sítě rozdělena podle hodin dne', fontsize=16)
                plt.xlabel('Hodiny dne', fontsize=14)
                plt.ylabel('Energie (kWh)', fontsize=14)
                plt.tick_params(axis='both', which='major', labelsize=12)
            plt.legend(loc='upper left')
        plt.show()

    def validate_and_adjust_tariff_periods(self, periods):
        """
        Validates and adjusts the tariff periods.

        Args:
            periods (list): A list of tuples representing the start and end times of tariff periods.

        Returns:
            list: A list of tuples representing the validated and concatenated tariff periods.

        Raises:
            ValueError: If the start and end times of a tariff period are the same.
            ValueError: If there are overlapping tariff periods.
        """
        # Convert string times to datetime.time objects
        periods = [(datetime.strptime(start, '%H:%M').time(), datetime.strptime(end, '%H:%M').time()) for start, end in periods]

        adjusted_periods = []
        for start, end in periods:
            if start == end:
                raise ValueError("Tariff period start and end times cannot be the same.")

            if end == time(0, 0):
                end = time(23, 59)  # Represent end as the end of the day
            if start > end:  # Spans midnight
                # Add two periods: one until the end of the day and another from the start of the next day
                adjusted_periods.append((start, time(23, 59)))
                adjusted_periods.append((time(0, 0), end))
            else:
                adjusted_periods.append((start, end))

        # Sort periods by start time
        adjusted_periods.sort(key=lambda x: x[0])

        # Check for overlapping periods
        for i in range(1, len(adjusted_periods)):
            if adjusted_periods[i][0] < adjusted_periods[i-1][1]:
                overlap1 = f"{adjusted_periods[i-1][0].strftime('%H:%M')} to {adjusted_periods[i-1][1].strftime('%H:%M')}"
                overlap2 = f"{adjusted_periods[i][0].strftime('%H:%M')} to {adjusted_periods[i][1].strftime('%H:%M')}"
                raise ValueError(f"Tariff periods cannot overlap: {overlap1} overlaps with {overlap2}")

        # Concatenate consecutive periods
        concatenated_periods = []
        last_start, last_end = adjusted_periods[0]
        for current_start, current_end in adjusted_periods[1:]:
            if current_start == last_end:
                # Extend the current period to the last period's end
                last_end = current_end
            else:
                concatenated_periods.append((last_start, last_end))
                last_start, last_end = current_start, current_end
        concatenated_periods.append((last_start, last_end))  # Add the last period

        return concatenated_periods  # Validated and concatenated periods

    def calculate_tariff_consumption(self, low_tariff_periods):
        """
        Calculates the energy consumption grouped by tariff type.

        Args:
            low_tariff_periods (list): A list of tuples representing the start and end times of low tariff periods.

        Returns:
            pandas.Series: A Series object containing the sum of energy consumption for each tariff type.
        """
        # Validate the tariff periods
        low_tariff_periods = self.validate_and_adjust_tariff_periods(low_tariff_periods)
        
        # Convert list of tuples to DataFrame
        data = {
            'timestamp': [x[0] for x in self.momentary_energy_used_from_grid_list],
            'consumed_with_PV': [x[1] for x in self.momentary_energy_used_from_grid_list]
        }
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Assign 'high' tariff by default
        df['tariff'] = 'high'

        # Update the tariff based on the low tariff periods provided
        for start, end in low_tariff_periods:
            # Mask for low tariff time range
            mask = (df.index.time >= start) & (df.index.time <= end)
            df.loc[mask, 'tariff'] = 'low'

        # Sum energy consumption by tariff type
        tariff_consumption = df.groupby('tariff')['consumed_with_PV'].sum()

        return tariff_consumption

    def plot_monthly_consumption(self):
        """
        Plots the monthly energy consumption with and without PV.

        This method retrieves the energy consumption data with and without PV from the object's attributes,
        calculates the monthly difference, and plots the monthly energy consumption from grid with and without using PV.

        Returns:
            None
        """
        data = {
            'timestamp': [x[0] for x in self.total_energy_used_from_grid_list],  # Extract timestamps
            'consumed_with_PV': [x[1] for x in self.total_energy_used_from_grid_list],
            'consumed_without_PV': [x[1] for x in self.consumed_energy_list]
        }

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp']) 

        # Set timestamp as the index
        df.set_index('timestamp', inplace=True)

        hour_five_data = df.between_time('05:00', '05:59')
        print(hour_five_data)

        print('Total energy consumed with PV between 5:00 and 5:59:')
        print(hour_five_data['consumed_with_PV'].head(5).sum())
        print('-----------------------------------')

        # Group by month and sum up the energy values
        monthly_first = df.resample('ME').first()
        monthly_last = df.resample('ME').last()

        # Calculate the difference
        monthly_data = monthly_last - monthly_first

        print(hour_five_data)

        plt.figure(figsize=(12, 6))
        plt.plot(monthly_data.index, monthly_data['consumed_with_PV'], label='Energy Consumed with PV', marker='o')
        plt.plot(monthly_data.index, monthly_data['consumed_without_PV'], label='Energy Consumed without PV', marker='o')

        plt.title('Monthly Energy Consumption')
        plt.xlabel('Month')
        plt.ylabel('Energy (Wh)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def log_blackout(self, blackout_data, start_time, end_time):
        """
        Logs a blackout event with its start time, end time, and duration.

        Args:
            blackout_data (dict): A dictionary containing the blackout data.
            start_time (datetime): The start time of the blackout event.
            end_time (datetime): The end time of the blackout event.
        """
        duration = end_time - start_time
        blackout_data['timestamp_start'].append(start_time)
        blackout_data['timestamp_end'].append(end_time)
        blackout_data['duration'].append(duration)

    def get_generation_df(self):
        """
        Retrieves the generation data as a DataFrame from csv.

        Returns:
            pandas.DataFrame: The generation data as a DataFrame.
        """
        if(self.dc_generation_file is not None):
                pd.read_csv(self.dc_generation_file, index_col=0, parse_dates=True)
        else:
            pass
    
    def limit_power_to_invertor_threshold(self, current_generated_power, with_battery=False):
        """
        Limit the generated power to the inverter's threshold and track any wasted power.

        Args:
            current_generated_power (float): The current power generated before limiting.

        Returns:
            tuple: A tuple containing the limited power and a flag indicating if power conversion is not possible.
        """
        cant_convert_from_battery = False
        if current_generated_power >= self.invertor_threshold:
            self.wasted_excess_energy += (current_generated_power - self.invertor_threshold)
            current_generated_power = self.invertor_threshold
            cant_convert_from_battery = True
        if with_battery:
            return current_generated_power, cant_convert_from_battery
        return current_generated_power

    def allocate_power_asymmetric_inverter(self, row, current_generated_power, max_single_phase_ratio):
        """
        Allocates power asymmetrically to the phases of an inverter based on the demand and power limits.

        Args:
            row (dict): A dictionary containing the demand values for each phase.
            current_generated_power (float): The total power currently being generated.
            max_single_phase_ratio (float): The maximum percentage of power that can be allocated to a single phase.

        Returns:
            tuple: A tuple containing two lists - generated_power_allocation and inverter_power_limit.
                - generated_power_allocation: A list representing the allocated power for each phase.
                - inverter_power_limit: A list representing the power limit for each phase of the inverter.
        """
        if current_generated_power < 0:
            raise ValueError("Current generated cant be negative.")
        if max_single_phase_ratio < 0.33 or max_single_phase_ratio > 1:
            print("Max single phase ratio should be between 0.33 and 1. Setting to default value of symetric inverter.")
            return self.allocate_power_symmetric_inverter(current_generated_power)

        max_demand_phase = max(range(1, 4), key=lambda x: row[f'phase_{x}'])
        max_demand = row[f'phase_{max_demand_phase}']

        if self.with_battery:
            max_demand /= self.inv_eff  # In battery methods we work with DC needs, so we need to convert AC to DC, example: inverter efficiency 0.9, 100W AC needed, 100/0.9 = 111.11W DC needed to cover that

        generated_power_allocation = [0, 0, 0]
        inverter_power_limit = [0, 0, 0]
        remaining_power = current_generated_power

        # Calculate the maximum power that can be allocated to the highest demand phase
        max_power_for_phase = self.invertor_threshold * max_single_phase_ratio

        # Allocate power to the highest demand phase up to the calculated limit
        power_needed = min(max_demand, remaining_power, max_power_for_phase)
        generated_power_allocation[max_demand_phase - 1] = power_needed
        if max_demand >= max_power_for_phase:
            inverter_power_limit[max_demand_phase - 1] = max_power_for_phase
        else:
            inverter_power_limit[max_demand_phase - 1] = max_demand * 1.01 # Add a 1% buffer to the limit
        remaining_power -= power_needed

        # Distribute the remaining power to other phases
        # And set the power limit for each phase based on the inverter threshold
        if remaining_power >= 0:
            other_phases = [1, 2, 3]
            other_phases.remove(max_demand_phase)
            equal_distribution = remaining_power / len(other_phases)
            equal_limit_distribution = (self.invertor_threshold - inverter_power_limit[max_demand_phase - 1]) / len(other_phases)

            for phase in other_phases:
                generated_power_allocation[phase - 1] = equal_distribution
                inverter_power_limit[phase - 1] = equal_limit_distribution

        return generated_power_allocation, inverter_power_limit

    def allocate_power_symmetric_inverter(self, current_generated_power, number_of_phases=3):
        """
        Allocates the current generated power symmetrically across the specified number of phases.

        Args:
            current_generated_power (float): The total currently generated power.
            number_of_phases (int, optional): The number of phases to allocate the power to. Defaults to 3.

        Returns:
            tuple: A tuple containing two lists:
                - current_generated_power_per_phase (list): The allocated power per phase.
                - inverter_power_limit (list): The power limit per phase based on the inverter threshold.
        """
        current_generated_power_per_phase = [current_generated_power / number_of_phases] * 3
        inverter_power_limit = [self.invertor_threshold / number_of_phases] * 3
        return current_generated_power_per_phase, inverter_power_limit

    def process_record_with_battery(self, timestamp, row, current_generated_power, time_of_power_exertion, selling_enabled, max_single_phase_ratio, asymetric_inverter, phases_count, battery_min_capacity, battery_max_capacity, cant_convert_from_battery, previous_timestamp):
        """
        Process a record with battery for a given timestamp and appends records to the lists for plotting and statistics.

        Args:
            - timestamp (int): The timestamp of the record.
            - row (dict): The data row containing power need for each phase.
            - current_generated_power (float): The current generated power from the PV system.
            - time_of_power_exertion (float): The time of power exertion.
            - selling_enabled (bool): Flag indicating if selling excess energy is enabled.
            - max_single_phase_ratio (float): The maximum single phase percentage.
            - asymetric_inverter (bool): Flag indicating if an asymmetric inverter is used.
            - phases_count (int): The number of phases.
            - battery_min_capacity (float): The minimum capacity of the battery.
            - battery_max_capacity (float): The maximum capacity of the battery.
            - cant_convert_from_battery (bool): Flag indicating if energy cannot be converted from the battery thanks to the full use of inverter.
            - previous_timestamp (int): The timestamp of the previous record.

        Returns:
            None
        """
        if asymetric_inverter:
            generated_power_allocation, inverter_power_limit = self.allocate_power_asymmetric_inverter(row, current_generated_power, max_single_phase_ratio)
        else:
            generated_power_allocation, inverter_power_limit = self.allocate_power_symmetric_inverter(current_generated_power)

        power_delta = {}
        energy_delta = {}
        momentary_energy_consumed_all_phases_with_PV = 0
        momentary_energy_consumed_all_phases_no_PV = 0

        self.data_to_plot['timestamp'].append(timestamp)
        self.data_to_plot['power_need_all_phases'].append(row['phase_1'] + row['phase_2'] + row['phase_3'])

        for phase in range(1, phases_count + 1):
            dc_power_consumption = row[f'phase_{phase}'] / self.inv_eff  # Compute how much dc power we need to cover AC consumption, example: inverter efficiency 0.9, 100W AC needed, 100/0.9 = 111.11W DC needed to cover that
            power_delta[phase] = generated_power_allocation[phase-1] - dc_power_consumption
            energy_delta[phase] = power_delta[phase] * time_of_power_exertion
            momentary_energy_consumed_all_phases_no_PV += row[f'phase_{phase}'] * time_of_power_exertion # AC tracking

            self.data_to_plot[f'power_need_phase_{phase}'].append(row[f'phase_{phase}'])
            self.data_to_plot[f'power_generated_phase_{phase}'].append(generated_power_allocation[phase-1])
            self.data_to_plot[f'power_delta_phase_{phase}'].append(power_delta[phase])

            if energy_delta[phase] == 0:
                self.wasted_excess_energy_phase[phase-1].append(self.wasted_excess_energy_phase[phase-1][-1]) # No energy is wasted, record the last value
            # in DC power
            if energy_delta[phase] > 0:  # generated energy is higher than consumed energy

                # If adding the excess to the battery keeps it within its operational range
                if (self.battery_current_storage + energy_delta[phase]) < battery_max_capacity and (self.battery_current_storage + energy_delta[phase]) > battery_min_capacity:
                    self.battery_current_storage += energy_delta[phase] * self.battery_loader_efficiency  # Charge the battery with surplus generated energy (after accounting for the battery loader efficiency), e. 100*0,981 = 98,1W to battery
                    self.wasted_excess_energy_phase[phase-1].append(self.wasted_excess_energy_phase[phase-1][-1])  # No energy is wasted
                    energy_delta[phase] = 0  # The battery successfully stores all excess, no energy excess

                elif (self.battery_current_storage + energy_delta[phase]) < battery_min_capacity:
                    self.battery_current_storage += (generated_power_allocation[phase-1] * self.battery_loader_efficiency) * time_of_power_exertion  # Charge the battery with all generated energy (after accounting for the battery loader efficiency)
                    self.wasted_excess_energy_phase[phase-1].append(self.wasted_excess_energy_phase[phase-1][-1])  # No energy is wasted but we keep track of it
                    energy_delta[phase] = row[f'phase_{phase}'] * time_of_power_exertion  # Assume all consumed energy is drawn from the grid

                elif(self.battery_current_storage + energy_delta[phase]) > battery_max_capacity: # if the battery is full
                    if selling_enabled:
                        self.energy_sold += abs(energy_delta[phase] * self.inv_eff)  # Sell the excess energy back to the grid after accounting inverter efficiency
                        self.wasted_excess_energy_phase[phase-1].append(self.wasted_excess_energy_phase[phase-1][-1]) # No energy is wasted
                    else:
                        # Add the new value to the last value in the list
                        self.wasted_excess_energy_phase[phase-1].append(self.wasted_excess_energy_phase[phase-1][-1] + abs(energy_delta[phase])) #DC
                        self.wasted_excess_energy_points_list.append((timestamp, energy_delta[phase])) #DC
                        self.wasted_excess_energy += abs(energy_delta[phase]) # DC
                    energy_delta[phase] = 0 # energy was sold or couldnt have been 
            
            if energy_delta[phase] < 0:  # consumed energy is higher than generated energy on the given phase
                """
                Here we have to check if the battery has enough energy to cover the deficit and stay above its minimum capacity
                Also we have to check if the invertor can take anymore power from battery to cover the deficit, if not, we have to take the energy from the grid
                That is done here in the following if-else statement
                """
                energy_delta[phase] = abs(energy_delta[phase])  # Convert the result to positive for easier calculations

                if self.battery_current_storage >= (battery_min_capacity + energy_delta[phase]) and cant_convert_from_battery is False: # if the battery has enough energy to cover the deficit and stay above its minimum capacity
                    
                    maximum_energy_to_use_from_battery =  ((inverter_power_limit[phase-1] - generated_power_allocation[phase-1]) * time_of_power_exertion) * self.inv_eff # what power is needed from the battery after accounting for the inverter efficiency losses
                    
                    if energy_delta[phase] <= maximum_energy_to_use_from_battery:
                        self.battery_current_storage -= energy_delta[phase] / self.inv_eff # convert to AC needs
                        energy_delta[phase] = 0 # No additional energy is needed from the grid
                    else:
                        self.battery_current_storage -= maximum_energy_to_use_from_battery
                        energy_delta[phase] = energy_delta[phase] - maximum_energy_to_use_from_battery
                
                self.wasted_excess_energy_phase[phase-1].append(self.wasted_excess_energy_phase[phase-1][-1]) # No energy is wasted
                momentary_energy_consumed_all_phases_with_PV += energy_delta[phase]/self.inv_eff # convert back to AC

            #maybe append it after the 3 phases are done, battery storage change is still accounted in all phases
            self.battery_storage_list.append((timestamp,self.battery_current_storage))
        
        self.total_consumed_energy_with_PV += momentary_energy_consumed_all_phases_with_PV  # Add the net energy used (or surplus if negative) to the total
        self.total_consumed_energy_without_PV += momentary_energy_consumed_all_phases_no_PV  # Track total energy consumption without considering PV generation at all
            
        # Add data to lists for plotting
        self.consumed_energy_list.append((timestamp, self.total_consumed_energy_without_PV))
        self.wasted_excess_energy_list.append((timestamp, self.wasted_excess_energy))
        self.momentary_energy_consumption_without_PV.append((timestamp, momentary_energy_consumed_all_phases_no_PV))
        self.momentary_energy_used_from_grid_list.append((timestamp, momentary_energy_consumed_all_phases_with_PV))
        self.total_energy_used_from_grid_list.append((timestamp, self.total_consumed_energy_with_PV))

    def process_record_NO_battery(self, timestamp, row, current_generated_power, time_of_power_exertion, selling_enabled, max_single_phase_ratio, asymetric_inverter, phases_count):
        """
        Process a record without battery for a given timestamp.

        Args:
            timestamp (): The timestamp of the record.
            row (dict): The data row containing power values for each phase.
            current_generated_power (float): The current generated power.
            time_of_power_exertion (float): The time of power exertion.
            selling_enabled (bool): Flag indicating if selling excess energy is enabled.
            max_single_phase_ratio (float): The maximum single phase percentage.
            asymetric_inverter (bool): Flag indicating if the inverter is asymmetric.
            phases_count (int): The number of phases.

        Returns:
            None
        """
        if asymetric_inverter:
            generated_power_allocation, inverter_power_limit = self.allocate_power_asymmetric_inverter(row, current_generated_power, max_single_phase_ratio)
        else:
            generated_power_allocation, inverter_power_limit = self.allocate_power_symmetric_inverter(current_generated_power, number_of_phases=phases_count)

        power_delta = {}
        energy_delta = {}
        momentary_energy_consumed_all_phases_with_PV = 0
        momentary_energy_consumed_all_phases_no_PV = 0

        self.data_to_plot['timestamp'].append(timestamp)
        self.data_to_plot['power_need_all_phases'].append(sum(row[f'phase_{i}'] for i in range(1, 4)))

        """
                In power_delta, we store the difference between the generated power and the consumed power.
                If the value is - positive, the system generated more power than it consumed.
                                - negative, the system consumed more power than it generated.
        """

        for phase in range(1, phases_count + 1):
            power_delta[phase] = generated_power_allocation[phase-1] - row[f'phase_{phase}']
            momentary_energy_consumed_all_phases_no_PV += row[f'phase_{phase}'] * time_of_power_exertion
            energy_delta[phase] = power_delta[phase] * time_of_power_exertion

            self.data_to_plot[f'power_need_phase_{phase}'].append(row[f'phase_{phase}'])
            self.data_to_plot[f'power_generated_phase_{phase}'].append(generated_power_allocation[phase-1])
            self.data_to_plot[f'power_delta_phase_{phase}'].append(power_delta[phase])

            if energy_delta[phase] == 0:
                self.wasted_excess_energy_phase[phase-1].append(self.wasted_excess_energy_phase[phase-1][-1]) # No energy is wasted, just record it

            if energy_delta[phase] > 0:  # generated energy is higher than consumed energy
                # NOTE: here the energy delta cant be over the invertor threshold, as it is already limited in the previous function
                if selling_enabled:
                    self.energy_sold += abs(energy_delta[phase])  # Sell the excess energy back to the grid
                    self.wasted_excess_energy_phase[phase-1].append(self.wasted_excess_energy_phase[phase-1][-1])  # No energy is wasted
                else:
                    self.wasted_excess_energy_phase[phase-1].append(self.wasted_excess_energy_phase[phase-1][-1] + abs(energy_delta[phase]) / self.inv_eff)  # we track it in DC so we have to convert it back to AC

                energy_delta[phase] = 0

            if energy_delta[phase] < 0:  # consumed energy is higher than generated energy on the given phase
                momentary_energy_consumed_all_phases_with_PV += abs(energy_delta[phase])
                self.wasted_excess_energy_phase[phase-1].append(self.wasted_excess_energy_phase[phase-1][-1])  # No energy is wasted

        self.total_consumed_energy_with_PV += momentary_energy_consumed_all_phases_with_PV  # Add the net energy used (or surplus if negative) to the total
        self.total_consumed_energy_without_PV += momentary_energy_consumed_all_phases_no_PV  # Track total energy consumption without considering PV generation at all

        # Add data to lists for plotting
        self.consumed_energy_list.append((timestamp, self.total_consumed_energy_without_PV))
        self.momentary_energy_consumption_without_PV.append((timestamp, momentary_energy_consumed_all_phases_no_PV))
        self.momentary_energy_used_from_grid_list.append((timestamp, momentary_energy_consumed_all_phases_with_PV))
        self.total_energy_used_from_grid_list.append((timestamp, self.total_consumed_energy_with_PV))

    def process_energy_3_phases_on_grid(self, selling_enabled=True,asymetric_inverter = False,max_single_phase_ratio=0.5, phases_count=3):
        """
        Process the energy consumption and generation data for a 3-phase grid-connected system without battery.

        Args:
            selling_enabled (bool, optional): Flag indicating whether selling excess energy is enabled. 
                                              Defaults to True.

        Returns:
            None
        """

        row_iterator = self.phases_data_frame.iterrows()

        self.data_to_plot = {'timestamp': [],
            **{f'power_generated_phase_{i}': [] for i in range(1, phases_count + 1)},
            **{f'power_need_phase_{i}': [] for i in range(1, phases_count + 1)},
            **{f'power_delta_phase_{i}': [] for i in range(1, phases_count + 1)},
                'power_need_all_phases': []}
        
        last_record_of_previous_hour = None
        previous_timestamp = None
        
        for index, current_generated_power in self.ac_generation_df.items():
            
            # Extract the month, day, and hour from the generation line
            current_hour = (index.strftime('%m-%d'), index.strftime('%H'))

            """
            Limiting the generated power to the invertor nominal threshold.
            """
            # I dont have to use this here, because here I work already with the AC power that went through the inverter
            current_generated_power = self.limit_power_to_invertor_threshold(current_generated_power)

            # setting previous_timestamp as first record of the next hour
            # then when jumping to the cycle we already have set previous timestamp and current timestamp
            # based on that we can compute the time delta (interval) of the time, the power was exerting
            # and from that get the energy = time * power
            if previous_timestamp is None:
                try:
                    previous_timestamp, row = next(row_iterator)
                except StopIteration:
                    break

                        # this is when we have earlier data of consumption than the generation data
            while current_hour > (previous_timestamp.strftime('%m-%d'), previous_timestamp.strftime('%H')):
                try:
                    previous_timestamp, row = next(row_iterator)
                except StopIteration:
                    break

            consumption_time = (previous_timestamp.strftime('%m-%d'), previous_timestamp.strftime('%H'))

            print(current_hour)

            while True:
                try:
                    timestamp, row = next(row_iterator)
                except StopIteration:
                    break

                consumption_time = (timestamp.strftime('%m-%d'), timestamp.strftime('%H'))
                
                # If we've moved to the next hour, keep the last record for comparison
                if consumption_time != current_hour:
                    last_record_of_previous_hour = (timestamp, row)
                    break

                if(self.phases_data_frame.index.freq.freqstr == 'h'):
                    delta_seconds = 3600
                else:
                    delta_seconds = (timestamp - previous_timestamp).total_seconds()
                time_of_power_exertion = delta_seconds / 3600  # Convert to hours for energy calculation

                self.process_record_NO_battery(
                timestamp, row, current_generated_power,
                time_of_power_exertion, selling_enabled,
                max_single_phase_ratio, asymetric_inverter,
                phases_count
                )
                    
                previous_timestamp = timestamp
            
            if last_record_of_previous_hour:
                last_timestamp, last_row = last_record_of_previous_hour
                last_consumption_time = (last_timestamp.strftime('%m-%d'), last_timestamp.strftime('%H'))

                if last_consumption_time != current_hour:
                    # Calculate the delta_seconds and time_of_power_exertion for the last record
                    delta_seconds = (last_timestamp - previous_timestamp).total_seconds()
                    time_of_power_exertion = delta_seconds / 3600

                    self.process_record_NO_battery(
                        last_timestamp, last_row, current_generated_power,
                        time_of_power_exertion, selling_enabled,
                        max_single_phase_ratio, asymetric_inverter,
                        phases_count
                    )
                    previous_timestamp = timestamp

        self.energy_data_df = pd.DataFrame(self.data_to_plot)

    def total_blackout_duration_per_day(self):
        """
        Calculate the total blackout time duration per day. Used in the plot_blackout_duration method.

        Returns:
            pandas.Series: A series containing the total blackout duration per day.
        """
        df = self.blackout_data_df
        # Extracting date from timestamp_start for grouping
        df['date'] = df['timestamp_start'].dt.date
        # Converting duration to timedelta
        df['duration'] = pd.to_timedelta(df['duration'])
        # Summing duration by date
        total_duration_per_day = df.groupby('date')['duration'].sum()
        return total_duration_per_day

    def blackout_days_count(self):
        """
        Calculates the number of unique days when blackouts occurred.

        Returns:
            int: The number of blackout days.
        """
        df = self.blackout_data_df
        # Extracting unique days when blackouts occurred
        blackout_dates = df['timestamp_start'].dt.date.unique()
        # Counting the unique days
        blackout_days = len(blackout_dates)
        return blackout_days
    
    def plot_blackout_duration(self):
        """
        Plots the blackout duration per day.

        This function takes the blackout duration data and plots it as a bar chart,
        showing the blackout duration per day. It also includes days without blackouts
        in the plot.

        Returns:
            None
        """
        df = self.split_blackouts_at_midnight()
        
        # Ensure all date/datetime operations are on datetime types
        df['timestamp_start'] = pd.to_datetime(df['timestamp_start'])
        df['date'] = df['timestamp_start'].dt.date
        
        # Summing duration by date
        total_duration_per_day = df.groupby('date')['duration'].sum().reset_index()
        total_duration_per_day['date'] = pd.to_datetime(total_duration_per_day['date'])

        # Creating a full date range from the start to the end of the blackout records
        date_range = pd.date_range(start=total_duration_per_day['date'].min(), end=total_duration_per_day['date'].max(), freq='D')
        date_df = pd.DataFrame(date_range, columns=['date'])
        
        # Merging the total duration per day with the full date range to include days without blackouts
        plot_df = pd.merge(date_df, total_duration_per_day, how='left', on='date')
        plot_df.set_index('date', inplace=True)
        
        plot_df['duration'] = plot_df['duration'].fillna(pd.Timedelta(seconds=0))
        plot_df['duration_hours'] = plot_df['duration'].dt.total_seconds() / 3600

        # Plotting
        plt.figure(figsize=(10, 6))
        plot_df['duration_hours'].plot(kind='bar', color='blue', alpha=0.8)

        plt.title('Duration of Power Outages per Day (in hours)', fontsize=16)
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Duration of Outages (Hours)', fontsize=14)

        # Set x-axis major formatter to show only month names
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def split_blackouts_at_midnight(self):
        """
        Splits blackouts that span across multiple days at midnight into separate segments. Used to handle blackouts for statistics.

        Returns:
            pandas.DataFrame: DataFrame containing the split blackout segments with their start time, end time, and duration.
        """
        df = self.blackout_data_df
        split_data = []  # List to collect data for new DataFrame

        for _, row in df.iterrows():
            start = pd.to_datetime(row['timestamp_start'])
            end = pd.to_datetime(row['timestamp_end'])

            while start < end:
                if start.hour == 0 and start.minute == 0:
                    # start is already at midnight, find the end of this day or the end of the blackout, whichever is first
                    next_midnight = start + timedelta(days=1)
                    segment_end = min(next_midnight, end)
                else:
                    # start is not at midnight, find the next midnight
                    next_midnight = datetime(start.year, start.month, start.day) + timedelta(days=1)
                    segment_end = min(next_midnight, end)

                # Calculate the duration and add to the list
                if start != segment_end:  # Ensure there's actually time between start and end
                    duration = segment_end - start
                    split_data.append({
                        'timestamp_start': start,
                        'timestamp_end': segment_end,
                        'duration': duration
                    })

                # Move start to the end of the current segment
                start = segment_end

        # Convert to DataFrame
        split_df = pd.DataFrame(split_data)
        split_df['duration'] = pd.to_timedelta(split_df['duration'])

        return split_df

    def check_for_blackout(self, energy_deltas, timestamp):
        """
        Checks if there is a blackout based on the energy deltas and if it is it starts recording it or ends an existing one.

        Args:
            energy_deltas (dict): A dictionary containing energy deltas for different sources.
            timestamp (float): The timestamp of the current data.

        Returns:
            bool: True if a blackout starts, False if a blackout ends.
        """
        if any(delta < 0 for delta in energy_deltas.values()) and not self.blackout_is_already:
            print("Blackout start")
            self.blackout_start = timestamp
            self.blackout_is_already = True
            return True
        elif all(delta >= 0 for delta in energy_deltas.values()) and self.blackout_is_already:
            print("Blackout end")
            self.log_blackout(self.blackout_data, self.blackout_start, end_time=timestamp)
            self.blackout_is_already = False
            return False

    def process_record_off_grid(self, timestamp, row, current_generated_power, time_of_power_exertion, max_single_phase_ratio, asymetric_inverter, phases_count, battery_min_capacity, battery_max_capacity, cant_convert_from_battery, previous_timestamp):
        """
        Process the record for off-grid power simulation.

        Args:
            timestamp (int): The timestamp of the record.
            row (dict): The data of consumption for the record.
            current_generated_power (float): The current generated power.
            time_of_power_exertion (float): The time of power exertion.
            max_single_phase_ratio (float): The maximum single phase percentage.
            asymetric_inverter (bool): Flag indicating if the inverter is asymmetric.
            phases_count (int): The number of phases.
            battery_min_capacity (float): The minimum capacity of the battery.
            battery_max_capacity (float): The maximum capacity of the battery.
            cant_convert_from_battery (bool): Flag indicating if conversion from battery is not possible.
            previous_timestamp (int): The timestamp of the previous record.

        Returns:
            None
        """
        if asymetric_inverter:
            generated_power_allocation, inverter_power_limit = self.allocate_power_asymmetric_inverter(row, current_generated_power, max_single_phase_ratio)
        else:
            generated_power_allocation, inverter_power_limit = self.allocate_power_symmetric_inverter(current_generated_power, phases_count)

        power_delta = {}
        energy_delta = {}
        momentary_energy_consumed_all_phases_with_PV = 0
        momentary_energy_consumed_all_phases_no_PV = 0

        self.data_to_plot['timestamp'].append(previous_timestamp)  # or any appropriate timestamp
        self.data_to_plot['power_need_all_phases'].append(row['phase_1'] + row['phase_2'] + row['phase_3'])

        """
        Here we have to work with DC power, because we are going to store the energy in the battery
        and then we have to convert it back to AC power.
        We will also work with the battery loader which has some efficiency, so we have to account for that.
        """
        energy_deltas_for_blackout_check = {}

        for phase in range(1, phases_count+1):
            # computing needed DC to get equivalent of AC on the phase after accounting inverter efficiency
            power_delta[phase] = generated_power_allocation[phase-1] - (row[f'phase_{phase}'] / self.inv_eff)
            momentary_energy_consumed_all_phases_no_PV += row[f'phase_{phase}'] * time_of_power_exertion # Stat variable, converting to energy E= P*delta(t)
            energy_delta[phase] = power_delta[phase] * time_of_power_exertion # Convert power to energy (based on the time delta)

            self.data_to_plot[f'power_need_phase_{phase}'].append(row[f'phase_{phase}'])
            self.data_to_plot[f'power_generated_phase_{phase}'].append(generated_power_allocation[phase-1])
            self.data_to_plot[f'power_delta_phase_{phase}'].append(power_delta[phase])
    
            if energy_delta[phase] == 0:
                self.wasted_excess_energy_phase[phase-1].append(self.wasted_excess_energy_phase[phase-1][-1]) # No energy is wasted, record the last value
            
            # in DC power
            if energy_delta[phase] > 0:  # generated energy is higher than consumed energy

                # If adding the excess to the battery keeps it within its operational range
                if (self.battery_current_storage + energy_delta[phase]) < battery_max_capacity and (self.battery_current_storage + energy_delta[phase]) > battery_min_capacity:
                    self.battery_current_storage += energy_delta[phase] * self.battery_loader_efficiency
                    self.wasted_excess_energy_phase[phase-1].append(self.wasted_excess_energy_phase[phase-1][-1])  # No energy is wasted
                    energy_delta[phase] = 0  # The battery successfully stores all excess, no energy excess

                elif (self.battery_current_storage + energy_delta[phase]) < battery_min_capacity:
                    self.battery_current_storage += (generated_power_allocation[phase-1] * self.battery_loader_efficiency) * time_of_power_exertion  # Charge the battery with all generated energy (after accounting for the battery loader efficiency)
                    self.wasted_excess_energy_phase[phase-1].append(self.wasted_excess_energy_phase[phase-1][-1])  # No energy is wasted but we keep track of it
                    energy_delta[phase] = row[f'phase_{phase}'] * time_of_power_exertion  # Assume all consumed energy is drawn from the grid

                elif(self.battery_current_storage + energy_delta[phase]) > battery_max_capacity: # if the battery is full
                        # Add the new value to the last value in the list
                        self.wasted_excess_energy_phase[phase-1].append(self.wasted_excess_energy_phase[phase-1][-1] + abs(energy_delta[phase])) #DC
                        self.wasted_excess_energy_points_list.append((timestamp, energy_delta[phase])) #DC
                        self.wasted_excess_energy += abs(energy_delta[phase]) # DC
                        energy_delta[phase] = 0 # energy was  

                # Using this just for blackout check, so it is not converted to absolute value, when there is deficit
                energy_deltas_for_blackout_check[phase] = energy_delta[phase]
            
            if energy_delta[phase] < 0:  # consumed energy is higher than generated energy on the given phase
                """
                Here we have to check if the battery has enough energy to cover the deficit and stay above its minimum capacity
                Also we have to check if the invertor can take anymore power from battery to cover the deficit, if not, we have to take the energy from the grid
                That is done here in the following if-else statement
                """
                energy_delta[phase] = abs(energy_delta[phase])  # Convert the result to positive for easier calculations

                if self.battery_current_storage >= (battery_min_capacity + energy_delta[phase]) and cant_convert_from_battery is False: # if the battery has enough energy to cover the deficit and stay above its minimum capacity
                    
                    maximum_energy_to_use_from_battery = (inverter_power_limit[phase-1] - generated_power_allocation[phase-1]) * time_of_power_exertion
                    
                    if energy_delta[phase] / self.inv_eff < maximum_energy_to_use_from_battery:
                        self.battery_current_storage -= energy_delta[phase] # still in DC, so no conversion to AC
                        energy_delta[phase] = 0 # No additional energy is needed from the grid
                    else:
                        self.battery_current_storage -= maximum_energy_to_use_from_battery
                        energy_delta[phase] = energy_delta[phase] - maximum_energy_to_use_from_battery
                    
                    # Here I should add generated energy not used?
                    # wasted_excess_energy_phase is for tracking energy that is available from battery and solars, but cant be used thx to invertor 
                    # USE MAYBE JUST FOR SOLAR ENERGY UNUSED
                    self.wasted_excess_energy_phase[phase-1].append(self.wasted_excess_energy_phase[phase-1][-1] + energy_delta[phase])

                # track the energy consumed from the grid even when it is blackout?
                momentary_energy_consumed_all_phases_with_PV += energy_delta[phase]/self.inv_eff # convert back to AC

                # Using this for blackout check, so it is not converted to absolute value, when there is deficit
                # This will always be 0 or negative because we are in  consuming energy branch
                energy_deltas_for_blackout_check[phase] = -energy_delta[phase]

            self.battery_storage_list.append((timestamp,self.battery_current_storage))

        self.check_for_blackout(energy_deltas_for_blackout_check, timestamp)

        # Sum the energy consumed from the grid for all phases
        self.total_consumed_energy_with_PV += momentary_energy_consumed_all_phases_with_PV  # Add the net energy used (or surplus if negative) to the total
        self.total_consumed_energy_without_PV += momentary_energy_consumed_all_phases_no_PV  # Track total energy consumption without considering PV generation at all
        
        # Add data to lists for plotting
        self.consumed_energy_list.append((timestamp, self.total_consumed_energy_without_PV))
        self.wasted_excess_energy_list.append((timestamp, self.wasted_excess_energy))
        self.momentary_energy_consumption_without_PV.append((timestamp, momentary_energy_consumed_all_phases_no_PV))
        self.momentary_energy_used_from_grid_list.append((timestamp, momentary_energy_consumed_all_phases_with_PV))
        self.total_energy_used_from_grid_list.append((timestamp, self.total_consumed_energy_with_PV))

    def init_off_grid_variables(self):
        """
        @helper
        Initializes variables for off-grid simulation with battery.

        This method initializes the variables used for off-grid simulation,
        including the blackout data, blackout status, and blackout start time.
        """
        self.blackout_data = {'timestamp_start': [], 'timestamp_end': [], 'duration': []}
        self.blackout_is_already = False
        self.blackout_start = None

    def get_month_day_hour(self, timestamp):
        """
        @helper
        Extract the month, day, and hour from a timestamp.
        """
        return (timestamp.strftime('%m-%d'), timestamp.strftime('%H'))

    def processing_records_with_battery(self, phases_count=3, asymetric_inverter = False,max_single_phase_ratio=0.5, selling_enabled=False, off_grid=False):
        """
        For methods with battery. Processing and synchronizing consumption and generation data for all records.

        Args:
            selling_enabled (bool, optional): Flag indicating whether selling excess energy is enabled. Defaults to True.
            phases_count (int, optional): Number of phases in the system.
            asymetric_inverter (bool, optional): Whether the inverter is asymetric.
            max_single_phase_ratio (float, optional): Max power percentage per phase for asymetric inverters.
            off_grid (bool, optional): Flag indicating if the system is off-grid. Defaults to False.
        Returns:
            None
        """
        # Prepared for future off-grid 1-phase system
        self.phases_count = phases_count

        row_iterator = self.phases_data_frame.iterrows()
    
        # Define battery parameters
        battery_min_capacity, battery_max_capacity = self.set_battery_operational_range()

        self.data_to_plot = {'timestamp': [],
        **{f'power_generated_phase_{i}': [] for i in range(1, phases_count + 1)},
        **{f'power_need_phase_{i}': [] for i in range(1, phases_count + 1)},
        **{f'power_delta_phase_{i}': [] for i in range(1, phases_count + 1)},
            'power_need_all_phases': []}
        
        if off_grid:
            self.init_off_grid_variables()
            
        last_record_of_previous_hour = None
        previous_timestamp = None

        for index, current_generated_power in self.dc_generation_df.items():
            # Extract the month, day, and hour from the generation line
            current_hour = self.get_month_day_hour(index)
            """
            Limiting the generated power to the invertor nominal threshold.
            """
            current_generated_power, cant_convert_from_battery = self.limit_power_to_invertor_threshold(current_generated_power, with_battery=True)

            # setting previous_timestamp as first record of the next hour
            # then when jumping to the cycle we already have set previous timestamp and current timestamp
            # based on that we can compute the time delta (interval) of the time, the power was exerting
            # and from that get the energy = time * power

            if previous_timestamp is None: # if we are at the beginning of the data
                try:
                    previous_timestamp, row = next(row_iterator)
                except StopIteration:
                    break
            
            # this is when we have earlier data of consumption than the generation data
            while current_hour > self.get_month_day_hour(previous_timestamp):
                try:
                    previous_timestamp, row = next(row_iterator)
                except StopIteration:
                    break
                
            consumption_time = self.get_month_day_hour(previous_timestamp)

            print(current_hour)

            while True:
                try:
                    timestamp, row = next(row_iterator)
                except StopIteration:
                    break

                consumption_time = self.get_month_day_hour(timestamp)

                if consumption_time != current_hour:
                    last_record_of_previous_hour = (timestamp, row)
                    break

                if(self.phases_data_frame.index.freq.freqstr == 'h'):
                    delta_seconds = 3600
                else:
                    delta_seconds = (timestamp - previous_timestamp).total_seconds()
                time_of_power_exertion = delta_seconds / 3600  # Convert to hours for energy calculation

                """
                In power_delta, we store the difference between the generated power and the consumed power.
                If the value is - positive, the system generated more power than it consumed.
                                - negative, the system consumed more power than it generated.
                """
                """
                Here we have to work with DC power, because we are going to store the energy in the battery
                and then we have to convert it back to AC power.
                We will also work with the battery loader which has some efficiency, so we have to account for that.
                """
                if off_grid:
                    self.process_record_off_grid(
                        timestamp, row, current_generated_power,
                        time_of_power_exertion, max_single_phase_ratio,
                        asymetric_inverter, phases_count,
                        battery_min_capacity, battery_max_capacity,
                        cant_convert_from_battery, previous_timestamp
                        )
                else:
                    self.process_record_with_battery(
                    timestamp, row, current_generated_power,
                    time_of_power_exertion, selling_enabled,
                    max_single_phase_ratio, asymetric_inverter,
                    phases_count, battery_min_capacity, battery_max_capacity,
                    cant_convert_from_battery, previous_timestamp
                )
                previous_timestamp = timestamp

            if last_record_of_previous_hour:
                last_timestamp, last_row = last_record_of_previous_hour
                last_consumption_time = self.get_month_day_hour(last_timestamp)

                if last_consumption_time != current_hour:
                    # Calculate the delta_seconds and time_of_power_exertion for the last record
                    delta_seconds = (last_timestamp - previous_timestamp).total_seconds()
                    time_of_power_exertion = delta_seconds / 3600

                if off_grid:
                    self.process_record_off_grid(
                        last_timestamp, last_row, current_generated_power,
                        time_of_power_exertion, max_single_phase_ratio,
                        asymetric_inverter, phases_count,
                        battery_min_capacity, battery_max_capacity,
                        cant_convert_from_battery, previous_timestamp
                    )
                else:
                    self.process_record_with_battery(
                        last_timestamp, last_row, current_generated_power,
                        time_of_power_exertion, selling_enabled,
                        max_single_phase_ratio, asymetric_inverter,
                        phases_count, battery_min_capacity, battery_max_capacity,
                        cant_convert_from_battery, previous_timestamp
                    )
                
                previous_timestamp = timestamp

        self.energy_data_df = pd.DataFrame(self.data_to_plot)
        if off_grid:
            self.blackout_data_df = pd.DataFrame(self.blackout_data)
