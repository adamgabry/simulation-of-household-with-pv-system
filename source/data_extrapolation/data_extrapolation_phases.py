"""
Author: Adam Gabrys
Date: 2024-02-05
Description: A file containing the DataExtrapolation class for extrapolating data for power plant simulation.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

class DataExtrapolation:
    """
    A class for extrapolating data for power plant simulation.

    Methods:
    - extrapolate_all_phases_data: Extrapolates data for all phases.
    - extrapolate_last_month: Extrapolates data for the last month.
    - extrapolate_with_STL: Extrapolates data using STL decomposition.
    - plot_extrapolated_data: Plots the extrapolated data.
    """
    def __init__(self):
        pass

    def extrapolate_last_month(self, df_resampled, phase_names):
        """
        Extrapolates data for the last month.

        Parameters:
        - df_resampled (DataFrame): The resampled data.
        - phase_names (list): The names of the phases.

        Returns:
        - full_year_df (DataFrame): The extrapolated data for the last month.
        """
        # Calculate the last date in the DataFrame
        last_date = df_resampled.index.max()
        
        # Determine the first day of the last month
        first_of_last_month = last_date.replace(day=1)
        
        # Filter the last month's data using .loc
        last_month_data = df_resampled.loc[first_of_last_month:last_date]

        # Define the full year range
        start_of_year = pd.date_range(start='2023-01-01', end=df_resampled.index.min(), freq='5s', inclusive='left')
        end_of_year = pd.date_range(start=df_resampled.index.max() + pd.Timedelta(seconds=5), end='2023-12-31 23:59:55', freq='5s')

        # Prepare the container for the full year data
        full_year_data = []

        for phase in phase_names:
            # Repeat the last month's pattern to cover the extrapolation periods
            repeat_factor_start = int(np.ceil(len(start_of_year) / len(last_month_data)))
            repeat_factor_end = int(np.ceil(len(end_of_year) / len(last_month_data)))

            # Create the extrapolated data for both periods, ensuring the series has the same name for later concatenation
            extrapolated_start_series = pd.concat([last_month_data[phase]] * repeat_factor_start, ignore_index=True)[:len(start_of_year)]
            extrapolated_start_series.index = start_of_year
            extrapolated_start_series.name = phase

            extrapolated_end_series = pd.concat([last_month_data[phase]] * repeat_factor_end, ignore_index=True)[:len(end_of_year)]
            extrapolated_end_series.index = end_of_year
            extrapolated_end_series.name = phase

            # Concatenate the extrapolated start, original, and extrapolated end data for the phase
            combined_series = pd.concat([
                extrapolated_start_series,
                df_resampled[phase],
                extrapolated_end_series
            ])

            # Store the combined series in the list
            full_year_data.append(combined_series)

        # Combine all phases into a single DataFrame
        full_year_df = pd.concat(full_year_data, axis=1)

        return full_year_df

    def plot_extrapolated_data(self, full_year_df, num_phases):
        """
        Plots the extrapolated data.

        Parameters:
        - full_year_df (DataFrame): The extrapolated data for all phases.
        - num_phases (int): The number of phases.

        Returns:
        - None
        """
        fig, axes = plt.subplots(num_phases, 1, figsize=(15, num_phases * 3), sharex=True)

        colors = ['b', 'g', 'r']  # List of colors

        for i, phase in enumerate(full_year_df.columns):
            axes[i].plot(full_year_df.index, full_year_df[phase], label=phase, linewidth=1, color=colors[i % len(colors)])
            axes[i].set_ylabel('Výkon (W)', fontsize=14)
            axes[i].legend(loc='upper right')

        axes[-1].set_xlabel('Čas')
        plt.suptitle('Extrapolovaná data všech 3 fází', fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def extrapolate_all_phases_data(self, df_resampled, period, phase_names):
        """
        NOTE: This method is not used in the final implementation.
        This method takes the resampled data, period for seasonal decomposition, and phase names as input.
        It performs seasonal decomposition on each phase's data and extrapolates the missing values for the entire year.
        The extrapolation is done by combining the trend, seasonal, and random residual components of the data.


        Parameters:
        - df_resampled (DataFrame): The resampled data.
        - period (int): The period for seasonal decomposition.
        - phase_names (list): The names of the phases.

        Returns:
        - full_year_df (DataFrame): The extrapolated data for all phases.
        """
        full_year_data = {}

        start_of_year = pd.date_range(start='2023-01-01', end=df_resampled.index.min(), freq='5s', inclusive='left')
        end_of_year = pd.date_range(start=df_resampled.index.max(), end='2023-12-31 23:59:55', freq='5s', inclusive='right')

        for phase in phase_names:
            decomp_results = seasonal_decompose(df_resampled[phase], model='additive', period=period)

            trend_values = decomp_results.trend.dropna()
            trend_start = np.interp((start_of_year - df_resampled.index[0]).total_seconds(),
                                    (decomp_results.trend.dropna().index - df_resampled.index[0]).total_seconds(),
                                    decomp_results.trend.dropna().values)
            trend_end = np.full(shape=(len(end_of_year),), fill_value=trend_values.iloc[-1])

            seasonal_cycle = decomp_results.seasonal[-24*60*60//5:] # Use the last day of the seasonal cycle by 5 secs interval, wouldnt have been hardcoded but wasnt used in the final implementation
            full_year_seasonal = np.tile(seasonal_cycle, int(np.ceil((len(start_of_year) + len(df_resampled) + len(end_of_year)) / len(seasonal_cycle))))
            seasonal_start = full_year_seasonal[:len(start_of_year)]
            seasonal_end = full_year_seasonal[-len(end_of_year):]

            window_size = 144 * 4
            smoothed_residuals = decomp_results.resid.rolling(window=window_size, center=True).mean().bfill().ffill()

            random_indices_start = np.random.randint(0, len(smoothed_residuals), size=len(start_of_year))
            random_indices_end = np.random.randint(0, len(smoothed_residuals), size=len(end_of_year))
            random_residuals_start = smoothed_residuals.iloc[random_indices_start].values
            random_residuals_end = smoothed_residuals.iloc[random_indices_end].values

            predicted_start = np.maximum(0, trend_start + seasonal_start + random_residuals_start)
            predicted_end = np.maximum(0, trend_end + seasonal_end + random_residuals_end)

            full_year_data[phase] = pd.concat([
                pd.DataFrame(predicted_start, index=start_of_year, columns=[phase]),
                df_resampled[[phase]],
                pd.DataFrame(predicted_end, index=end_of_year, columns=[phase])
            ])
        
        full_year_df = pd.concat(full_year_data.values(), axis=1)
        return full_year_df
    