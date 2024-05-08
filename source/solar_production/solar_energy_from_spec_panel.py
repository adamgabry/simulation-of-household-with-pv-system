"""
@file solar_energy_calculator.py
@brief This file contains the SolarEnergyCalculator class, which is responsible for calculating the solar energy output of a photovoltaic system.
@Author: Adam Gabrys

NOTE: many functions are using the pvlib library, which is a Python library for simulating the performance of photovoltaic energy systems.

"""
import pvlib
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from pvlib.location import Location
from pvlib.pvsystem import PVSystem

class SolarEnergyCalculator:
    def __init__(self, latitude, longitude,
                array_configs,
                start, end,
                panel_info, invertor_threshold,
                pvgis_poa = False,
                invertor_efficiency=0.975,
                input_csvs_file_path = None):
        self.location = Location(
            latitude=latitude,
            longitude=longitude,
            name="FIT VUT"
        )
        self.start = start
        start_date = datetime.strptime(start, "%Y-%m-%d %H:%M")
        self.start_year = start_date.year
        self.end = end
        self.input_csvs_file_path = input_csvs_file_path
        self.panel_info = panel_info
        self.array_configs = array_configs
        self.pvgis_poa = pvgis_poa

        self.irradiance_data = None
        self.plane_irradiance_data = None
        self.hourly_solar_position = None

        self.invertor_threshold = invertor_threshold
        self.invertor_efficiency = invertor_efficiency
        self.total_number_of_panels = self.calculate_total_number_of_panels()
        self.run()

    def run(self):
        """
        Runs the simulation for each configuration in the array_configs.
        Calculates the solar position, angle of incidence, effective irradiance, cell temperature,
        single diode model parameters, DC output, and accumulates the results.
        Stores the DC and AC outputs in the respective class variables.
        """
        self.dc_outputs = {}  # Dictionary to store each configuration's output

        for i, config in enumerate(self.array_configs):
            # Set up the current configuration
            self.setup_current_config(config)

            if self.input_csvs_file_path is not None:
                self.poa_data = self.fetch_weather_underground_data(Fahrenheit=True, config=config)
            else:
                self.poa_data = self.fetch_poa_data(config)

            if i == 0:
                #initialize the total_dc_output here, because the date_range can change with WU data
                total_dc_output = pd.Series(0, index=pd.date_range(start=self.start, end=self.end, freq="h"))

            self.solar_pos = self.calculate_solar_position()
            self.aoi = self.calculate_aoi()
            self.effective_irradiance = self.calculate_effective_irradiance()
            self.temp_cell = self.calculate_cell_temperature()

            self.I_L_ref, self.I_o_ref, self.R_s, self.R_sh_ref, self.a_ref, self.Adjust = self.calculate_single_diode_model_params()

            # Calculate outputs for this configuration
            dc_output = self.calculate_dc_mpp()

            # Store the output in the dictionary
            self.dc_outputs[f'Config {i + 1}: Tilt {config["surface_tilt"]}, Azimuth {config["surface_azimuth"]}'] = dc_output

            # Accumulate the results
            total_dc_output += dc_output

        self.total_dc_output = total_dc_output
        self.total_ac_output = self.calculate_ac_output()

    def run_in_optimizer(self):
        """
        Runs the optimizer for the solar energy production simulation.
        It was created to be used in the optimizer, where the irradiance data is already fetched/available and in right format.

        This method iterates through each configuration in the array_configs and calculates the solar energy output
        for each configuration. It stores the output in a dictionary and accumulates the results to calculate the
        total DC and AC output.

        Returns:
            None
        """
        self.dc_outputs = {}  # Dictionary to store each configuration's output

        for i, config in enumerate(self.array_configs):
            # Set up the current configuration
            self.setup_current_config(config)

            if self.pvgis_poa: # We have to use the irradiance data not specific for given config, so cant use POA here
                self.irradiance_data = pd.read_csv("weather_data/ghi_dni_dhi_data_for_timeout.csv", index_col=0, parse_dates=True)
            
            self.poa_data = self.compute_poa_for_optimizer(self.calculate_solar_position(), self.irradiance_data)

            if i == 0:
                #initialize the total_dc_output here, because the date_range can change with WU data
                total_dc_output = pd.Series(0, index=pd.date_range(start=self.start, end=self.end, freq="h"))

            self.solar_pos = self.calculate_solar_position()
            self.aoi = self.calculate_aoi()
            self.effective_irradiance = self.calculate_effective_irradiance()
            self.temp_cell = self.calculate_cell_temperature()

            self.I_L_ref, self.I_o_ref, self.R_s, self.R_sh_ref, self.a_ref, self.Adjust = self.calculate_single_diode_model_params()

            # Calculate outputs for this configuration
            dc_output = self.calculate_dc_mpp()

            # Store the output in the dictionary
            self.dc_outputs[f'Config {i + 1}: Tilt {config["surface_tilt"]}, Azimuth {config["surface_azimuth"]}'] = dc_output

            # Accumulate the results
            total_dc_output += dc_output

        self.total_dc_output = total_dc_output
        self.total_ac_output = self.calculate_ac_output()

    def calculate_total_number_of_panels(self):
        """
        Calculates the total number of solar panels based on the configurations.

        Returns:
            int: The total number of solar panels.
        """
        total_panels = 0
        for config in self.array_configs:
            total_panels += config['strings'] * config['modules_per_string']
        return total_panels

    def setup_current_config(self, config):
            """
            Set up the current configuration of the solar panels installed.

            Parameters:
            config (dict): A dictionary containing the configuration parameters.
                - surface_tilt (float): The tilt angle of the solar panel surface in degrees.
                - surface_azimuth (float): The azimuth angle of the solar panel surface in degrees.
                - strings (int): The number of strings in the solar panel system.
                - modules_per_string (int): The number of modules per string in the solar panel system.

            Returns:
            None
            """
            self.surface_tilt = config['surface_tilt']
            self.surface_azimuth = config['surface_azimuth']
            self.strings = config['strings']
            self.modules_per_string = config['modules_per_string']
            self.solar_pos = self.calculate_solar_position()

    def load_and_combine_csvs(self, directory_path):
        """
        Loads and combines multiple CSV files from a directory into a single DataFrame.

        Args:
            directory_path (str): The path to the directory containing the CSV files.

        Returns:
            pd.DataFrame: The combined DataFrame containing data from all the CSV files.

        Raises:
            FileNotFoundError: If no matching CSV files are found in the directory.
        """
        # Get a list of all .csv files in the directory
        all_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

        # Sort the files to ensure we start with the earliest
        all_files.sort()

        # Initialize an empty list to store DataFrames
        data_frames = []

        for file_name in all_files:
            file_path = os.path.join(directory_path, file_name)
            # Skip loading empty files
            if os.path.getsize(file_path) > 0:
                df = pd.read_csv(file_path, parse_dates=[0], index_col=0)
                # Skip adding DataFrames that are completely empty or all NA after loading
                if not df.empty and not df.dropna(how='all').empty:
                    data_frames.append(df)

        # Concatenate all non-empty DataFrames
        if data_frames:
            combined_data = pd.concat(data_frames, ignore_index=False)
            # Sort the data by the index (which is the datetime) to ensure it's in chronological order
            combined_data.sort_index(inplace=True)
            print(combined_data.tail())

            # Set self.start and self.end based on the combined data
            self.start = combined_data.index[0].replace(minute=0, second=0)
            self.start_year = self.start.year
            self.end = combined_data.index[-1].replace(minute=0, second=0)

            return combined_data
        else:
            raise FileNotFoundError("No matching CSV files found in the directory.")

    def fetch_weather_underground_data(self, Fahrenheit=False, config=None):
        """
        Fetches weather data from Weather Underground API and performs data processing.

        Args:
            Fahrenheit (bool, optional): Flag indicating whether the temperature should be converted to Fahrenheit. Defaults to False.
            config (str, optional): Configuration information. Defaults to None.

        Returns:
            pandas.DataFrame: Processed weather data.

        Raises:
            Exception: If an error occurs during data processing.

        """
        try:
            # Read the CSV data into a DataFrame
            data = self.load_and_combine_csvs(self.input_csvs_file_path)

            # Rename columns for consistency
            data = data.rename(columns={'Solar': 'ghi'})
            data = data.rename(columns={'Temperature': 'temp_air'})
            data = data.rename(columns={'Wind Speed': 'wind_speed'})
            
            # Ensure index is datetime and localize to None to remove timezone
            data.index = pd.to_datetime(data.index).tz_localize(None)

            # Change the year of each datetime object in the index to match the start year
            data.index = data.index.map(lambda dt: dt.replace(year=self.start_year))
            
            # Calculate solar position
            solar_position = self.location.get_solarposition(times=data.index)

            # Resample data to hourly frequency and interpolate missing values
            resampled_data = data.resample('h').mean().interpolate(limit=3)
            
            # Convert temperature to Celsius if Fahrenheit flag is True
            if Fahrenheit:
                resampled_data['temp_air'] = (resampled_data['temp_air']-32) *(5/9)

            # Calculate solar position for resampled data
            hourly_solar_position = self.location.get_solarposition(times=resampled_data.index)
            
            # Fill missing values with 0
            resampled_data = resampled_data.fillna(0)
            
            # Calculate direct normal irradiance using dirint model
            dirint_model_data = pvlib.irradiance.dirint(ghi=resampled_data['ghi'], solar_zenith=hourly_solar_position['zenith'], times=resampled_data.index)
            dirint_model_data = dirint_model_data.fillna(0)
            aligned_dni = dirint_model_data.reindex(resampled_data.index, fill_value=0)
            resampled_data['dni'] = aligned_dni
            
            # Calculate diffuse horizontal irradiance
            resampled_data['dhi'] = resampled_data['ghi'] - resampled_data['dni'] * np.cos(hourly_solar_position['zenith'].apply(np.radians))

            # Save resampled data to CSV file
            resampled_data.to_csv("weather_data/ghi_dni_dhi_data_for_timeout.csv")
            self.irradiance_data = resampled_data

            # Save resampled data to another CSV file
            resampled_data.to_csv("experiments/pvgis_data_test/resampled_ghi_data.csv", index=True)

            # Calculate plane of array (POA) irradiance components
            poa_irradiance = pvlib.irradiance.get_total_irradiance(
                self.surface_tilt,
                self.surface_azimuth,
                hourly_solar_position['apparent_zenith'],
                hourly_solar_position['azimuth'],
                resampled_data['dni'],
                resampled_data['ghi'],
                resampled_data['dhi']
            )

            self.plane_irradiance_data = poa_irradiance

            mapped_data = pd.DataFrame({
                'poa_global': poa_irradiance['poa_global'],
                'poa_direct': poa_irradiance['poa_direct'],
                'poa_diffuse': poa_irradiance['poa_diffuse'],
                'temp_air': resampled_data['temp_air'],
                'wind_speed': resampled_data['wind_speed']
            }, index=resampled_data.index)

            #self.plot_poa(mapped_data)

            print(f"POA data were calculated successfully for config: {config}.")
            mapped_data.to_csv("mapped_data.csv")
            return mapped_data

        except Exception as e:
            print(f"An error occurred when processing Weather Underground data: {e}")
            return pd.DataFrame()  # Return an empty DataFrame in case of an error
    
    def fetch_poa_data(self, config):
        """
        Fetches the Plane of Array (POA) data for solar irradiance.

        Args:
            config (str): Configuration for fetching the POA data.

        Returns:
            pd.DataFrame: DataFrame containing the formatted POA data.

        Raises:
            ValueError: If there is an error retrieving or calculating the POA data.
        """
        try:
            # Fetch the raw TMY data
            if self.pvgis_poa: # this is set if we want to use the POA data from PVGIS and not call the API - for testing purposes
                try:
                    data = pd.read_csv("weather_data/ghi_dni_dhi_data_for_timeout.csv", index_col=0, parse_dates=True)
                except:
                    print("Fetching TMY data from PVGIS...")
                    data, months_selected, inputs, metadata = pvlib.iotools.get_pvgis_tmy(
                        self.location.latitude,
                        self.location.longitude,
                        url='https://re.jrc.ec.europa.eu/api/5.0/',
                        map_variables=True,
                        #timeout=60
                    )
            else:
                print("Fetching TMY data from PVGIS...")
                data, months_selected, inputs, metadata = pvlib.iotools.get_pvgis_tmy(
                    self.location.latitude,
                    self.location.longitude,
                    url='https://re.jrc.ec.europa.eu/api/v5_2/',
                    map_variables=True,
                    #timeout=60
                )
            # Ensure index is datetime and localize to None to remove timezone
            data.index = pd.to_datetime(data.index).tz_localize(None)
            
            # Change the year of each datetime object in the index to match the start year
            data.index = data.index.map(lambda dt: dt.replace(year=self.start_year))

            data = data[(data.index >= self.start) & (data.index <= self.end)]
            
            data.to_csv("weather_data/ghi_dni_dhi_data_for_timeout.csv")

            # Calculate solar position for the location and times
            solar_position = self.location.get_solarposition(times=data.index)

            data[['dni', 'ghi', 'dhi']].to_csv("experiments/pvgis_data_test/tmy_raw_data.csv", index=True)

            self.irradiance_data = data

            # Calculate the POA irradiance components
            poa_irradiance = pvlib.irradiance.get_total_irradiance(
                self.surface_tilt,
                self.surface_azimuth,
                solar_position['apparent_zenith'],
                solar_position['azimuth'],
                data['dni'],
                data['ghi'],
                data['dhi'],
            )
            
            self.plane_irradiance_data = poa_irradiance

            # Create a DataFrame for the final formatted data
            formatted_data = pd.DataFrame({
                'poa_global': poa_irradiance['poa_global'],
                'poa_direct': poa_irradiance['poa_direct'],
                'poa_diffuse': poa_irradiance['poa_diffuse'],
                'temp_air': data['temp_air'],
                'wind_speed': data['wind_speed']
            }, index=data.index)

            print(f"POA data were calculated successfully for config: {config}.")
            return formatted_data
        except Exception as e:
            print(f"An error occurred when retrieving and calculating POA data: {e}")
            raise ValueError("Invalid input data") from e
        
    def compute_poa_for_optimizer(self, solar_position, data):
        """
        Compute the plane of array (POA) irradiance components for the optimizer.
        NOTE: This method is used for the optimizer, where the irradiance data is already fetched/available.

        Args:
            solar_position (dict): Solar position data including apparent zenith and azimuth angles.
            data (pd.DataFrame): DataFrame containing solar irradiance data including dni, ghi, and dhi.

        Returns:
            pd.DataFrame: DataFrame containing the formatted POA irradiance data along with temperature and wind speed.

        Raises:
            Exception: If an error occurs when retrieving and calculating POA data.

        """
        try:
            # Calculate the POA irradiance components
            data.index = pd.to_datetime(data.index).tz_localize(None)
            
            poa_irradiance = pvlib.irradiance.get_total_irradiance(
                self.surface_tilt,
                self.surface_azimuth,
                solar_position['apparent_zenith'],
                solar_position['azimuth'],
                data['dni'],
                data['ghi'],
                data['dhi'],
            )

            self.plane_irradiance_data = poa_irradiance
            
            # Create a DataFrame for the final formatted data
            formatted_data = pd.DataFrame({
                'poa_global': poa_irradiance['poa_global'],
                'poa_direct': poa_irradiance['poa_direct'],
                'poa_diffuse': poa_irradiance['poa_diffuse'],
                'temp_air': data['temp_air'],
                'wind_speed': data['wind_speed']
            }, index=data.index)

            print("Retrieved and calculated POA data successfully.")
            return formatted_data
        except Exception as e:
            print(f"An error occurred when retrieving and calculating POA data: {e}")
            return pd.DataFrame()  # Return an empty DataFrame in case of an error
        
    def calculate_solar_position(self):
        """
        Calculates the solar position for the specified time range.

        Returns:
            pandas.DataFrame: Solar position data for each hour in the specified time range.
        """
        return self.location.get_solarposition(times=pd.date_range(start=self.start, end=self.end, freq="h"))

    def calculate_aoi(self):
        """
        Calculate the angle of incidence (AOI) of solar radiation on the panel.

        Returns:
            The angle of incidence in degrees.
        """
        return pvlib.irradiance.aoi(self.surface_tilt, self.surface_azimuth, self.solar_pos.apparent_zenith, solar_azimuth=self.solar_pos.azimuth)

    """
    The following method was adapted from the PV-Tutorials repository by PV-Tutorials.
    @author: PV-Tutorials
    @cited 2023-11-26 
    @url: https://github.com/PV-Tutorials/pyData-2021-Solar-PV-Modeling/blob/main/Tutorial%20A%20-%20Single%20Diode%20Model.ipynb
    @note: part 9
    """
    def calculate_effective_irradiance(self):
        """
        Calculate the effective irradiance based on the incident angle modifier (IAM).

        Returns:
            float: The effective irradiance.
        """
        # Calculate the incidence angle modifier
        iam = pvlib.iam.ashrae(self.aoi)
        effective_irradiance = self.poa_data["poa_direct"] * iam + self.poa_data["poa_diffuse"]
        return effective_irradiance

    def calculate_cell_temperature(self):
        """
        Calculates the cell temperature of a solar panel.

        The cell temperature is calculated using the Faiman model, which takes into account the global horizontal irradiance,
        ambient air temperature, and wind speed.

        Returns:
            float: The calculated cell temperature in degrees Celsius.
        """
        cell_temperature = pvlib.temperature.faiman(self.poa_data["poa_global"], self.poa_data["temp_air"], self.poa_data["wind_speed"])
        return cell_temperature

    def calculate_single_diode_model_params(self):
            """
            Calculates the parameters of the single diode model for the solar panel.

            Returns:
                Tuple[float, float, float, float, float, float]: The calculated parameters of the single diode model.
                    - I_L_ref: Light-generated current at reference conditions (A)
                    - I_o_ref: Diode saturation current at reference conditions (A)
                    - R_s: Series resistance (ohm)
                    - R_sh_ref: Shunt resistance at reference conditions (ohm)
                    - a_ref: Ideality factor at reference conditions
                    - Adjust: Adjustment factor for the diode equation
            """
            I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust = pvlib.ivtools.sdm.fit_cec_sam(
            celltype=self.panel_info['celltype'],
            v_mp=self.panel_info['v_mp'],
            i_mp=self.panel_info['i_mp'],
            v_oc=self.panel_info['v_oc'],
            i_sc=self.panel_info['i_sc'],
            alpha_sc=self.panel_info['alpha_sc'],
            beta_voc=self.panel_info['beta_voc'],
            gamma_pmp = self.panel_info['gamma_pmax'],
            cells_in_series=self.panel_info['cells_in_series'],
            temp_ref=self.panel_info['temp_ref']
            )
            return I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust

    def calculate_params_for_IV_curve(self):
            """
            Calculates the CEC parameters for the IV curve based on the effective irradiance and cell temperature.
            Also uses the parameters for CEC single diode model based on the panel techsheet.

            Returns:
                dict: A dictionary containing the CEC parameters.
            """
            # Using the parameters from calculate_single_diode_model_params to calculate the CEC parameters
            cec_params = pvlib.pvsystem.calcparams_cec(
                self.effective_irradiance,
                self.temp_cell,
                alpha_sc=self.panel_info['alpha_sc'],
                a_ref=self.a_ref,
                I_L_ref=self.I_L_ref,
                I_o_ref=self.I_o_ref,
                R_sh_ref=self.R_sh_ref,
                R_s=self.R_s,
                Adjust=self.Adjust
            )
            return cec_params
        
    def calculate_mppt(self):
        """
        Calculates the maximum power point (MPP) of the solar panel.

        Returns:
            Tuple: A tuple containing the voltage and current at the MPP.
        """
        mpp = pvlib.pvsystem.max_power_point(*self.calculate_params_for_IV_curve(), method='newton')
        return mpp

    def calculate_dc_mpp(self):
        """
        Calculates the DC power at the maximum power point (MPP) of the PV system.

        Returns:
            float: The DC power at the MPP.
        """
        system = PVSystem(modules_per_string=self.modules_per_string, strings_per_inverter=self.strings)
        dc_scaled = system.scale_voltage_current_power(self.calculate_mppt())
        return dc_scaled.p_mp

    def calculate_ac_output(self):
        """
        Calculates the AC output power of the solar panel system.

        Returns:
            float: The AC output power in watts.
        """
        return pvlib.inverter.pvwatts(pdc=self.total_dc_output, pdc0=self.invertor_threshold, eta_inv_nom=self.invertor_efficiency)
    
    def save_output(self, dc_output_file, ac_output_file):
        """
        Save the total DC and AC power output to CSV files.

        Parameters:
        - dc_output_file (str): The file path to save the total DC power output.
        - ac_output_file (str): The file path to save the total AC power output.
        """
        total_dc_output_df = self.total_dc_output.to_frame(name='Total DC Power Output (W)')
        total_dc_output_df.to_csv(dc_output_file)

        ac_output_df = self.total_ac_output.to_frame(name='Total AC Power Output (W)')
        ac_output_df.to_csv(ac_output_file)

    def get_optimal_angles(self):
        #NOTE: This method was not used in the final implementation due to handling more panel configurations and letting user keep how he wants it, but it can be used to retrieve the optimal angles for the location
        # method is planned to be used in the future for further optimization of the panel configurations
        try:
            data, inputs, metadata = pvlib.iotools.get_pvgis_hourly(
                self.location.latitude,
                self.location.longitude,
                optimalangles=True,
                map_variables=True
            )
            slope_value = inputs['mounting_system']['fixed']['slope']['value']
            azimuth_value = inputs['mounting_system']['fixed']['azimuth']['value']
            print(f"Optimal slope: {slope_value}, Optimal azimuth: {azimuth_value}")
            return slope_value, azimuth_value + 180 # Add 180 to the azimuth value to get the correct value (library is inconsistent with pvlib get_hourly/tmy methods)
        except Exception as e:
            print(f"An error occurred when retrieving optimal angles: {e}")
            return self.surface_tilt, self.surface_azimuth

    def plot_dhi_dni_ghi(self):
        """
        Plots the development of DNI, GHI, and DHI based on the provided data.
        This method was used for testing purposes.

        This method reads the DNI, GHI, and DHI data from CSV files and plots them on a graph.
        The first subplot shows the DNI, GHI, and DHI after estimating GHI and DHI using the DIRINT model
        with WeatherUnderground data.
        The second subplot shows the DNI, GHI, and DHI from TMY PVGIS data.

        Returns:
            None
        """
        import matplotlib.dates as mdates
        fig, axs = plt.subplots(2, figsize=(10,12))

        self.dhi_data = pd.read_csv("experiments/pvgis_data_test/resampled_ghi_data.csv", index_col=0, parse_dates=True)

        axs[0].plot(self.dhi_data['dni'], label='DNI')
        axs[0].plot(self.dhi_data['ghi'], label='GHI')
        axs[0].plot(self.dhi_data['dhi'], label='DHI')

        axs[0].set_title('Vývoj DNI, GHI a DHI po estimaci GHI a DHI DIRINT modelem(WeatherUnderground data)')
        axs[0].text(0.01, -0.13, 'Čas', transform=axs[0].transAxes)
        axs[0].set_ylabel('(W/m^2)')
        axs[0].legend()

        # Format the x-axis to not show the years
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))


        tmy_data = pd.read_csv("experiments/pvgis_data_test/tmy_raw_data.csv", index_col=0, parse_dates=True)

        axs[1].plot(tmy_data['dni'], label='DNI')
        axs[1].plot(tmy_data['ghi'], label='GHI')
        axs[1].plot(tmy_data['dhi'], label='DHI')

        axs[1].set_title('Vývoj DNI, GHI, a DHI (TMY PVGIS data)')
        axs[1].text(0.01, -0.13, 'Čas', transform=axs[1].transAxes)
        axs[1].set_ylabel('(W/m^2)')
        axs[1].legend()

        # Format the x-axis to not show the years
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))


        plt.tight_layout()
        plt.show()

    def plot_solar_position(self, solar_position):
        """
        Plots the solar zenith and azimuth over time.

        Parameters:
        - solar_position (dict): A dictionary containing the solar position data. Used for mainly for testing purposes.

        Returns:
        - None
        """
        fig, axs = plt.subplots(2, figsize=(10, 6))

        # Plot the solar zenith
        axs[0].plot(solar_position['apparent_zenith'], label='Apparent Zenith')
        axs[0].set_title('Solar Apparent Zenith over Time')
        axs[0].set_ylabel('Degrees')
        axs[0].legend()

        # Plot the solar azimuth
        axs[1].plot(solar_position['azimuth'], label='Azimuth', color='orange')
        axs[1].set_title('Solar Azimuth over Time')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Degrees')
        axs[1].legend()

        # Display the plot
        plt.tight_layout()
        plt.show()
            
    def plot_configurations(self):
        """
        Plots the power output of different configurations of solar panels.

        This method creates a figure with subplots arranged vertically, where each subplot represents
        the power output of a specific configuration of solar panels. The power output is plotted against
        the date. If there is only one configuration, the subplot is wrapped in a list.

        Parameters:
        None

        Returns:
        None
        """
        # Number of configurations
        num_configs = len(self.dc_outputs)

        # Create a figure with subplots arranged vertically
        fig, axes = plt.subplots(nrows=num_configs, ncols=1, figsize=(16, 9), sharex=True)
        
        # If there is only one configuration, wrap axes in a list
        if num_configs == 1:
            axes = [axes]
        
        for ax, (label, output) in zip(axes, self.dc_outputs.items()):
            output.plot(ax=ax, label=label, color='tab:blue')
            ax.set_title(label)
            ax.set_ylabel('DC Power Output (W)', fontsize=12)
            ax.legend()

        # Set a common X label
        axes[-1].set_xlabel('Date', fontsize=14)

        plt.tight_layout()
        plt.show()

    def plot_output(self, czech_labels=False):
        """
        Plots the DC and AC power output of the photovoltaic system.

        Parameters:
        czech_labels: bool, optional (default = False)
            If True, the labels will be in Czech. If False, the labels will be in English.

        Returns:
        None
        """
        fig, ax = plt.subplots(figsize=(16, 9))

        if czech_labels:
            self.total_dc_output.plot(ax=ax, label='Celkový výkon DC (W)')
            self.total_ac_output.plot(ax=ax, label='Celkový výkon AC (W)')
            ax.set_title('Výkonový graf fotovoltaického systému', fontsize=16)
            ax.set_ylabel('Výkon (W)', fontsize=14)
        else:
            self.total_dc_output.plot(ax=ax, label='Total DC power (W)')
            self.total_ac_output.plot(ax=ax, label='Total AC power (W)')
            ax.set_title('Power graph of the photovoltaic system', fontsize=16)
            ax.set_ylabel('Power (W)', fontsize=14)

        ax.legend()

        plt.show()

    def print_monthly_output(self):
        """
        Prints the monthly DC and AC output of the solar panel.

        This method resamples the data to monthly frequency and calculates the sum of DC and AC output.
        The values are then converted from Watts to kilowatts (kW) and printed to the console.

        Parameters:
        None

        Returns:
        None
        """
        # Resample data to monthly frequency and calculate sum
        total_dc_output_monthly = self.total_dc_output.resample('ME').sum()/1000 # Convert W to kW
        total_ac_output_monthly = self.total_ac_output.resample('ME').sum()/1000 # Convert W to kW

        # Print the monthly output
        print("Monthly DC Output (kWh):")
        print(total_dc_output_monthly)
        print("\nMonthly AC Output (kWh):")
        print(total_ac_output_monthly)

    def print_annual_output(self):
        """
        Prints the annual DC and AC output of the solar panel system.

        Returns:
        None
        """
        # Calculate annual output
        total_dc_output_annual = self.total_dc_output.sum()/1000
        total_ac_output_annual = self.total_ac_output.sum()/1000

        # Print the annual output
        print(f"Annual DC Output (kWh): {total_dc_output_annual:.2f}")
        print(f"Annual AC Output (kWh): {total_ac_output_annual:.2f}")
        print("Maximum DC Output (W):", self.total_dc_output.max())
        print("Maximum AC Output (W):", self.total_ac_output.max())

    def plot_energy_output(self, plot_months=False, czech_labels=False):
        fig, ax = plt.subplots(figsize=(16, 9))

        if plot_months:
            # Resample data to monthly frequency and calculate sum
            total_dc_output_monthly = self.total_dc_output.resample('ME').sum()/1000 # Convert W to kW
            total_ac_output_monthly = self.total_ac_output.resample('ME').sum()/1000 # Convert W to kW

            month_names = [month[:3] for month in total_dc_output_monthly.index.month_name()]

            if czech_labels:
                ax.bar(month_names, total_dc_output_monthly.values, label='Celková vyrobená energie DC (kWh)', width=0.4, align='center')
                ax.bar(month_names, total_ac_output_monthly.values, label='Celková vyrobená energie AC (kWh)', width=0.4, align='edge')
                ax.set_ylabel('Energie (kWh)', fontsize=14)
                ax.set_title('Měsíční graf vyrobené energie fotovoltaického systému', fontsize=16)
            else:
                ax.bar(month_names, total_dc_output_monthly.values, label='Total DC energy produced (kWh)', width=0.4, align='center')
                ax.bar(month_names, total_ac_output_monthly.values, label='Total AC energy produced (kWh)', width=0.4, align='edge')
                ax.set_ylabel('Energy (kWh)', fontsize=14)
                ax.set_title('Monthly energy production of the photovoltaic system', fontsize=16)
        else:
            if czech_labels:
                self.total_dc_output.plot(ax=ax, label='Celkový výkon DC (W)')
                self.total_ac_output.plot(ax=ax, label='Celkový výkon AC (W)')
                ax.set_title('Výkonový graf fotovoltaického systému')
                ax.set_ylabel('Výkon (W)')
            else:
                self.total_dc_output.plot(ax=ax, label='Total DC power (W)')
                self.total_ac_output.plot(ax=ax, label='Total AC power (W)')
                ax.set_title('Power graph of the photovoltaic system')
                ax.set_ylabel('Power (W)')
        ax.legend()
        plt.show()

    def plot_poa(self, data_frame):
        """
        Plots the parts of the sunlight, namely POA (Plane of Array) Global, POA Direct, and POA Diffuse.

        Parameters:
        - data_frame: pandas DataFrame containing the irradiance data

        Returns:
        None
        """
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Plotting irradiance data
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Irradiance (W/m^2)', color='tab:red')
        ax1.plot(data_frame.index, data_frame['poa_global'], label='POA Global', color='tab:red')
        ax1.plot(data_frame.index, data_frame['poa_direct'], label='POA Direct', color='tab:orange')
        ax1.plot(data_frame.index, data_frame['poa_diffuse'], label='POA Diffuse', color='tab:green')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.legend(loc='upper left')

        fig.tight_layout()
        plt.show()

    def plot_eff_irradiance(self):
        """
        Plot the effective irradiance over time.

        This method creates a line plot of the effective irradiance values
        against the corresponding time index.

        Parameters:
            None

        Returns:
            None
        """
        plt.figure(figsize=(14, 7))
        plt.plot(self.effective_irradiance.index, self.effective_irradiance, label='Effective Irradiance')
        plt.xlabel('Time')
        plt.ylabel('Effective Irradiance')
        plt.title('Effective Irradiance over Time')
        plt.legend()
        plt.show()

    def plot_eff_irradiation(self):
        plt.figure(figsize=(14, 7))

        # assume that the data is already in hours so W/m^2*h = Wh/m^2
        # but do the check later
        effective_irradiation = self.effective_irradiance
        freq = pd.infer_freq(effective_irradiation.index)

        #If the data are not in hourly frequency, resample it to hourly frequency and calculate mean
        if freq != 'h':
            # Resample data to hourly frequency and calculate mean to get energy in Wh/m^2
            effective_irradiation = self.effective_irradiance.resample('h').mean()

        # Resample data to monthly frequency and calculate mean
        monthly_eff_irradiance = effective_irradiation.resample('ME').sum()/1000 # Convert W/m^2 to kW/m^2

        # Get month names and convert to three-letter abbreviations
        month_names = [month[:3] for month in monthly_eff_irradiance.index.month_name()]

        # Plot the resampled monthly data
        plt.bar(month_names, monthly_eff_irradiance.values, label='Monthly Effective Irradiation (kWh/m^2)')

        plt.xlabel('Month')
        plt.ylabel('Effective Irradiation (kWh/m^2)', fontsize=10)
        plt.title('Effective Irradiation over Time - Monthly Averages', fontsize=14)
        plt.legend()
        plt.show()

    def print_eff_irradiation(self):

        effective_irradiation = self.effective_irradiance
        freq = pd.infer_freq(effective_irradiation.index)
        #If the data are not in hourly frequency, resample it to hourly frequency and calculate mean
        if freq != 'h':
            # Resample data to hourly frequency and calculate mean to get energy in Wh/m^2
            effective_irradiation = self.effective_irradiance.resample('h').mean()

        # Resample data to monthly frequency and calculate sum
        total_eff_irradiance_monthly = self.effective_irradiance.resample('ME').sum()/1000
        print("Monthly Effective Irradiation (kWh/m^2):", total_eff_irradiance_monthly)

    def plot_cell_temperature(self, cell_temperature):
        """
        Plots the cell temperature over time.
        Method can be used to visualize the cell temperature calculation and help test if we are not working with Fahrenheit data.

        Parameters:
        - cell_temperature (pandas.Series): The cell temperature data.

        Returns:
        None
        """
        plt.figure(figsize=(14, 7))
        plt.plot(cell_temperature.index, cell_temperature, label='Cell Temperature')
        plt.xlabel('Time')
        plt.ylabel('Cell Temperature')
        plt.title('Cell Temperature over Time')
        plt.legend()
        plt.show()
