import numpy as np
import pandas as pd
from functools import partial
from energy_processing.energy_data_processor import EnergyDataProcessor
from solar_production.solar_energy_from_spec_panel import SolarEnergyCalculator
from itertools import product
from deap import base, creator, tools, algorithms
import random

class  PVOptimizer:
    def __init__(self,
            processor: EnergyDataProcessor,
            solar_calculator: SolarEnergyCalculator,
            cost_per_kwh, selling_cost_per_kwh,
            one_panel_cost,
            panel_degradation,
            panel_configurations_list,
            battery_cost,
            invertor_configs,
            invertor_cost,
            installation_cost,
            annual_expenses,
            battery_type, battery_info,
            current_inverter = None,
            current_battery = None,
            with_battery=True,
            off_grid=False,
            battery_sizes=None,
            battery_configs=None,
            ):
        """
        Initializes the PV_Optimizer object with the given parameters.

        Parameters:
        - processor: EnergyDataProcessor object for processing energy data.
        - solar_calculator: SolarEnergyCalculator object for calculating solar energy.
        - cost_per_kwh: Cost per kilowatt-hour of electricity.
        - selling_cost_per_kwh: Cost per kilowatt-hour for selling electricity to the grid.
        - one_panel_cost: Cost of a single solar panel.
        - panel_degradation: Rate of degradation of solar panels over time.
        - panel_configurations_list: List of panel configurations to be considered.
        - battery_sizes: List of battery sizes to be considered. NOTE: If None, the battery size will be randomly generated from the processor's battery capacity. This is here only if user knows exactly the sizes he wants to consider
        - battery_cost: Cost of a single battery.
        - invertor_configs (list of dict): List of dictionaries, each representing an inverter configuration with keys 'size', 'efficiency', and 'cost'.
        - invertor_cost: Cost of an invertor.
        - installation_cost: Cost of installation.
        - battery_type (string): Type of battery.
        - battery_info: Information about the battery found in input_info/battery_info.
        - consumed_energy_without_PV: Energy consumed from the grid without PV system.
        - consumed_energy_with_PV: Energy consumed from the grid with PV system.
        - energy_sold_to_grid: Energy sold to the grid.
        """
        
        self.processor = processor
        self.solar_calculator = solar_calculator
        self.with_battery = with_battery
        self.off_grid = off_grid
        self.invertor_configs = invertor_configs
        self.panel_configurations_list = panel_configurations_list
        self.battery_sizes = battery_sizes

        self.cost_per_kwh = cost_per_kwh
        self.selling_cost_per_kwh = selling_cost_per_kwh
        self.panel_cost = one_panel_cost
        self.annual_expenses = annual_expenses
        self.panel_degradation = panel_degradation
        self.battery_cost = battery_cost
        self.invertor_cost = invertor_cost
        self.battery_type = battery_type
        self.battery_info = battery_info
        self.battery_configs = battery_configs
        self.installation_cost = installation_cost
        self.generated_power = solar_calculator.total_dc_output

        self.current_inverter_config = current_inverter
        self.current_battery_config = current_battery

        self.consumed_energy_with_PV, self.consumed_energy_without_PV, self.energy_sold_to_grid, _ = self.processor.get_processed_simulation_data()

        self.storage_capacity = processor.battery_capacity
        self.number_of_panels = solar_calculator.total_number_of_panels
        self.max_blackout_days = 365  # Default value for the maximum number of days without blackout
        self.params_roi_lists = []
        self.battery_changes = 0

        # Setting up the genetic algorithm
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # Minimize the fitness value, we want the lowest ROI years
        creator.create("Individual", list, fitness=creator.FitnessMin)
        if with_battery:
            self.setup_toolbox_with_battery(panel_configurations_list, battery_sizes)
        else:
            self.setup_toolbox_without_battery(panel_configurations_list)
        self.test()

    def calculate_current_savings_per_year(self):
            #NOTE here can be added the low and high tariff for the energy computation
            """
            Calculates the current savings per year based on the consumed energy without PV, consumed energy with PV,
            energy sold to the grid, cost per kWh, and selling cost per kWh.

            Returns:
                saved_in_a_year (float): The amount of money saved in a year.
            """
            print(f'consumed_energy_without_PV: {self.consumed_energy_without_PV/1000} kWh')
            print(f'consumed_energy_with_PV: {self.consumed_energy_with_PV/1000} kWh')
            print(f'energy_sold_to_grid: {self.energy_sold_to_grid/1000} kWh')
            consumed_energy_diff = (self.consumed_energy_without_PV - self.consumed_energy_with_PV)/1000 # W to kWh
            consumed_energy_diff_cost = consumed_energy_diff * self.cost_per_kwh
            sold_energy = self.energy_sold_to_grid/1000
            sold_energy_cost = sold_energy * self.selling_cost_per_kwh
            saved_in_a_year = consumed_energy_diff_cost + sold_energy_cost - self.annual_expenses
            print(f'saved_in_a_year: {saved_in_a_year}')
            return saved_in_a_year
    
    def cycle_years_opt(self, processor_method):
        """
        This method is for the detail analysis of the ROI. It cycles through the years until the savings reach the total cost
        taking into account the degradation of the panels and battery.
        
        parameters:
        processor_method: the method to process the energy data

        returns:
        roi_years: the number of years it takes to reach the total cost via the savings
        """
        total_cost = self.panel_cost * self.number_of_panels + self.battery_cost + self.invertor_cost + self.installation_cost
        savings = 0.0
        year_number = 0
        # Lists to store the values for each year
        consumed_energy_with_PV_list = []
        consumed_energy_without_PV_list = []
        energy_sold_to_grid_list = []
        wasted_energy_list = []
        savings_list = []

        last_year_savings = 0.0

        # Keep cycling until the savings reach the total cost
        while savings < total_cost:
            # we keep self.generated_power as the original value, because the degradation is cumulative from the original
            generated_power = self.degradate_panel_output_yearly(year_number)
            battery_capacity = self.degradate_battery_capacity_yearly(year_number)
            print(f'Year: {year_number}, generated_power: {self.generated_power}')
            self.processor.reset_data()
            self.processor.dc_generation_df = generated_power
            self.processor.battery_capacity = battery_capacity
            processor_method()
            self.consumed_energy_with_PV, self.consumed_energy_without_PV, self.energy_sold_to_grid, wasted_e = self.processor.get_processed_simulation_data()
            last_year_savings = self.calculate_current_savings_per_year()
            savings += last_year_savings

            # Append the current values to the lists
            consumed_energy_with_PV_list.append(self.consumed_energy_with_PV)
            consumed_energy_without_PV_list.append(self.consumed_energy_without_PV)
            energy_sold_to_grid_list.append(self.energy_sold_to_grid)
            wasted_energy_list.append(wasted_e)
            savings_list.append(savings)

            year_number += 1

        # Calculate the fraction of the year at the end of the loop
        roi_years = year_number - 1 + (total_cost - (savings - last_year_savings)) / last_year_savings

        return roi_years

    def degradate_panel_output_yearly(self, year_number: int):
        """
        Degrades the generated power of the panel based on the panel degradation factor.

        The generated power is multiplied by the panel degradation factor to simulate the yearly degradation
        of the panel's output.

        Parameters:
        None

        Returns:
        None
        """
        return self.generated_power * (1 - (self.panel_degradation * year_number))

    def degradate_battery_capacity_yearly(self, year_number: int):
        """
        Degrades the battery capacity based on the battery degradation factor.

        The battery capacity is multiplied by the battery degradation factor to simulate the yearly degradation
        of the battery's capacity.

        Parameters:
        None

        Returns:
        None
        """
        return self.storage_capacity * (1 - (self.battery_info[self.battery_type]['annual_degradation'] * year_number))
        
    def calculate_battery_cost(self, battery_size):
        """
        Calculate the cost of a battery based on its size.

        Args:
            battery_size (float): The size of the battery in kWh.

        Returns:
            float: The cost of the battery in currency units.
        """
        return self.battery_info[self.battery_type]['cost_per_kWh_capacity'] * battery_size / 1000

    def calculate_ROI_with_battery(self, processor_method):
        """
        Calculate the return on investment (ROI) for the PV optimizer.

        Parameters:
        - processor_method: A function that processes the simulation data.

        Returns:
        - roi: The number of years it takes to cover the initial and replacement costs.

        This method calculates the ROI based on the current savings and the total cost of the PV optimizer system.
        It iteratively calculates the total savings over time and determines the number of years it takes to cover
        the total cost. If the cost is still higher than the savings after the battery lifespan, it adds the cost
        of a new battery and continues the calculation.

        Note: The method assumes that the simulation data has been processed before calling this method.
        """
        self.blackout_days = 0
        processor_method()
        if self.off_grid:
            self.blackout_days = self.processor.blackout_days_count()
            if self.blackout_days > self.max_blackout_days:
                return 1000
        self.consumed_energy_with_PV, self.consumed_energy_without_PV, self.energy_sold_to_grid, wasted_e = self.processor.get_processed_simulation_data()

        # Calculate the return on investment based on the current savings
        current_savings_per_year = self.calculate_current_savings_per_year()
        total_cost = self.panel_cost * self.number_of_panels + self.battery_cost + self.invertor_cost + self.installation_cost
        print(f'Cost of investment before accounting for possible battery change: {total_cost}')
        
        battery_lifespan = self.battery_info[self.battery_type]['lifespan_years']
        
        total_savings = 0.0
        years = 0.0
        max_cycles = 7 # Maximum number of battery replacements allowed
        self.battery_changes = 0
        
        # Calculate how many times you need to replace the battery until the total savings cover the total cost
        while total_cost > total_savings and self.battery_changes < max_cycles:

            # Add the savings for each year_number until you reach the battery lifespan
            for year_number in range(1, battery_lifespan + 1):
                if total_savings + current_savings_per_year > total_cost:
                    # Calculate the fraction of the year_number needed for the remaining savings
                    fraction_year = (total_cost - total_savings) / current_savings_per_year
                    years += fraction_year
                    total_savings = total_cost  # Rounding to break out of the while loop due to small differences
                    break
                else:
                    total_savings += current_savings_per_year
                    years += 1
                    
                if total_savings >= total_cost:
                    break

            # If after the lifespan the cost is still higher than savings, another battery has to be installed
            if total_cost > total_savings:
                total_cost += self.battery_cost
                self.battery_changes += 1

        if self.battery_changes >= max_cycles:
            print(f"ROI calculation stopped after {max_cycles} cycles due to not making an investment return.")
            return years
        
        print("Battery changes done:", self.battery_changes)
        print("Total cost after accounting possible battery changes:", total_cost)
        return years
 
    def find_best_configuration(self, results):
        """
        Finds the best configuration based on the results.

        Parameters:
        results (list): A list of tuples representing the results, where each tuple contains the configuration details and the ROI years.

        Returns:
        tuple: The best configuration with the minimum ROI years.
        """
        best_result = min(results, key=lambda x: x[2])
        return best_result

    def evaluate_individual_with_battery(self, individual):
            """
            Evaluates the fitness of an individual in the optimization process.

            Parameters:
            individual (list): The individual to be evaluated.

            Returns:
            float: The fitness value of the individual.
            """
            # If battery_configs is not set, then the battery size is the first element of the individual and based on this is also called simulate_system_with_battery
            if self.battery_configs is None:
                battery_size = individual[0]
            else:
                battery_config = individual[0]
            inverter_config = individual[-1]
            configurations = []
            it = iter(individual[1:-1])
            for config in self.panel_configurations_list:
                configurations.append({'surface_tilt': config['surface_tilt'], 'surface_azimuth': config['surface_azimuth'],
                                       'strings': next(it), 'modules_per_string': next(it)})
            if self.battery_configs is None:
                return self.simulate_system_with_battery(panel_configurations=configurations,inverter_config=inverter_config,battery_size=battery_size),
            else:
                return self.simulate_system_with_battery(panel_configurations=configurations, inverter_config=inverter_config, battery_config=battery_config),

    def evaluate_individual_without_battery(self, individual):
            """
            Evaluates the fitness of an individual in the optimization process.

            Parameters:
            individual (list): The individual to be evaluated.

            Returns:
            float: The fitness value of the individual.
            """
            inverter_config = individual[-1]
            configurations = []
            it = iter(individual[:-1])
            for config in self.panel_configurations_list:
                configurations.append({'surface_tilt': config['surface_tilt'], 'surface_azimuth': config['surface_azimuth'],
                                       'strings': next(it), 'modules_per_string': next(it)})
            return self.simulate_system_without_battery(panel_configurations=configurations, inverter_config=inverter_config),

    def set_battery_configs(self, battery_config, battery_size=None):
        """
        Sets the battery configurations for the PV optimizer.

        Parameters:
        - battery_config (dict): A dictionary containing the battery configuration details.
        - battery_size (float, optional): The size of the battery in kilowatt-hours (kWh). Defaults to None.

        Returns:
        None
        """
        if self.battery_configs is None: # The configs are set with priority
            self.processor.battery_capacity = battery_size
            self.battery_cost = self.calculate_battery_cost(battery_size)
        else:
            self.processor.battery_capacity = battery_config['size']
            self.battery_type = battery_config['type']
            self.processor.battery_type = battery_config['type']
            self.battery_cost = battery_config['cost']

    def simulate_system_with_battery(self, panel_configurations, inverter_config, battery_size=None, battery_config=None):
            """
            Simulates the PV power plant system with the given panel configurations and battery size.
            
            Args:
                panel_configurations (list): List of panel configurations.
                battery_size (float): Size of the battery.
            
            Returns:
                float: Number of years of return on investment (ROI) for the system.
            """
            self.set_and_run_new_solar_config(inverter_config, panel_configurations)
            self.set_new_processor_config(inverter_config)            
            self.set_battery_configs(battery_config, battery_size)
            # Process the energy and calculate ROI

            processor_method = partial(self.processor.processing_records_with_battery, asymetric_inverter=self.processor.asymetric_inverter, off_grid=self.off_grid, selling_enabled=self.processor.selling_enabled, max_single_phase_ratio=self.processor.max_single_phase_ratio)

            years_of_ROI = self.calculate_ROI_with_battery(processor_method)

            # This method takes into account the degradation of the panels and battery
            # But is really computationally expensive and slow, fit for very detailed analysis, but not for optimization
            #years_of_ROI  = self.cycle_years_opt(processor_method)
            if self.battery_configs is None:
                self.params_roi_lists.append((panel_configurations, battery_size, self.battery_changes, years_of_ROI, inverter_config, self.blackout_days))
            else:
                self.params_roi_lists.append((panel_configurations, battery_config, self.battery_changes, years_of_ROI, inverter_config, self.blackout_days))

            return years_of_ROI

    def set_and_run_new_solar_config(self, inverter_config, panel_configurations):
        """
        Set the new solar configuration for the optimizer.

        params:
         - inverter_config: The current inverter configuration.
         - panel_configurations: List of panel configurations.
        """
        self.solar_calculator.invertor_threshold = inverter_config['size']
        self.solar_calculator.invertor_efficiency = inverter_config['efficiency']
        self.solar_calculator.array_configs = panel_configurations
        self.solar_calculator.run_in_optimizer()
        self.number_of_panels = self.solar_calculator.calculate_total_number_of_panels()

    def set_new_processor_config(self, inverter_config):
        """
        Sets the new processor configuration.

        Args:
            inverter_config (dict): A dictionary containing the inverter configuration parameters.
                - 'size': The size of the inverter.
                - 'efficiency': The efficiency of the inverter.

        Returns:
            None
        """
        self.processor.reset_data()
        self.processor.dc_generation_df = self.solar_calculator.total_dc_output
        self.processor.ac_generation_df = self.solar_calculator.total_ac_output
        self.processor.invertor_threshold = inverter_config['size']
        self.processor.asymetric_inverter = inverter_config.get('asymetric', False)
        self.processor.max_single_phase_ratio = inverter_config.get('max_single_phase_ratio', 0.4)
        self.processor.inv_eff = inverter_config['efficiency']
        self.invertor_cost = inverter_config['cost']

    def simulate_system_without_battery(self, panel_configurations, inverter_config):
            """
            Simulates the PV power plant system without considering battery configurations.

            Args:
            panel_configurations (list): List of panel configurations.
            inverter_config (dict): Inverter configuration.

            Returns:
            float: Number of years of return on investment (ROI) for the system.
            """
            self.set_and_run_new_solar_config(inverter_config, panel_configurations)
            self.set_new_processor_config(inverter_config)

            self.number_of_panels = self.solar_calculator.calculate_total_number_of_panels()
            self.invertor_cost = inverter_config['cost']
            print(f'Inverter threshold: {self.processor.invertor_threshold}')
            # Process the energy and calculate ROI
            processor_method = partial(self.processor.process_energy_3_phases_on_grid, selling_enabled=self.processor.selling_enabled, asymetric_inverter = self.processor.asymetric_inverter , max_single_phase_ratio=self.processor.max_single_phase_ratio)
            years_of_ROI = self.calculate_ROI_without_battery(processor_method)

            self.params_roi_lists.append((panel_configurations, inverter_config, years_of_ROI))

            return years_of_ROI

    def run_genetic_algorithm(self, ngen=10, pop_size=30):
        pop = self.create_initial_population(pop_size)
        hof = tools.HallOfFame(1)

        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Now raising the mutpb to 0.4, because the mutation is not happening enough
        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=0.8, mutpb=0.4, ngen=ngen, stats=stats, halloffame=hof, verbose=True)
        
        # Save statistics to a CSV file
        df = pd.DataFrame(log)
        df.to_csv('output_of_simulations/statistics.csv', index=False)

        best_individual = hof[0]
        best_configuration = self.decode_individual(best_individual)
        return best_configuration, best_individual.fitness.values

    def decode_individual(self, individual):
            """
            Decodes an individual from the genetic algorithm population.

            Parameters:
            individual (list): The individual to be decoded.

            Returns:
            tuple: A tuple containing the decoded configurations and battery config.
            """
            panel_configurations = []
            if self.with_battery:
                battery_config = individual[0]
                it = iter(individual[1:])
                for config in self.panel_configurations_list:
                    panel_configurations.append({'surface_tilt': config['surface_tilt'], 'surface_azimuth': config['surface_azimuth'],
                                        'strings': next(it), 'modules_per_string': next(it)})
                inverter_config = individual[-1]
                return panel_configurations, battery_config, inverter_config
            else:
                it = iter(individual)
                for config in self.panel_configurations_list:
                    panel_configurations.append({'surface_tilt': config['surface_tilt'], 'surface_azimuth': config['surface_azimuth'],
                                        'strings': next(it), 'modules_per_string': next(it)})
                inverter_config = individual[-1]
                return panel_configurations, inverter_config
    
    def set_days_without_blackout(self, days: int): 
        """ Computes the best configuration to ensure no blackout for the given number of days.
        Parameters:
        days (int): The number of days without blackout.

        Returns:
        tuple: The best configuration that ensures no blackout for the given number of days.
        """
        self.max_blackout_days = days
        
    def random_battery_size(self, battery_lower_bound, battery_upper_bound):
        # This will generate a random number within the bounds, but only in multiples of 1000
        return random.randint(battery_lower_bound // 1000, battery_upper_bound // 1000) * 1000

    def custom_mutate_with_battery(self, individual, low, up, hybrid_inverters, indpb):
        """
        Mutates the given individual by randomly changing its values within the specified range.
        
        Parameters:
            individual (list): The individual to be mutated. individual[0] is the battery size, individual[1] is the inverter configuration, and the rest are the panel configurations.
            low (list): The lower bounds for each value in the individual.
            up (list): The upper bounds for each value in the individual.
            indpb (float): The probability of mutating each value in the individual.

        NOTE: invertor configuration is on the last index because it would affect the indexes in low and up lists, which are only for battery size and panel configurations.
        
        Returns:
            tuple: A tuple containing the mutated individual.
        """
        # Iterate over all attributes in the individual
        for i in range(len(individual)):
            if random.random() < indpb:
                if i == 0:  # Assuming the first value is the battery size/config
                    # Mutate battery size within specified bounds, rounding to nearest 1000
                    if self.battery_sizes is None and self.battery_configs is None:
                        individual[i] = random.randint(low[i] // 1000, up[i] // 1000) * 1000 # Choose a new battery size randomly within the bounds
                    else:
                        if self.battery_configs is not None:
                            individual[i] = random.choice(self.battery_configs)
                        else:
                            individual[i] = random.choice(self.battery_sizes) # Choose a new battery size randomly from the list
                elif i == len(individual) - 1:  # Assuming the last value is the inverter configuration
                    # Choose a new inverter configuration randomly from the list
                    individual[i] = random.choice(hybrid_inverters)
                else:
                    # Mutate other attributes within their bounds
                    individual[i] = random.randint(low[i], up[i])

        return (individual,) #dont delete brackets or , !!!  DEAP requires a tuple as output
    
    def create_initial_population(self, num_individuals):
        # Create the initial population as a list of individuals
        population = self.toolbox.population(n=num_individuals - 1)  # Create one less to add the specific configuration

        # Create the specific individual using the toolbox
        specific_individual = self.toolbox.individual()
        index_in_config = 0

        # Set the specific individual with current settings dynamically
        if self.with_battery:
            if self.battery_sizes is None and self.battery_configs is None:
                specific_individual[0] = self.processor.battery_capacity  # Assuming the first index is battery
            elif self.battery_configs is not None:
                specific_individual[0] = self.current_battery_config  # Assuming proper current config extraction
            else:
                specific_individual[0] = random.choice(self.battery_sizes)
            index_in_config = 1


        for config in self.panel_configurations_list:
            specific_individual[index_in_config] = config['strings']
            specific_individual[index_in_config+1] = config['modules_per_string']
            index_in_config += 2
        
        # Assuming the last index is for the inverter
        specific_individual[-1] = self.current_inverter_config

        # Add this properly initialized specific individual to the population
        population.append(specific_individual)

        return population

    def setup_toolbox_with_battery(self, panel_configurations_list, battery_sizes=None):
        """
        Set up the toolbox for the genetic algorithm optimization.

        Parameters:
        - panel_configurations_list (list): A list of dictionaries containing the panel configurations.
        - battery_sizes (list): A list of battery sizes.

        Returns:
        None
        """
        self.toolbox = base.Toolbox()
        
        # Attribute generators for battery sizes and panel configurations.
        if battery_sizes is None and self.battery_configs is None:
            battery_lower_bound = round((self.processor.battery_capacity / 2) / 1000) * 1000  # Lower bound rounded to the nearest thousand
            battery_upper_bound = round((self.processor.battery_capacity * 2.5) / 1000) * 1000  # Upper bound rounded to the nearest thousand
            self.toolbox.register("attr_battery_size", self.random_battery_size, battery_lower_bound, battery_upper_bound)
        else:
            if self.battery_configs is not None:
                self.toolbox.register("attr_battery_config", random.choice, self.battery_configs)
                attributes = [self.toolbox.attr_battery_config]
            else:
                self.toolbox.register("attr_battery_size", random.choice, battery_sizes)
                attributes = [self.toolbox.attr_battery_size]

        # Attribute generator for inverter configurations.
        # Filter hybrid inverters only
        hybrid_inverters = [inv for inv in self.invertor_configs if inv.get('hybrid', False)]
        self.toolbox.register("attr_inverter_config", random.choice, hybrid_inverters)
        #self.toolbox.register("attr_inverter_config", random.choice, self.invertor_configs)

        # Attribute generators for the number of strings and modules per string for each panel configuration.
        for index_in_config, config in enumerate(panel_configurations_list):
            self.toolbox.register(f"attr_string_{index_in_config}", random.randint, config['min_strings'], config['max_strings'])
            self.toolbox.register(f"attr_module_{index_in_config}", random.randint, config['min_modules'], config['max_modules'])
            attributes.append(self.toolbox.__getattribute__(f"attr_string_{index_in_config}"))
            attributes.append(self.toolbox.__getattribute__(f"attr_module_{index_in_config}"))
        attributes.append(self.toolbox.attr_inverter_config)

        # Register the individual creation function that cycles through each attribute to initialize an individual.
        self.toolbox.register("individual", tools.initCycle, creator.Individual, attributes, n=1)

        # Register the population creation function that repeats the individual creation for the whole population.
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Genetic operators
        self.toolbox.register("evaluate", self.evaluate_individual_with_battery)  # Fitness evaluation function for an individual.
        self.toolbox.register("mate", tools.cxTwoPoint)  # Crossover function: two-point crossover.
        self.toolbox.register("select", tools.selTournament, tournsize=3)  # Selection function: tournament selection with size 3.

        # Mutation settings
        if battery_sizes is None and self.battery_configs is None:
            low_bounds = [battery_lower_bound]
            up_bounds = [battery_upper_bound]
        else:
            low_bounds = [1000] # Default lower bound for battery size
            up_bounds = [25000] # Default upper bound for battery size

        for config in panel_configurations_list:
            low_bounds.extend([config['min_strings'], config['min_modules']])
            up_bounds.extend([config['max_strings'], config['max_modules']])

        self.toolbox.register("mutate", self.custom_mutate_with_battery, low=low_bounds, up=up_bounds, hybrid_inverters=hybrid_inverters, indpb=0.4)

    def setup_toolbox_without_battery(self, panel_configurations_list):
        """
        Set up the toolbox for the genetic algorithm optimization without considering battery configurations.

        Parameters:
        - panel_configurations_list (list): A list of dictionaries containing the panel configurations.

        Returns:
        None
        """
        self.toolbox = base.Toolbox()

        # Attribute generator for inverter configurations.
        self.toolbox.register("attr_inverter_config", random.choice, self.invertor_configs)

        attributes = []
        # Attribute generators for the number of strings and modules per string for each panel configuration.
        for index_in_config, config in enumerate(panel_configurations_list):
            self.toolbox.register(f"attr_string_{index_in_config}", random.randint, config['min_strings'], config['max_strings'])
            self.toolbox.register(f"attr_module_{index_in_config}", random.randint, config['min_modules'], config['max_modules'])
            attributes.append(self.toolbox.__getattribute__(f"attr_string_{index_in_config}"))
            attributes.append(self.toolbox.__getattribute__(f"attr_module_{index_in_config}"))
        attributes.append(self.toolbox.attr_inverter_config)

        # Register the individual creation function that cycles through each attribute to initialize an individual.
        self.toolbox.register("individual", tools.initCycle, creator.Individual, attributes, n=1)

        # Register the population creation function that repeats the individual creation for the whole population.
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Genetic operators
        self.toolbox.register("evaluate", self.evaluate_individual_without_battery)  # Fitness evaluation function for an individual.
        self.toolbox.register("mate", tools.cxTwoPoint)  # Crossover function: two-point crossover.
        self.toolbox.register("select", tools.selTournament, tournsize=3)  # Selection function: tournament selection with size 3.

        # Mutation settings
        low_bounds = []
        up_bounds = []

        for config in panel_configurations_list:
            low_bounds.extend([config['min_strings'], config['min_modules']])
            up_bounds.extend([config['max_strings'], config['max_modules']])
        
        # NOTE: changed indpb to 0.4
        self.toolbox.register("mutate", self.custom_mutate_without_battery, low=low_bounds, up=up_bounds, indpb=0.4)

    def custom_mutate_without_battery(self, individual, low, up, indpb):
        """
        Mutates the given individual by randomly changing its values within the specified range.

        Parameters:
        individual (list): The individual to be mutated.
        low (list): The lower bounds for each value in the individual.
        up (list): The upper bounds for each value in the individual.
        indpb (float): The probability of mutating each value in the individual.

        Returns:
        tuple: A tuple containing the mutated individual.
        """
        for i in range(len(individual)):
            if random.random() < indpb:
                if i == len(individual) - 1:  # Assuming the last value is the inverter configuration
                    # Choose a new inverter configuration randomly from the list
                    individual[i] = random.choice(self.invertor_configs)
                else:
                    individual[i] = random.randint(low[i], up[i])

        return individual,

    def calculate_ROI_without_battery(self, processor_method):
        """
        Calculate the return on investment (ROI) focusing only on panels and inverters, without battery considerations.
        """
        self.blackout_days = 0
        processor_method()

        # Gather data from the energy processor
        self.consumed_energy_with_PV, self.consumed_energy_without_PV, self.energy_sold_to_grid, _ = self.processor.get_processed_simulation_data()
        total_cost = self.panel_cost * self.number_of_panels + self.invertor_cost + self.installation_cost

        current_savings_per_year = self.calculate_current_savings_per_year()
        savings_list = []
        print(f'Cost of investment beffore accounting for annual expenses: {total_cost}')

        total_savings = 0.0
        years = 0.0

        while total_cost > total_savings:
            if total_savings + current_savings_per_year > total_cost:
                # Calculate the fraction of the year_number needed for the remaining savings
                fraction_year = (total_cost - total_savings) / current_savings_per_year
                years += fraction_year
                total_savings = total_cost  # Rounding to break out of the while loop due to small differences
                break
            
            total_savings += current_savings_per_year
            years += 1
            if total_savings >= total_cost:
                break
        
        print("Total cost after accounting annual expenses:", total_cost)
        return years

    def print_best_config_off_grid(self, ddf, max_blackout_days):
        # Filter the DataFrame
        filtered_ddf = ddf[ddf['Blackout Days'] <= max_blackout_days]

        if not filtered_ddf.empty:
            # Find the row with the lowest 'ROI Years' in the filtered DataFrame
            min_roi_years_row = filtered_ddf.loc[filtered_ddf['Years of ROI'].idxmin()]
            print('Panel configuration:', min_roi_years_row['Panel configuration'], 'Battery Size:', min_roi_years_row['Battery Size'])
            print('ROI Years:', min_roi_years_row['Years of ROI'])
            print('Blackout Days:', min_roi_years_row['Blackout Days'])
            print('Inverter config:', min_roi_years_row['Inverter config'])
            print('Battery exchanges done:', min_roi_years_row['Battery exchanges done'])
            return min_roi_years_row
        else:
            # If no such row exists, print the row with the lowest 'Blackout Days'
            min_blackout_days_row = ddf.loc[ddf['Blackout Days'].idxmin()]
            print('Panel configuration:', min_blackout_days_row['Panel configuration'], 'Battery Size:', min_blackout_days_row['Battery Size'])
            print('Blackout Days:', min_blackout_days_row['Blackout Days'])
            print('Inverter config:', min_blackout_days_row['Inverter config'])
            print('Battery Changes:', min_blackout_days_row['Battery exchanges done'])
            print('If you want to achieve desired maximal amount of blackout days, you need to try these steps:')
            print('1. Increase the battery size.')
            print('2. Increase the maximal number of panels.')
            print('3. Increase the inverter size.')
            return min_blackout_days_row

    def run_best_config(self, config):
        """
        Runs the simulation with the best configuration.

        Args:
            config (dict): The configuration parameters for the simulation.

        Returns:
            float: The number of years to achieve return on investment (ROI).
        """
        self.processor.reset_data()
        if self.with_battery:
            self.set_and_run_new_solar_config(inverter_config=config[-1], panel_configurations=config[0])
            self.set_new_processor_config(config[-1])
            self.set_battery_configs(config[-2])
            processor_method = partial(self.processor.processing_records_with_battery, asymetric_inverter=self.processor.asymetric_inverter, off_grid=self.off_grid, selling_enabled=self.processor.selling_enabled, max_single_phase_ratio=self.processor.max_single_phase_ratio)
            years_of_ROI = self.calculate_ROI_with_battery(processor_method)
        else:
            self.set_and_run_new_solar_config(inverter_config=config[-1], panel_configurations=config[0])
            self.set_new_processor_config(config[-1])
            processor_method = partial(self.processor.process_energy_3_phases_on_grid, selling_enabled=self.processor.selling_enabled, asymetric_inverter=self.processor.asymetric_inverter, max_single_phase_ratio=self.processor.max_single_phase_ratio)
            years_of_ROI = self.calculate_ROI_without_battery(processor_method)
        return years_of_ROI
        
    def check_new_roi(self, new_roi, original_roi):
        if new_roi < original_roi:
            print('The new configuration is better than the original one.')
        else:
            if new_roi == original_roi:
                print('The new configuration is the same as the original one.')
            else:
                print('The new configuration is worse than the original one. Mistake could occur or low number of population and generations was set.')

    def compute_original_configuration(self, processor_method):
        """
        Compute the original configuration of the PV power plant system.

        Parameters:
        - processor_method: A function that processes the simulation data.

        Returns:
        - years_of_ROI: The number of years it takes to achieve ROI (Return on Investment).
        """
        self.processor.reset_data()
        if self.with_battery:
            years_of_ROI = self.calculate_ROI_with_battery(processor_method)
        else:
            years_of_ROI = self.calculate_ROI_without_battery(processor_method)
        return years_of_ROI

    def was_new_configuration_better(self, new_roi, original_roi):
        """
        Check if the new configuration is better than the original one.

        Parameters:
        - new_roi: The ROI years of the new configuration.
        - original_roi: The ROI years of the original configuration.

        Returns:
        - bool: True if the new configuration is better, False otherwise.
        """
        return new_roi < original_roi

    def set_config_to_off_grid_config(self, row):
        config = [None, None, None]
        config[0] = row['Panel configuration']
        config[1] = row['Battery Size']
        config[2] = row['Inverter config']
        return config

### UNUSED METHODS

    def test(self):
        individual = self.toolbox.individual()
        print("Before mutation:", individual)
        self.toolbox.mutate(individual)
        print("After mutation:", individual)