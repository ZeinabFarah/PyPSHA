from modules.seismic_distance_calculator import SeismicDistanceCalculator
from modules.magnitude_frequency_distribution import MagnitudeFrequencyDistribution
import numpy as np
import pandas as pd
import pygmm

class EarthquakeGenerator:    
    def __init__(self, setup, magnitude_frequency_distribution, ground_motion_equation_model, ground_motion_type, num_scenarios, period=None):
        """
        Initializes the EarthquakeGenerator with necessary parameters for seismic hazard analysis.

        Parameters:
        -----------
        setup : object
            An instance of setup class containing site and source data.
        magnitude_frequency_distribution : str
            Name of the magnitude frequency distribution function to use (gr_recurrence_law).
        ground_motion_equation_model : str
            Name of the ground motion equation model to use (BooreStewartSeyhanAtkinson2014).
        ground_motion_type : str
            Type of ground motion to analyze ('PGA', 'PGV', or 'SA').
        num_scenarios : int
            Number of earthquake scenarios to generate.
        period : float, optional
            The period of interest in seconds for spectral acceleration (SA).
        """
        self.setup = setup
        self.num_scenarios = num_scenarios
        self.magnitude_frequency_distribution = getattr(MagnitudeFrequencyDistribution, magnitude_frequency_distribution)
        self.ground_motion_type = ground_motion_type
        self.period = period
        self.ground_motion_equation_model = ground_motion_equation_model
        
    def generate_scenarios(self):  
        """
        Generates earthquake scenarios with associated magnitudes and sources.

        Returns:
        --------
        tuple: A tuple containing arrays of magnitudes and corresponding source names.
        """
        Magnitudes = np.linspace(self.setup.m_min_min, self.setup.m_max_max, self.num_scenarios)

        probabilities = {
            source_name: self.magnitude_frequency_distribution(Magnitudes, self.setup.source_m_min[source_name], self.setup.source_m_max[source_name])[0]
            for source_name in self.setup.source_data
        }
    
        prob_of_magnitudes = np.sum([self.setup.source_nu[source_name] * probabilities[source_name] for source_name in self.setup.source_data], axis=0)
        prob_of_magnitudes /= np.sum(prob_of_magnitudes)
    
        magnitudes = np.random.choice(Magnitudes, size=self.num_scenarios, p=prob_of_magnitudes)
    
        prob_of_magnitudes_given_source = {
            source_name: self.magnitude_frequency_distribution(magnitudes, self.setup.source_m_min[source_name], self.setup.source_m_max[source_name])[0]
            for source_name in self.setup.source_data
        }

        prob_of_sources_given_magnitude = np.transpose([
            self.setup.source_nu[source_name] * prob_of_magnitudes_given_source[source_name] / 
            sum(self.setup.source_nu[source_name] * prob_of_magnitudes_given_source[source_name] for source_name in self.setup.source_data)
            for source_name in self.setup.source_data
        ])
    
        source = [list(self.setup.source_data.keys())[np.random.choice(range(self.setup.num_sources), p=prob)] 
                 for prob in prob_of_sources_given_magnitude]
    
        return magnitudes, source

    def generate_inter_event_residuals(self):
        """
        Generates a set of normalized inter-event residuals for each site.
    
        Returns:
        --------
        dict: A dictionary with (site_id, scenario_index) as keys and inter-event residuals as values.
        """
        median_ground_motion, intra_event_std, inter_event_std = self.calculate_ground_motion_parameters()
        
        inter_event_residual = {}     
        for j, site_id in enumerate(self.setup.site_id):
            # Simulate Eta_1 from a univariate normal distribution with zero mean and unit std.
            Eta_1 = np.random.normal(0, 1, self.num_scenarios)
            
            for i in range(self.num_scenarios):
                # Calculate normalized inter-event residual for the ith scenario at jth site.
                inter_event_residual[site_id, i] = Eta_1[i] * (inter_event_std[site_id, 0] / inter_event_std[site_id, i]) 
        
        return inter_event_residual

    def generate_intra_event_residuals(self):
        """
        Generates a set of intra-event residuals using a spatial correlation model.

        Returns:
        --------
        dict: A dictionary with site IDs as keys and intra-event residuals as values.
        """
        def determine_range_parameter_by_ground_motion_type(vs30, ground_motion_type, period):
            """
            Determine the range parameter (b) for the correlation model given Vs30 values, 
            the ground motion type (PGA, PGV, or SA), and period if applicable.
        
            Parameters:
            vs30 (pd.Series): A Pandas series containing Vs30 values at different locations.
            ground_motion_type (str): Type of ground motion ('PGA', 'PGV', or 'SA').
            period (float, optional): The period of interest in seconds for 'SA'.
        
            Returns:
            float: The range parameter b for the correlation model.
            """            
            std_vs30 = vs30.std()
            clustering_threshold = 50
        
            if ground_motion_type == 'PGA' or ground_motion_type == 'SA':
                if period is None and ground_motion_type == 'SA':
                    raise ValueError("Period must be provided for Spectral Acceleration (SA).")
                period = 0 if ground_motion_type == 'PGA' else period

                case = 1 if std_vs30 > clustering_threshold else 2

                if period < 1:
                    if case == 1:
                        b = 8.5 + 17.2 * period
                    else:
                        b = 40.7 - 15.0 * period
                else:
                    b = 22.0 + 3.7 * period
        
            elif ground_motion_type == 'PGV':
                b = 83.4
        
            else:
                raise ValueError("Invalid ground motion type. Choose 'PGA', 'PGV', or 'SA'.")
                
            return b

        b = determine_range_parameter_by_ground_motion_type(self.setup.site_vs30, self.ground_motion_type, self.period)

        corr_matrix = np.zeros((self.setup.num_sites, self.setup.num_sites))
        intra_event_residual = {}

        for i in range(self.num_scenarios):
            for j, site_id1 in enumerate(self.setup.site_id):
                for k, site_id2 in enumerate(self.setup.site_id[j:]):
                    dist = SeismicDistanceCalculator.calculate_haversine_distance(self.setup.site_lat[j], self.setup.site_lon[j], self.setup.site_lat[j + k], self.setup.site_lon[j + k])
                    corr = np.exp(-3*dist/b)
                    corr_matrix[j, j + k] = corr_matrix[j + k, j] = corr

            epsilon = np.random.multivariate_normal(np.zeros(self.setup.num_sites), corr_matrix)
            for j, site_id in enumerate(self.setup.site_id):
                intra_event_residual[site_id, i] = epsilon[j]
    
        return intra_event_residual

    def calculate_ground_motion_parameters(self):
        """
        Calculates the median ground motion parameters and the associated standard deviations.

        Returns:
        --------
        tuple: A tuple containing dictionaries for median ground motion, intra-event std, and inter-event std.
        """
        def _get_gm_indices(model_class, ground_motion_type):
            if ground_motion_type == 'PGA':
                return model_class.INDEX_PGA
            elif ground_motion_type == 'PGV':
                return model_class.INDEX_PGV
            elif ground_motion_type == 'SA':
                return model_class.INDICES_PSA
            else:
                raise ValueError("Invalid ground motion type. Choose 'PGA', 'PGV', or 'SA'.")

        magnitudes, source = self.generate_scenarios()
        
        median_ground_motion, intra_event_std, inter_event_std = {}, {}, {} 
        
        for j, site_id in enumerate(self.setup.site_id):
            for i, M in enumerate(magnitudes):
                source_info   = self.setup.source_data[source[i]]
    
                site_loc   = (self.setup.site_lat[j], self.setup.site_lon[j], self.setup.site_depth[j])
                source_loc = (source_info['lat'], source_info['lon'], source_info['depth'])
    
                rupture_distance      = SeismicDistanceCalculator.calculate_rupture_distance(site_loc, source_loc, source_info['strike'], source_info['dip'])
                joyner_boore_distance = SeismicDistanceCalculator.calculate_joyner_boore_distance(site_loc, source_loc)
                horizontal_distance   = SeismicDistanceCalculator.calculate_horizontal_distance(site_loc, source_loc, source_info['strike'])

                model_class = getattr(pygmm, self.ground_motion_equation_model)
                gm_index    = _get_gm_indices(model_class, self.ground_motion_type)
                
                scenario = pygmm.Scenario(mag=M, dist_rup=rupture_distance, dist_jb=joyner_boore_distance, dist_x=horizontal_distance, site_cond=self.setup.site_condition[j], v_s30=self.setup.site_vs30[j] , dip=source_info['dip'], mechanism=source_info['mechanism'])
                model    = model_class(scenario)
    
                if self.ground_motion_type == 'PGA':
                    median_ground_motion[site_id, i] = model.pga
                elif self.ground_motion_type == 'PGV':
                    median_ground_motion[site_id, i] = model.pgv
                elif self.ground_motion_type == 'SA':
                    median_ground_motion[site_id, i] = model.spec_accels
                else:
                    raise ValueError("Invalid ground motion type. Choose 'PGA', 'PGV', or 'SA'.")
                    
                intra_event_std[site_id, i] = model._calc_ln_std()[2][gm_index]
                inter_event_std[site_id, i] = model._calc_ln_std()[1][gm_index]
                
        return median_ground_motion, intra_event_std, inter_event_std

    def generate_earthquake_objects(self):
        """
        Generates a dictionary of earthquake objects with calculated intensity measures.

        Returns:
        --------
        dict: A dictionary with tuples of (site_id, scenario_index) as keys and intensity measure values as values.
        """
        magnitudes, sources = self.generate_scenarios()
        median_ground_motion, intra_event_std, inter_event_std = self.calculate_ground_motion_parameters()
        inter_event_residual = self.generate_inter_event_residuals()
        intra_event_residual = self.generate_intra_event_residuals()

        intensity_measures = {}
        for i, M in enumerate(magnitudes):
            for j, site_id in enumerate(self.setup.site_id):
                im_value = np.exp(np.log(median_ground_motion[site_id, i]) +
                                  intra_event_std[site_id, i] * intra_event_residual[site_id, i] +
                                  inter_event_std[site_id, i] * inter_event_residual[site_id, i])
                intensity_measures[(site_id, i)] = im_value

        return intensity_measures
