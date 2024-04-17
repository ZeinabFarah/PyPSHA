
from modules.seismic_distance_calculator import SeismicDistanceCalculator
from modules.magnitude_frequency_distribution import MagnitudeFrequencyDistribution
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
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
            Name of the magnitude frequency distribution function to use.
        ground_motion_equation_model : str
            Name of the ground motion equation model to use.
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
        # Generate linearly spaced magnitudes within the specified range.
        Magnitudes = np.linspace(self.setup.m_min_min, self.setup.m_max_max, self.num_scenarios)

        # Calculate the probability of each magnitude for each source.
        probabilities = {
            source_name: self.magnitude_frequency_distribution(Magnitudes, self.setup.source_m_min[source_name], self.setup.source_m_max[source_name])[0]
            for source_name in self.setup.source_data
        }

        # Sum the probabilities across all sources, weighted by the annual rate of exceedance.
        prob_of_magnitudes = np.sum([self.setup.source_nu[source_name] * probabilities[source_name] for source_name in self.setup.source_data], axis=0)
        # Normalize the probabilities to sum to 1.
        prob_of_magnitudes /= np.sum(prob_of_magnitudes)

        # Select magnitudes based on the calculated probabilities.
        magnitudes = np.random.choice(Magnitudes, size=self.num_scenarios, p=prob_of_magnitudes)

        # Calculate the probability of each source given the selected magnitudes.
        prob_of_magnitudes_given_source = {
            source_name: self.magnitude_frequency_distribution(magnitudes, self.setup.source_m_min[source_name], self.setup.source_m_max[source_name])[0]
            for source_name in self.setup.source_data
        }

        # Normalize the probabilities of sources given the magnitude.
        prob_of_sources_given_magnitude = np.transpose([
            self.setup.source_nu[source_name] * prob_of_magnitudes_given_source[source_name] /
            sum(self.setup.source_nu[source_name] * prob_of_magnitudes_given_source[source_name] for source_name in self.setup.source_data)
            for source_name in self.setup.source_data
        ])

        # Select sources for each scenario based on the calculated probabilities.
        sources = [list(self.setup.source_data.keys())[np.random.choice(range(self.setup.num_sources), p=prob)]
                 for prob in prob_of_sources_given_magnitude]

        return magnitudes, sources

    def calculate_ground_motion_parameters(self, magnitudes, sources):
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

        median_ground_motion, intra_event_std, inter_event_std = {}, {}, {}

        for i, magnitude in enumerate(magnitudes):
          source_info = self.setup.source_data[sources[i]]
          source_loc = (source_info['lat'], source_info['lon'], source_info['depth'])

          for j, site_id in enumerate(self.setup.site_id):
            site_loc   = (self.setup.site_lat[j], self.setup.site_lon[j], self.setup.site_depth[j])

            rupture_distance      = SeismicDistanceCalculator.calculate_rupture_distance(site_loc, source_loc, source_info['strike'], source_info['dip'])
            joyner_boore_distance = SeismicDistanceCalculator.calculate_joyner_boore_distance(site_loc, source_loc)
            horizontal_distance   = SeismicDistanceCalculator.calculate_horizontal_distance(site_loc, source_loc, source_info['strike'])

            model_class = getattr(pygmm, self.ground_motion_equation_model)
            gm_index    = _get_gm_indices(model_class, self.ground_motion_type)

            scenario = pygmm.Scenario(mag=magnitude, dist_rup=rupture_distance, dist_jb=joyner_boore_distance, dist_x=horizontal_distance, site_cond=self.setup.site_condition[j], v_s30=self.setup.site_vs30[j] , dip=source_info['dip'], mechanism=source_info['mechanism'], event_type=source_info['event_type'])
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

            # v_ref     = 1180 # Consider refernce velocity as 1180 m/s
            # resp_ref  = np.exp(model._calc_ln_resp(v_ref, np.nan))

            # intra_event_std[site_id, i] = model._calc_ln_std(resp_ref)[2][gm_index]
            # inter_event_std[site_id, i] = model._calc_ln_std(resp_ref)[1][gm_index]

        return median_ground_motion, intra_event_std, inter_event_std

    def generate_norm_inter_event_residuals(self, inter_event_std):
        """
        Generates a set of normalized inter-event residuals for each scenario.

        Parameters:
        -----------
        inter_event_std : dict
            A dictionary with (site_id, scenario_index) as keys and standard deviation of inter-event residuals as values.

        Returns:
        --------
        dict: A dictionary with (site_id, scenario_index) as keys and normalized inter-event residuals as values.
        """

        norm_inter_event_residuals = {}
        # Generate normalized inter-event residuals for each scenario.
        for i in range(self.num_scenarios):
            Eta = np.random.normal(0, 1)
            # Base standard deviation for normalization
            base_inter_event_std = inter_event_std[self.setup.site_id[0], i]  
            for j, site_id in enumerate(self.setup.site_id):
                norm_inter_event_residuals[site_id, i] = (base_inter_event_std/inter_event_std[site_id, i]) * Eta

        return norm_inter_event_residuals

    def generate_norm_intra_event_residuals(self):
        """
        Generates a set of intra-event residuals using a spatial correlation model.

        Returns:
        --------
        dict: A dictionary with (site_id, scenario_index) as keys and intra-event residuals as values.
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

        # Range parameter b based on ground motion type.
        b = determine_range_parameter_by_ground_motion_type(self.setup.site_vs30, self.ground_motion_type, self.period)

        # Coordinate matrix for sites.
        coords = np.column_stack((self.setup.site_lat, self.setup.site_lon))

        norm_intra_event_residuals = {}
        # Generate residuals for each scenario.
        for i in range(self.num_scenarios):
            # Calculate the pairwise Haversine distance matrix for the current scenario's source location.
            distances = pdist(coords, lambda u, v: SeismicDistanceCalculator.calculate_haversine_distance(u[0], u[1], v[0], v[1]))

            # Convert distance matrix to a square-form and calculate the correlation matrix.
            corr_matrix = np.exp(-3 * squareform(distances) / b)

            # Set the diagonal to 1 to ensure positive definiteness.
            np.fill_diagonal(corr_matrix, 1)

            # Generate the intra-event residuals for this scenario using the correlation matrix.
            epsilon = np.random.multivariate_normal(np.zeros(self.setup.num_sites), corr_matrix)

            # Assign residuals to each site for the current scenario.
            for j, site_id in enumerate(self.setup.site_id):
                norm_intra_event_residuals[site_id, i] = epsilon[j]

        return norm_intra_event_residuals

    def generate_earthquake_objects(self):
        """
        Generates a dictionary of earthquake objects with calculated intensity measures.

        Returns:
        --------
        dict: A dictionary with tuples of (site_id, scenario_index) as keys and intensity measure values as values.
        """
        magnitudes, sources  = self.generate_scenarios()
        median_ground_motion, intra_event_std, inter_event_std = self.calculate_ground_motion_parameters(magnitudes, sources)
        norm_inter_event_residual = self.generate_norm_inter_event_residuals(inter_event_std)
        norm_intra_event_residual = self.generate_norm_intra_event_residuals()

        intensity_measure_dict = {}
        intensity_measure_data = []
        for i, magnitude in enumerate(magnitudes):
            source_info = self.setup.source_data[sources[i]]
            for j, site_id in enumerate(self.setup.site_id):
                im_value = np.exp(np.log(median_ground_motion[site_id, i]) +
                                  intra_event_std[site_id, i] * norm_intra_event_residual[site_id, i] +
                                  inter_event_std[site_id, i] * norm_inter_event_residual[site_id, i])

                distance = SeismicDistanceCalculator.calculate_haversine_distance(self.setup.site_lat[j], self.setup.site_lon[j],  source_info['lat'], source_info['lon'])

                intensity_measure_dict[(site_id, i, magnitude, distance)] = im_value
                intensity_measure_data.append([site_id, i, magnitude, distance, im_value])

        # Create a DataFrame from the intensity measures
        intensity_measure_df = pd.DataFrame(intensity_measure_data, columns=['site_id', 'scenarion_index', 'magnitude', 'distance', 'intensity_measure'])

        return intensity_measure_dict, intensity_measure_df
