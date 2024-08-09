
from modules.seismic_distance_calculator import SeismicDistanceCalculator
from modules.magnitude_frequency_distribution import MagnitudeFrequencyDistribution
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import pygmm

class EarthquakeGenerator:
    def __init__(self, setup, magnitude_frequency_distribution, ground_motion_equation_model):
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
        """
        self.setup = setup
        self.magnitude_frequency_distribution = getattr(MagnitudeFrequencyDistribution, magnitude_frequency_distribution)
        self.ground_motion_equation_model = ground_motion_equation_model

    def generate_scenarios(self, num_scenarios):
        """
        Generates earthquake scenarios with associated magnitudes and sources.

        Parameters:
        -----------
        num_scenarios : int
            Number of earthquake scenarios to generate.

        Returns:
        --------
        tuple: A tuple containing arrays of magnitudes and corresponding source names.
        """
        # Generate magnitudes for each source within their specified range.
        source_magnitudes = {
            source_name: np.linspace(self.setup.source_m_min[source_name], self.setup.source_m_max[source_name], num_scenarios)
            for source_name in self.setup.source_data
        }

        # Flatten the source magnitudes into a single array
        all_magnitudes = np.concatenate([source_magnitudes[source_name] for source_name in self.setup.source_data])

        # Calculate the probability of each magnitude for each source.
        probabilities = {
            source_name: self.magnitude_frequency_distribution(source_magnitudes[source_name], self.setup.source_m_min[source_name], self.setup.source_m_max[source_name])[0]
            for source_name in self.setup.source_data
        }

        # Sum the probabilities across all sources, weighted by the annual rate of exceedance.
        prob_of_magnitudes = np.sum([self.setup.source_nu[source_name] * probabilities[source_name] for source_name in self.setup.source_data], axis=0)

        # Normalize the probabilities to sum to 1.
        prob_of_magnitudes /= np.sum(prob_of_magnitudes)

        # Expand probabilities to match the size of the concatenated magnitudes array
        expanded_prob_of_magnitudes = np.tile(prob_of_magnitudes, len(self.setup.source_data))

        # Normalize the expanded probabilities to sum to 1.
        expanded_prob_of_magnitudes /= np.sum(expanded_prob_of_magnitudes)

        # Select magnitudes based on the calculated probabilities.
        magnitudes = np.random.choice(all_magnitudes, size=num_scenarios, p=expanded_prob_of_magnitudes)

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

    def calculate_ground_motion_parameters(self, ground_motion_type, magnitudes, sources):
        """
        Calculates the median ground motion parameters and the associated standard deviations.

        Parameters:
        -----------
        ground_motion_type : str
            Type of ground motion to analyze ('PGA', 'PGV', or 'SA').

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
            source_loc = (source_info['lon'], source_info['lat'], source_info['depth'])

            for j, site_id in enumerate(self.setup.site_id):
                site_loc = (self.setup.site_lon[j], self.setup.site_lat[j], self.setup.site_depth[j])

                # Calculate distances
                rupture_distance = SeismicDistanceCalculator.calculate_rupture_distance(site_loc, source_loc, source_info['strike'], source_info['dip'])
                joyner_boore_distance = SeismicDistanceCalculator.calculate_joyner_boore_distance(site_loc, source_loc)
                horizontal_distance = SeismicDistanceCalculator.calculate_horizontal_distance(site_loc, source_loc, source_info['strike'])
                haversine_distance = SeismicDistanceCalculator.calculate_haversine_distance(self.setup.site_lat[j], self.setup.site_lon[j], source_info['lat'], source_info['lon'])

                # Ground motion model
                model_class = getattr(pygmm, self.ground_motion_equation_model)
                gm_index = _get_gm_indices(model_class, ground_motion_type)
                scenario = pygmm.Scenario(
                    mag=magnitude,
                    dist_rup=rupture_distance,
                    dist_jb=joyner_boore_distance,
                    dist_x=horizontal_distance,
                    site_cond=self.setup.site_condition[j],
                    v_s30=self.setup.site_vs30[j],
                    dip=source_info['dip'],
                    mechanism=source_info['mechanism'],
                    event_type=source_info['event_type']
                )
                model = model_class(scenario)

                resp_ref = np.exp(model._calc_ln_resp(model.V_REF, np.nan))
                ln_std, tau, phi = model._calc_ln_std(resp_ref)

                median_ground_motion[site_id, i] = np.exp(model._calc_ln_resp(scenario.v_s30, resp_ref)[gm_index])
                inter_event_std[site_id, i] = tau[gm_index]
                intra_event_std[site_id, i] = phi[gm_index]

        return median_ground_motion, intra_event_std, inter_event_std

    def generate_norm_inter_event_residuals(self, num_scenarios, inter_event_std):
        """
        Generates a set of normalized inter-event residuals for each scenario.

        Parameters:
        -----------
        num_scenarios : int
            Number of earthquake scenarios to generate.
        inter_event_std : dict
            A dictionary with (site_id, scenario_index) as keys and standard deviation of inter-event residuals as values.

        Returns:
        --------
        dict: A dictionary with (site_id, scenario_index) as keys and normalized inter-event residuals as values.
        """

        norm_inter_event_residuals = {}
        # Generate normalized inter-event residuals for each scenario.
        for i in range(num_scenarios):
            Eta = np.random.normal(0, 1)
            for j, site_id in enumerate(self.setup.site_id):
                # Base standard deviation for normalization
                base_inter_event_std = inter_event_std[self.setup.site_id[0], i]
                norm_inter_event_residuals[site_id, i] = (base_inter_event_std/inter_event_std[site_id, i]) * Eta

        return norm_inter_event_residuals

    def generate_norm_intra_event_residuals(self, num_scenarios, ground_motion_type, sources, period=None):
        """
        Generates a set of intra-event residuals using a spatial correlation model.

        Returns:
        --------
        dict: A dictionary with (site_id, scenario_index) as keys and intra-event residuals as values.
        """

        def determine_range_parameter_by_ground_motion_type(vs30, ground_motion_type, period=None):
            """
            Determine the range parameter (b) for the correlation model given Vs30 values,
            the ground motion type (PGA, PGV, or SA), and period if applicable.

            Parameters:
            -----------
            vs30 : (pd.Series)
                A Pandas series containing Vs30 values at different locations.
            ground_motion_type : str
                Type of ground motion to analyze ('PGA', 'PGV', or 'SA').
            period : float, optional
                The period of interest in seconds for spectral acceleration (SA).

            Returns:
            --------
            float: The range parameter b for the correlation model.
            """
            if isinstance(vs30, (list, np.ndarray, pd.Series)) and len(vs30) > 1:
                std_vs30 = np.std(vs30)
            else:
                std_vs30 = 0

            clustering_threshold = 50

            if ground_motion_type == 'PGA':
                period = 0
                case = 1 if std_vs30 > clustering_threshold else 2

                if case == 1:
                    b = 8.5 + 17.2 * period
                else:
                    b = 40.7 - 15.0 * period

            elif ground_motion_type == 'PGV':
                period = 1
                b = 25.7

            elif ground_motion_type == 'SA':
                if period is None:
                    raise ValueError("Period must be provided for Spectral Acceleration (SA).")

                case = 1 if std_vs30 > clustering_threshold else 2

                if period < 1:
                    if case == 1:
                        b = 8.5 + 17.2 * period
                    else:
                        b = 40.7 - 15.0 * period
                else:
                    b = 22.0 + 3.7 * period

            else:
                raise ValueError("Invalid ground motion type. Choose 'PGA', 'PGV', or 'SA'.")
            return b

        # Range parameter b based on ground motion type.
        b = determine_range_parameter_by_ground_motion_type(self.setup.site_vs30, ground_motion_type, period)

        # Coordinate matrix for sites.
        coords = np.column_stack((self.setup.site_lat, self.setup.site_lon))

        # Calculate the pairwise Haversine distance matrix for the current scenario's source location.
        distances = pdist(coords, lambda u, v: SeismicDistanceCalculator.calculate_haversine_distance(u[0], u[1], v[0], v[1]))

        # Convert distance matrix to a square-form and calculate the correlation matrix.
        corr_matrix = np.exp(-3 * squareform(distances) / b)

        # Set the diagonal to 1 to ensure positive definiteness.
        np.fill_diagonal(corr_matrix, 1)

        norm_intra_event_residuals = {}
        # Generate residuals for each scenario.
        for i in range(num_scenarios):
            # Generate the intra-event residuals for this scenario using the correlation matrix.
            epsilon = np.random.multivariate_normal(np.zeros(self.setup.num_sites), corr_matrix)

            # Assign residuals to each site for the current scenario.
            for j, site_id in enumerate(self.setup.site_id):
                norm_intra_event_residuals[(site_id, i)] = epsilon[j]

        return norm_intra_event_residuals

    def generate_earthquake_objects(self, num_scenarios):
        """
        Generates a dictionary of earthquake objects with calculated intensity measures for both PGA and PGV,
        and a DataFrame with separate columns for PGA and PGV.

        Parameters:
        -----------
        num_scenarios : int
            Number of earthquake scenarios to generate.

        Returns:
        --------
        tuple: A tuple containing a dictionary and a DataFrame. The dictionary and DataFrame have keys and columns for
               scenario index, site_id, source_id, magnitude, and distance, and separate values/columns for PGA and PGV intensity measures.
        """
        magnitudes, sources = self.generate_scenarios(num_scenarios)

        intensity_measure_dict = {}
        intensity_measure_data = []
        combined_data = {}

        for ground_motion_type in ['PGA', 'PGV']:
            median_ground_motion, intra_event_std, inter_event_std = self.calculate_ground_motion_parameters(
                ground_motion_type, magnitudes, sources)
            norm_inter_event_residual = self.generate_norm_inter_event_residuals(num_scenarios, inter_event_std)
            norm_intra_event_residual = self.generate_norm_intra_event_residuals(num_scenarios, ground_motion_type, sources)

            for i, magnitude in enumerate(magnitudes):
                source_info = self.setup.source_data[sources[i]]
                source_loc = (source_info['lon'], source_info['lat'], source_info['depth'])
                source_id = sources[i]
                for j, site_id in enumerate(self.setup.site_id):
                    site_loc = (self.setup.site_lon[j], self.setup.site_lat[j], self.setup.site_depth[j])
                    ln_im_value = np.log(median_ground_motion[site_id, i]) \
                    + inter_event_std[site_id, i] * norm_inter_event_residual[site_id, i] \
                    + intra_event_std[site_id, i] * norm_intra_event_residual[site_id, i]

                    distance = SeismicDistanceCalculator.calculate_haversine_distance(
                        self.setup.site_lat[j], self.setup.site_lon[j],
                        source_info['lat'], source_info['lon'])

                    key = (i, site_id, source_id, magnitude, distance)
                    if key not in combined_data:
                        combined_data[key] = {}
                    combined_data[key][ground_motion_type] = np.exp(ln_im_value)

        # Transform combined data into the final dictionary and DataFrame format
        for key, values in combined_data.items():
            scenario_index, site_id, source_id, magnitude, distance = key
            intensity_measure_dict[key] = values
            intensity_measure_data.append([
                scenario_index, site_id, source_id, magnitude, distance,
                values.get('PGA', None),
                values.get('PGV', None)
            ])

        # Create a DataFrame from the intensity measures
        intensity_measure_df = pd.DataFrame(intensity_measure_data, columns=[
            'scenario_index', 'site_id', 'source_id', 'magnitude', 'distance', 'PGA', 'PGV'])

        return intensity_measure_dict, intensity_measure_df
