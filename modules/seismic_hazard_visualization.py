import matplotlib.pyplot as plt
import numpy as np
import folium
from folium.plugins import HeatMap
from scipy.interpolate import griddata

class SeismicHazardVisualization:
    def __init__(self, setup, intensity_measures, return_period):
        """
        Initializes the SeismicHazardVisualization with necessary parameters for visualization.

        Parameters:
        -----------
        setup : object
            An instance of setup class containing site and source data.
        intensity_measures : dict
            A dictionary of intensity measures for each site and scenario.
        return_period : int
            The return period for seismic hazard analysis in years.
        """
        self.setup = setup
        self.intensity_measures = intensity_measures
        self.return_period = return_period

    def prepare_data(self, exceedance_probability):
        """
        Prepare the data for plotting based on the specified exceedance probability.

        Parameters:
        -----------
        exceedance_probability : float
            The exceedance probability for which to prepare the data.

        Returns:
        --------
        list
            A list of tuples containing latitude, longitude, and exceedance value for each site.
        """
        processed_data = []
        for site_id in self.setup.site_data['id']:
            site_im_values = [im for (id, _), im in self.intensity_measures.items() if id == site_id]        
            exceedance_value = np.percentile(site_im_values, 100 * (1 - exceedance_probability))
            try:
                row = self.setup.site_data[self.setup.site_data['id'] == site_id].iloc[0]
                lat, lon = row['latitude'], row['longitude']
            except IndexError:
                print(f"Site ID {site_id} not found in site_data")

            processed_data.append((lat, lon, exceedance_value))
        return processed_data

    def plot_hazard_map(self, exceedance_probability):
        """
        Plot the seismic hazard map with site locations and exceedance probability indicated.

        Parameters:
        -----------
        exceedance_probability : float
            The probability of exceedance to be visualized on the hazard map.
        """
        data = self.prepare_data(exceedance_probability)
        lats, lons, ims = zip(*data)

        plt.figure(figsize=(10, 8))
        plt.scatter(lons, lats, c=ims, cmap='Reds', marker='o', label='Site Locations')
        plt.colorbar(label='Intensity Measure at Exceedance Probability')

        # Plotting source locations
        source_lats, source_lons = zip(*[(source['lat'], source['lon']) for source in self.setup.source_data.values()])
        plt.scatter(source_lons, source_lats, c='yellow', marker='*', s=300, edgecolors='black', label='Source Locations')

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Seismic Hazard Map - {self.return_period} years, {exceedance_probability*100}% Exceedance')
        plt.legend()
        plt.show()
       
    def plot_heatmap(self, exceedance_probability):
        """
        Plot a heatmap representing seismic hazard based on the exceedance probability.

        Parameters:
        -----------
        exceedance_probability : float
            The exceedance probability for which to generate the heatmap.
        """
        data = self.prepare_data(exceedance_probability)

        # Create a map centered around the mean latitude and longitude
        mean_lat = np.mean([lat for lat, _, _ in data])
        mean_lon = np.mean([lon for _, lon, _ in data])
        map_obj  = folium.Map(location=[mean_lat, mean_lon], zoom_start=5)

        # Add HeatMap layer
        heat_data = [[lat, lon, im] for lat, lon, im in data]
        HeatMap(heat_data).add_to(map_obj)

        return map_obj

    def plot_contour_map(self, exceedance_probability):
        """
        Plot a contour map representing seismic hazard based on the exceedance probability.

        Parameters:
        -----------
        exceedance_probability : float
            The exceedance probability for which to generate the contour map.
        """
        # Prepare the data
        data = self.prepare_data(exceedance_probability)
        lats, lons, ims = zip(*data)

        # Generate a grid to interpolate onto
        grid_lons, grid_lats = np.meshgrid(np.linspace(min(lons), max(lons), 100),
                                           np.linspace(min(lats), max(lats), 100))

        # Interpolate the data onto the grid
        grid_ims = griddata((lons, lats), ims, (grid_lons, grid_lats), method='cubic')

        # Create the contour plot
        plt.figure(figsize=(10, 6))
        contour = plt.contourf(grid_lons, grid_lats, grid_ims, levels=100, cmap='viridis')
        plt.colorbar(contour)

        for source_id, source_info in self.setup.source_data.items():
            plt.plot(source_info['lon'], source_info['lat'], '*', color='yellow', markersize=15, label='Source Locations')

        # Annotations and titles
        plt.title(f"Seismic Hazard Contour Map - {self.return_period} years, {exceedance_probability*100}% Exceedance")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.show()


