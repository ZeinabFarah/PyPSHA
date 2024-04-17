
import matplotlib.pyplot as plt
import numpy as np
import folium
from folium.plugins import HeatMap
from scipy.interpolate import griddata
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

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

        annual_exceedance_probability = 1 - (1 - exceedance_probability)**(1/self.return_period)

        processed_data = []
        for site_id in self.setup.site_data['id']:
            site_im_values = [im for (id, *rest), im in self.intensity_measures.items() if id == site_id]
            exceedance_value = np.percentile(site_im_values, 100 * (1 - annual_exceedance_probability))
            try:
                row = self.setup.site_data[self.setup.site_data['id'] == site_id].iloc[0]
                lat, lon = row['latitude'], row['longitude']
            except IndexError:
                print(f"Site ID {site_id} not found in site_data")

            processed_data.append((lat, lon, exceedance_value))
        return processed_data

    def generate_earthquake_geotiff(self, exceedance_probability, geotiff_file_path):
        """
        Prepares and returns geographic and exceedance data for sites above a specified probability threshold.
        Each site's data includes its latitude, longitude, and the exceedance value that meets or exceeds
        the given threshold, useful for spatial risk visualization.

        Parameters:
        -----------
        exceedance_probability : float
            The probability threshold for selecting sites, expressed as a decimal (e.g., 0.1 for 10% chance).

        Returns:
        --------
        list of tuples
            A list where each tuple contains latitude, longitude, and exceedance value for each selected site.
        """
        data = self.prepare_data(exceedance_probability)
        lats, lons, ims = zip(*data)

        west = min(lons)
        east = max(lons)
        north = max(lats)
        south = min(lats)

        # Define grid resolution
        pixel_width  = 0.01
        pixel_height = 0.01

        # Calculate grid dimensions
        width = int((east - west) / pixel_width)
        height = int((north - south) / pixel_height)

        # Create the transformation for the GeoTIFF
        transform = from_origin(west, north, pixel_width, -pixel_height)

        # Initialize the grid
        intensity_grid = np.full((height, width), np.nan, dtype='float32')

        # Populate the grid with intensity measures
        for lat, lon, im in zip(lats, lons, ims):
            # Calculate row and col indices
            lat_index  = int((north - lat) / pixel_height)
            lon_index  = int((lon - west) / pixel_width)

            # Ensure the indices do not exceed the grid dimensions
            row = max(0, min(lat_index, height - 1))
            col = max(0, min(lon_index, width - 1))

            # Assign the intensity measure to the correct grid cell
            intensity_grid[row, col] = im

        # Write the grid to a GeoTIFF file
        with rasterio.open(
            f'{geotiff_file_path}/intensity_measures.tif',
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=intensity_grid.dtype,
            crs='+proj=latlong',
            transform=transform,
        ) as dst:
            dst.write(intensity_grid, 1)

    def generate_continuous_earthquake_geotiff(self, exceedance_probability, grid_resolution, geotiff_file_path):
        """
        Generates a continuous raster GeoTIFF representing earthquake exceedance values over an area,
        based on interpolation of exceedance data at specific sites.

        Parameters:
        -----------
        exceedance_probability : float
            The probability threshold for selecting sites, expressed as a decimal (e.g., 0.1 for 10% chance).
        grid_resolution : float, optional
            The resolution of the grid in degrees, determining the spacing of points in the mesh.
            Default is 0.01 degrees.

        Returns:
        --------
        None
            Creates a GeoTIFF file named 'continuous_intensity_measures.tif' with the interpolated raster data.
        """

        data = self.prepare_data(exceedance_probability)
        lats, lons, ims = zip(*data)

        # Create a mesh grid covering the entire area of interest
        xi = np.arange(min(lons), max(lons), grid_resolution)
        yi = np.arange(min(lats), max(lats), grid_resolution)
        X, Y = np.meshgrid(xi, yi)

        # Interpolate intensity measures onto the mesh grid
        Z = griddata((lons, lats), ims, (X, Y), method='cubic')

        # Define the transformation for the GeoTIFF
        west, north = X.min(), Y.max()
        pixel_width, pixel_height = grid_resolution, grid_resolution
        transform = from_origin(west, north, pixel_width, -pixel_height)

        # Write the interpolated grid to a GeoTIFF file
        with rasterio.open(
            f'{geotiff_file_path}/continuous_intensity_measures.tif',
            'w',
            driver='GTiff',
            height=Z.shape[0],
            width=Z.shape[1],
            count=1,
            dtype=Z.dtype,
            crs='+proj=latlong',
            transform=transform,
        ) as dst:
            dst.write(Z, 1)

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

    def plot_tract_intensity_map(self, exceedance_probability, census_file_path):
        """
        Plot a map representing seismic hazard with intensity measure values averaged for each census tract.

        Parameters:
        -----------
        exceedance_probability : float
            The exceedance probability for which to generate the contour map.
        census_file_path : str
            The path to the shapefile for North America.
        """
        # Prepare the intensity measure data
        data = self.prepare_data(exceedance_probability)
        lats, lons, ims = zip(*data)

        # Create a GeoDataFrame from the intensity measure data
        points_df = pd.DataFrame({'intensity_measure': ims, 'geometry': [Point(lon, lat) for lon, lat in zip(lons, lats)]})
        points_gdf = gpd.GeoDataFrame(points_df, geometry='geometry')

        # Read the shapefile and ensure points_gdf has the same CRS as the shapefile data
        tracts_gdf = gpd.read_file(census_file_path)
        points_gdf.crs = tracts_gdf.crs

        # Perform a spatial join between the points and the tracts
        tracts_with_ims = gpd.sjoin(tracts_gdf, points_gdf, how="inner", predicate='contains')

        # Compute the mean intensity measure for each tract
        tract_intensity = tracts_with_ims.groupby('TRACTCE')['intensity_measure'].mean().reset_index()

        # Merge this back with the original tracts data
        tracts_gdf = tracts_gdf.merge(tract_intensity, on='TRACTCE', how='left')

        # Plot the tracts colored by the mean intensity measure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        tracts_gdf.plot(column='intensity_measure', ax=ax, legend=True, cmap='viridis', missing_kwds={
            "color": "lightgrey",
            "edgecolor": "red",
            "hatch": "///",
            "label": "Missing values",
        })

        # Plot the seismic sources as yellow stars
        for source_info in self.setup.source_data.values():
            ax.plot(source_info['lon'], source_info['lat'], '*', color='yellow', markersize=15, label='Seismic Sources')

        # Annotations and titles
        plt.title(f"Seismic Hazard Map - {self.return_period} years, {exceedance_probability*100}% Exceedance Probability")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.show()
