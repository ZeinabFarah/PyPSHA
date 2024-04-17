import pandas as pd
import json

class InitialSetup:
    def __init__(self, csv_file_path, json_file_path):   
        """
        Initializes the InitialSetup with paths to site and source data files.
        
        Parameters:
        -----------
        csv_file_path : str
            The file path to the CSV file containing site data.
        json_file_path : str
            The file path to the JSON file containing source data.
        
        Attributes:
        -----------
        site_data : pandas.DataFrame
            A dataframe containing the processed site data.
        num_sites : int
            The number of site locations.
        source_data : dict
            A dictionary containing the processed source data.
        num_sources : int
            The number of earthquake sources.
        """
        self.site_data, self.num_sites = self.read_site_data(csv_file_path)
        self.source_data, self.num_sources = self.read_source_data(json_file_path)
        self.extract_source_data()
        self.extract_site_data()

    def read_site_data(self, file_path):
        """
        Reads a CSV file and processes it to ensure it has the required columns.
        Also calculates the number of sites (rows) in the data.

        Parameters:
        -----------
        file_path (str): 
            Path to the CSV file.

        Returns:
        -----------
        site_data (pandas dataframe): 
            A data frame with the processed data.
        num_sites (int): 
            Number of sites (rows) in the DataFrame.
        """
        # Read the CSV file
        try:
            site_data = pd.read_csv(file_path)
        except Exception as e:
            raise IOError(f"Error reading the file: {e}")

        # Validate the columns
        required_columns = ['id', 'latitude', 'longitude', 'depth', 'vs30', 'condition']
        if not all(column in site_data.columns for column in required_columns):
            raise ValueError(f"The CSV file must contain the following columns: {required_columns}")

        # Calculate the number of sites
        num_sites = site_data.shape[0]

        return site_data, num_sites

    def read_source_data(self, file_path):
        """
        Reads earthquake source characteristics from a JSON file.
    
        Parameters:
        -----------
        file_path (str): 
            Path to the JSON file.
    
        Returns:
        -----------
        source_data (dict): 
            Dictionary containing earthquake source data.
        num_sources (int):
            Number of earthquake sources in the data.
        """
        try:
            with open(file_path, 'r') as file:
                source_data = json.load(file)
            
            num_sources = len(source_data)
            return source_data, num_sources
    
        except Exception as e:
            raise IOError(f"Error reading the file: {e}")

    def extract_source_data(self):
        """Extracts parameters from each source in source_data."""
        self.source_m_min = {name: data["M_min"] for name, data in self.source_data.items()}
        self.source_m_max = {name: data["M_max"] for name, data in self.source_data.items()}
        self.source_nu = {name: data["nu"] for name, data in self.source_data.items()}
        self.source_lat = {name: data["lat"] for name, data in self.source_data.items()}
        self.source_lon = {name: data["lon"] for name, data in self.source_data.items()}
        self.source_depth = {name: data["depth"] for name, data in self.source_data.items()}
        self.source_strike = {name: data["strike"] for name, data in self.source_data.items()}
        self.source_dip = {name: data["dip"] for name, data in self.source_data.items()}
        self.source_mechanism = {name: data["mechanism"] for name, data in self.source_data.items()}
        self.source_event_type = {name: data["event_type"] for name, data in self.source_data.items()}
    
        # Determining the overall min and max magnitudes
        self.m_min_min = max(self.source_m_min.values())
        self.m_max_max = min(self.source_m_max.values())

    def extract_site_data(self):
        """Extracts parameters from each site in site_data"""
        self.site_id  = self.site_data['id']
        self.site_lat = self.site_data['latitude']
        self.site_lon = self.site_data['longitude']
        self.site_depth = self.site_data['depth']
        self.site_vs30 = self.site_data['vs30']
        self.site_condition = self.site_data['condition']




