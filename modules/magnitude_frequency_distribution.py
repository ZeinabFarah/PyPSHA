import numpy as np

class MagnitudeFrequencyDistribution:
    
    @staticmethod
    def gr_recurrence_law(Magnitudes, m_min, m_max):
        """
        Computes the probability of earthquakes in a particular region using the Gutenberg-Richter recurrence law
    
        Parameters:
        -----------
        Magnitudes : float
            An array of magnitude values.   
        m_min : float
            The minimum magnitude of earthquakes considered in the analysis.
        m_max : float
            The maximum magnitude of earthquakes considered in the analysis.          
    
        Returns:
        --------
        probability : numpy.ndarray
            An array of corresponding probability for each magnitude value in Magnitudes.
        pdf : numpy.ndarray
            An array of corresponding pdf for each magnitude value in Magnitudes.
        """
        b_value     = 1  # The b-value of the Gutenberg-Richter recurrence law describes the relationship between the magnitude and frequency of earthquakes.
        probability = 10 ** (-b_value * (Magnitudes - m_min))
        pdf         = (b_value*np.log(10)*10**(-b_value*(Magnitudes-m_min)))/(1-10**(-b_value*(m_max-m_min)))

        return probability , pdf
    
   