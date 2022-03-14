##########################################################################
# ---------------------------------------------------------------------------------------------------------------------
# This is Python code to produce IPCC AR6 WGI Figures 2.12 (b)-(e)
# Creator: Florian Ladstädter, Wegener Center, University of Graz, Austria
# Contact: florian.ladstaedter@uni-graz.at
# Last updated on: July 9th, 2021
# --------------------------------------------------------------------------------------------------------------------
#
# - Code functionality: atmoplots -- This is a Python package, installable using `pip install .` in the root directory, and provides the command `atmoplots`. It reads vertically resolved temperature trend data from various sources (in netCDF format), and plots them on a common altitude grid. Data provided on a pressure grid are translated to the common altitude grid via an empirical pressure-to-altitude mapping, derived from GPS Radio Occultation data. Usage is described in the README. Other code provided is subsidiary to atmoplots. Input data is not part of the package. 
# - Input data: Input data in netCDF format for each dataset is needed (RO-ROMSAF, RO-UCAR, RO-WEGC, ERA5.1, AIRS, RAOBCORE, RICH) for the two time ranges (2002-2019, 1980-2019, where available) and the two latitude ranges (20S-20N, 70S-70N). The data are available from the author upon request and have also been archived through the IPCC data repository. 
# - Output variables: The code plots the figures 2.12 (b)-(e) as in the report, depending on the parameters given. This is detailed in the README.
#
# ----------------------------------------------------------------------------------------------------
# Information on  the software used
# - Software Version: Python 3.7 and Python packages as detailed in the provided requirements.txt.
# - Landing page to access the software: 
# - Operating System: Debian 10
# - Environment required to compile and run: Linux
#  ----------------------------------------------------------------------------------------------------
#
#  License: Creative Commons Attribution 4.0 International License (http://creativecommons.org/licenses/by/4.0/)
#
# ----------------------------------------------------------------------------------------------------
# How to cite: https://doi.org/10.5281/zenodo.6353809
# When citing this code, please include both the code citation and the following citation for the related report component:
########################################################################














Am keeping this for reference:
# ----------------------------------------------------------------------------------------------
# Acknowledgement: The template for this file was created by Lina E. Sitz (https://orcid.org/0000-0002-6333-4986), Paula A. Martinez (https://orcid.org/0000-0002-8990-1985), and J. B. Robin Matthews (https://orcid.org//0000-0002-6016-7596)
