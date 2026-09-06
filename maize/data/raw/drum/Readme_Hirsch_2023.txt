This readme.txt file was generated on 2023-02-04 by Dorothy Sweet

Recommended citation for the data: Sweet, Dorothy D; Tirado, Sara B; Cooper, Julian S; Springer, Nathan M; Hirsch, Cory D; Hirsch, Candice N. (2023). Temporally resolved growth patterns in diverse maize panel. Retrieved from the Data Repository for the University of Minnesota. https://conservancy.umn.edu/handle/11299/250609.

-------------------
GENERAL INFORMATION
-------------------

1. Title of Dataset: Temporally resolved growth patterns in diverse maize panel

2. Author Information

	Author Contact:  Candice N Hirsch (cnhirsch@umn.edu)

	Name:  Dorothy D Sweet
	Institution: University of Minnesota
	Email: kirsc168@umn.edu
	ORCID: 0000-0002-9614-5436


	Name:  Sara B Tirado
	Institution: University of Minnesota
	Email: tirad014@umn.edu
	ORCID: 0000-0003-0432-091X


	Name:  Julian S Cooper
	Institution: University of Minnesota
	Email: coop0409@umn.edu
	ORCID: 0000-0001-8390-3915


	Name:  Nathan M Springer
	Institution: University of Minnesota
	Email: springer@umn.edu
	ORCID: 0000-0002-7301-4759


	Name:  Cory D Hirsch
	Institution: University of Minnesota
	Email: cdhirsch@umn.edu
	ORCID: 0000-0002-3409-758X


	Name:  Candice N Hirsch
	Institution: University of Minnesota
	Email: cnhirsch@umn.edu
	ORCID: 0000-0002-8833-3023


3. Date published or finalized for release: 2023-01-27


4. Date of data collection (single date, range, approximate date): 2018-05-23 to 2021-08-06


5. Geographic location of data collection (where was data collected?): Univeristy of Minnesota St Paul Campus


6. Information about funding sources that supported the collection of the data:
	Minnesota Corn Growers Association
	National Science Foundation
	Bayer Crop Science
	UMII-MnDRIVE PhD Graduate Assistantship


7. Overview of the data (abstract):
Plant height is used in many breeding programs for assessing plant health across environments and predicting yield, which can be used in identifying superior hybrids or evaluating abiotic stress factors. This has often been measured at a single time point when plants have reached their terminal height for the season. Collection of plant height using unoccupied aerial vehicles (UAVs) is faster, allowing for measurements throughout the growing season which could facilitate a better understanding of plant-environment interaction and responses. To assess variation in plant height and growth rate throughout development, plant height data was collected weekly for a panel of ~500 diverse inbred lines over four growing seasons. The variation in plant height throughout the season was found to be significantly explained by genotype, year, and genotype-by-year interactions throughout vegetative growth. However, the relative contributions of these different sources of variation fluctuated throughout development. This variation was further captured by FrÃ©chet distance values which identified genotypes with consistently high or low distances in each of the four years - high distance genotypes being more dissimilar between replications and therefore capturing more environmental variation. Genome-wide association studies revealed many significant SNPs associated with plant height and growth rate at different parts of the growing season that would not be identified by terminal height alone. When comparing growth rates estimated from plant height to growth rates estimated from another morphological characteristic, canopy cover, we found greater stability in growth curves estimated by plant height. This potentially makes canopy cover more useful for understanding environmental modulation of overall plant growth and plant height better for understanding genotypic modulation of overall plant growth. Overall, this suggests evaluations of plant growth throughout the season provide more information than terminal plant height alone.


--------------------------
SHARING/ACCESS INFORMATION
--------------------------

1. Licenses/restrictions placed on the data: CC0 1.0 Universal (http://creativecommons.org/publicdomain/zero/1.0/)

2. Terms of Use: Data Repository for the U of Minnesota (DRUM) By using these files, users agree to the Terms of Use. https://conservancy.umn.edu/pages/drum/policies/#terms-of-use


---------------------
DATA & FILE OVERVIEW
---------------------

File List

	Filename: mmddyyyynew_geotiffDEM.tif 
	Short description: Digital Elevation Model (DEM)
	
	Filename: mmddyyyynewAgisoft_geotiffDEM.tif 
	Short description: Digital Elevation Model (DEM)

	Filename: mmddyyyynew_geotiff.tif 
	Short description: orthomosaic (ortho)

	Filename: mmddyyyynewAgisoft_geotiff.tif 
	Short description: orthomosaic (ortho)

	Filename: Date_to_GDD_StPaul_All.csv 
	Short description: Flight dates to Growing Degree Days all years

	Filename: 20XX_plotboundaries.zip 
	Short description: Plot boundaries (2018 - 24 shapefiles; 2019 - 44 shps, 2020 - 21 shps, 2021 - 22 shps)

	Filename: Genotype_Plant_Height_Data_StPaul_20XX.txt 
	Short description: UAV Plot Plant Height Measurements and Plot Descriptive Data

	Filename: HandHeight_20XX.zip 
	Short description: Manual Height Measurements (2018-2020 - 9 .csv files each, 2021 - 8 .csv files)

	Filename: Weather_20XX.csv 
	Short description: Weather Station Data

	Filename: 20XX_plantratio_plot.csv 
	Short description: Canopy cover data and plots

	Filename: mmddyyyy_c6m36doubledilated.tif 
	Short description: Masked canopy cover TIFFs



2. Relationship between files: mmddyyyynew_geotiffDEM.tif and mmddyyyynewAgisoft_geotiffDEM.tif are the same file but in different itterations. Some Digital Elevation Models had to be reprocessed resulting in mmddyyyynewAgisoft_geotiffDEM.tif instead of mmddyyyynew_geotiffDEM.tif. Similarly, mmddyyyynew_geotiff.tif and mmddyyyynewAgisoft_geotiff.tif are both orthomosaics but in different iterations. Both the orthomosaics and Digital Elevation Models (DEM)s are outputs from Agisoft Metashape Pro. 20XX_plotboundaries.zip contain shapefiles defining the boundaries of each plot for data extraction.  mmddyyyy_c6m36doubledilated.tif contains the masks used for differentiating canopy from background pixels in extracting canopy cover data. Genotype_Plant_Height_Data_StPaul_20XX.txt and 20XX_plantratio_plot.csv contain that raw extracted data (plant height and canopy cover). HandHeight_20XX.zip contains files with manual plant height data collected on random plots throughout the season for quality control. Weather_20XX.csv contains weather information such as minimum and maximum temperatures per day for calculation of growing degree days and Date_to_GDD_StPaul_All.csv contains those growing degree days for data comparison across years. 


--------------------------
METHODOLOGICAL INFORMATION
--------------------------

1. Description of methods used for collection/generation of data: 


Experimental design and protocol for height extraction can be found at Tirado et al. 2020 (https://onlinelibrary.wiley.com/doi/10.1002/pld3.230)

Experimental Field Design:

A set of 501 diverse inbred lines from the Wisconsin Diversity Panel were grown in summer 2018, 2019, 2020, and 2021. Lines were grown in single-row plots that were 15.5 feet long center-to-center including 4 foot alleys with 30 inch row-spacing, and were planted at a density of approximately 70,000 plants per hectare. All experiments were planted as a randomized complete-block design with two replicates. Within replicates, genotypes were further blocked by flowering time with the earlier flowering lines flowering at approximately 71-80 days after planting and the later flowering lines flowering at approximately 80-87 days after planting and randomized within the block within the replicate. These flowering time blocks within replicates were included to account for variation due to flowering time that often contributes to variation in plant height. Inbred lines B73 and PH207 were planted as checks within each block, with five entries of each check per block. These trials were planted on May 14, 2018, May 30, 2019, May 7, 2020, and May 6, 2021 at the Minnesota Agricultural Experiment Station in St. Paul, MN. 

UAV Data Collection and Processing:

The experiment was imaged weekly from planting until plants reached terminal height using a DJI Phantom 4 Advanced drone in 2018 and 2019 and a DJI Phantom 4 RTK drone in 2020 and 2021. Images were collected at an altitude of 30 m above ground to achieve a ground sampling distance (GSD) of approximately 0.82 cm with 80% front overlap and 80% side overlap to maximize reconstruction efficiency. Flights were collected at 14 timepoints in 2018, 27 timepoints in 2019, and 12 timepoints in 2020 and 11 timepoints in 2021 (Supplemental Table 2). Ground targets of known height and half a meter wide were placed around the border of the field for use as ground control points (GCPs). There were 9 GCPs included in 2018, 12 in 2019, 8 in 2020, and 7 in 2021. The real world coordinates of these GCPs were collected using real time kinematic positioning with a Swift Console (v 2.3.17) base station and rover (GNSS compass configuration).

Weather Data and Growing Degree Days Calculation:

Daily min and max temperature data from the University of Minnesota St Paul weather station (Station ID 218450) was extracted. Growing Degree Units (GDUs) were then calculated utilizing the max and min temperatures for each date and the cumulative sum of these was extracted and assigned to each date of data collection based on the planting date. 


2. Methods for processing the data: <describe how the submitted data were generated from the raw or collected data>


Methods can be found at Tirado et al. 2020 (https://onlinelibrary.wiley.com/doi/10.1002/pld3.230)

Briefly, Agisoft Software (Agisoft Metashape Professional v1.7.5) was used to process the images to generate crop service models (CSMs) and RGB orthomosaics for each flight. QGIS software (QGIS v3.16, 2021) was used for plot boundary extraction by overlaying a grid based on plot size and spacing and exporting plot coordinates. Custom MATLAB scripts for plant height were used to extract height estimates for individual plots using a previously described exposed alley subtraction method.

All extracted plant heights were normalized to real world measurements by comparing the extracted GCP heights to the known GCP heights. This was completed by dividing the real world GCP height by the extracted GCP height and then multiplying the value by the extracted plant height (Supplemental Figure 2). Plots with less than 10 plants were removed from the analysis due to the decrease in extracted plant height accuracy and the competition from neighboring plots affecting the actual plant height. This filtering step removed 325 out of the original 4,160 plots across the four environments. Individual data points (i.e. single plots within a single flight date) were also removed if they were classified as a dip when the plot height was less than 80% of the previous day and also remained less than the next day, or if the individual data points were classified as a peak when the plot height was more than 120% of the next day while still remaining more than the previous day (n=2,273 data points removed as dips or peaks). Entire plots were removed from a location if the plot height dipped or peaked on 3 or more occasions (n=172 plots removed). In total, 3,663 of the original 4,160 plots across the four years had at least 9 individual measurements within the year after these filtering steps and were retained for downstream analysis. 


3. Instrument- or software-specific information needed to interpret the data:


MATLAB is necessary for extracting plant height and canopy cover from the tiff images, but the raw data is also present. Plot boundary files can be viewed in QGIS software. 


4. Standards and calibration information, if appropriate:


Ground targets of known height and half a meter wide were placed around the border of the field for use as ground control points (GCPs). There were 9 GCPs included in 2018, 12 in 2019, 8 in 2020, and 7 in 2021. The real world coordinates of these GCPs were collected using real time kinematic positioning with a Swift Console (v 2.3.17) base station and rover (GNSS compass configuration). 
 

5. Describe any quality-assurance procedures performed on the data:


All extracted plant heights were normalized to real world measurements by comparing the extracted GCP heights to the known GCP heights. This was completed by dividing the real world GCP height by the extracted GCP height and then multiplying the value by the extracted plant height. Plots with less than 10 plants were removed from the analysis due to the decrease in extracted plant height accuracy and the competition from neighboring plots affecting the actual plant height. Individual data points (i.e. single plots within a single flight date) were also removed if they were classified as a dip when the plot height was less than 80% of the previous day and also remained less than the next day, or if the individual data points were classified as a peak when the plot height was more than 120% of the next day while still remaining more than the previous day. Entire plots were removed from a location if the plot height dipped or peaked on 3 or more occasions. 


6. People involved with sample collection, processing, analysis and/or submission:


Dorothy Sweet, Sara Tirado, Julian Cooper, Nathan Springer, Cory Hirsch, and Candice Hirsch


-----------------------------------------
DATA-SPECIFIC INFORMATION FOR: Date_to_GDD_StPaul_All
-----------------------------------------
A. Number of rows:  47
B. Number of columns: 3
C. Variable list
	1. Name: Date
	Description: Date of image capture (d-Mon)

	2. Name: GDD
	Description: Cumulative growing degree days for date of image capture

	3. Name: Year
	Description: Year of image capture

-----------------------------------------
DATA-SPECIFIC INFORMATION FOR: Weather_20XX
-----------------------------------------
A. Number of rows:  8760
B. Number of columns: 10
C. Variable list
	1. Name: Date
	Description: Date and time of weather capture (m/d/yy h:mm)

	2. Name: RN
	Description: data point collection number

	3. Name: Rainfall
	Description: rainfall (inches)

	4. Name: AirTMax(F)
	Description: max air temperature (Degrees F)

	5. Name: AirTMin(F)
	Description: minimum air temperature (Degrees F)

	6. Name: AirTSmp(F)
	Description: air temperature SMP (Degrees F)

	7. Name: AirTAvg(F)
	Description: average air temperature during the hour of interest

	8. Name: WindMphMax
	Description: maximum windspeed in miles per hour

	9. Name: WindMphAvg
	Description: average windspeed per hour in miles per hour

	10. Name: WindDir
	Description: wind direction during the hour (degrees)




-----------------------------------------
DATA-SPECIFIC INFORMATION FOR: HandHeight_20XX.zip CSVs
-----------------------------------------
Variable list
	1. Name: Plot
	Description: plot name 

	2. Name: Height1
	Description: manual height of plant (inches)

	3. Name: Height2
	Description: manual height of plant (inches)

	4. Name: Height3
	Description: manual height of plant (inches)

	5. Name: Height4
	Description: manual height of plant (inches)

	6. Name: Height5
	Description: manual height of plant (inches)


-----------------------------------------
DATA-SPECIFIC INFORMATION FOR: XXXX_plantratio_plot.csv
-----------------------------------------

Variable List
	1. Name: Plot
	Description: plot name

	2. Remaining columns: Names are growing degree days for the data in the column
	Description: canopy cover plant ratio values for each flight date and each plot



