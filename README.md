# Earth Observation Datasets

This repository contains information about the multimodal, multilabel, wide area datasets collated during the [2023 ESA-funded Frontier Development Lab](https://fdleurope.org/fdl-europe-2023) project. While the primary aim of the project was to develop generalizable machine learning models for Synthetic Aperture Radar (SAR) data, the resulting datasets can be used for any machine learning application requiring matching tiles of Earth Observation (EO) data and associated labels. For each dataset, we provide co-aligned, tiled chips covering an area of 4480m x 4480m (448x448 pixels at 10m/pixel) each. The following image shows an overview of the available datasets. 

![samples](imgs/samples.png)

## Areas Of Interest

The areas of interest (AOIs) covered by our datasets make up approximately 10% of Earth's landmass. 

![samples](imgs/aois.png)

Each AOI has a `.geojson` file associated with the geometries and ids of every image chip. We also provide predefined train, test and validation splits that can be used for repeatability and comparability of experiments. Using `.geojson` files, users can also select and download only chips of a certain dataset or region. For instance, the following figure illustrates the chips covering Massachusetts.

![AOIs](imgs/regionchips.png)

## Datasets

Our compiled datasets cover >60TB and >25M tiles derived from 13 data sources. For certain data sources, we provide multiple versions of the compiled datasets, e.g. containing seasonal or monthly composites. The table below details the size and number of tiles for each available dataset.

![datasets](imgs/datasets.png)

**ARIA Sentinel-1 Geocoded Unwrapped Interferograms**. See https://asf.alaska.edu/data-sets/derived-data-sets/sentinel-1-interferograms/

- `gunw-yyyy-mm`: ARIA Sentinel-1 Geocoded Unwrapped Interferograms on year yyyy and month mm, selecting the date within each month that has most interferometric pairs as first date.

- `gunw-dateinit_dateend`: ARIA Sentinel-1 Geocoded Unwrapped Interferograms, selecting within the [dateinit, datend] period the date that has most interferometric pairs as first date.

- `gunw48-yyyy-mm`: ARIA Sentinel-1 Geocoded Unwrapped Interferograms on year yyyy and month mm, selecting all interferometric pairs at most 48 days apart.

**Global Seasonal Sentinel-1 Interferometric Coherence and Backscatter Dataset**

- `gssic`: Global Seasonal Sentinel-1 Interferometric Coherence and Backscatter Dataset 2020. See https://asf.alaska.edu/datasets/derived/global-seasonal-sentinel-1-interferometric-coherence-and-backscatter-dataset/

**Sentinel 1 GRD**

- `s1grd-yyyy`: Sentinel-1 SAR GRD: C-band Synthetic Aperture Radar Ground Range Detected, three channels (vv, vh, vv/vh) taking the seasonal median (4 seasons per year) for both ascending and descending modes. See https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD

- `s1grdm-yyyy-asc`: Sentinel 1 SAR GRD: C-band Synthetic Aperture Radar Ground Range Detected, three channels (vv, vh, vv/vh) taking the monthly median for ascending passes. See https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD

**Sentinel 2 RGB**

- `s2rgb-yyyy`: Harmonized Sentinel-2 Level 2A, three channels (red, green, blue) seasonal cloudless median (4 per year). See https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED

- `s2rgbm-yyyy`: Harmonized Sentinel-2 Level 2A, three channels (red, green, blue) monthly cloudless median. See https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED

**Digital Elevation Model**

- `strmdem`: NASA SRTM DEM 30m resolution, from https://developers.google.com/earth-engine/datasets/catalog/CGIAR_SRTM90_V4

**Labelled datasets**

- `biomass-yyyy`: ESA CCI Above Ground Biomass for year yyyy. See https://climate.esa.int/en/projects/biomass/

- `firecci51`: ESA CCI Burned Area Pixel Product version 5.1. See https://developers.google.com/earth-engine/datasets/catalog/ESA_CCI_FireCCI_5_1

- `lcci-yyyy`: ESA CCI Medium Resolution Land Cover for year yyyy. See https://www.esa-landcover-cci.org/

- `esaworldcover-2020`: ESA World Cover v100, from https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v100

- `esaworldcover-2021`: ESA World Cover v200, from https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200

- `modis44b006veg`: MODIS Vegetation Continuous Field Yearly Global 250m. See https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD44B

- `ghsl-built-S-yyyy`: EU JRC Global Human Settlement Layer, Builtup Surface for year yyyy. See https://ghsl.jrc.ec.europa.eu/download.php?ds=bu

- `globalfloods-2018`: Global Flood Database v1, for year 2018. See https://developers.google.com/earth-engine/datasets/catalog/GLOBAL_FLOOD_DB_MODIS_EVENTS_V1

## Example notebooks

To faciliate the usage of our datasets, we provide jupyter notebooks to download, open, and interact with the different datasets.

- `01 - inspect chip definitions.ipynb` to download and understand the chip definition `.geojson`
- `02 - download chips from gcp bucket` to select a region and download the chips from that region on the datasets of your selection
- `03 - inspect and visualize datasets` to open downloaded image chips, understand their channels and metadata structure and visualize them.

## Requirements & Data Access

The datasets can be accessed via the Google Cloud Platform (GCP) [INSERT_LINK_ONCE_READY]. Therefore, a working GCP account with `gsutils` installed locally is needed.
