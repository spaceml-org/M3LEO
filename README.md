# generalisableSAR

This repo contains information about multimodal multilabel wide area datasets, collated during the [2022 ESA-funded Frontier Development Lab](https://fdleurope.org/fdl-europe-2023) project focussed on developing generalizable machine learning models for Synthetic Aperture Radar (SAR) data. 

Each datasets contains co-aligned, tiled chips of Earth Observation data. Each chip covers an area 4480m x 4480m (448x448 pixels at 10/pixel)

![samples](imgs/samples.png)

## AOIs

The AOIs covered by these datasets cover approximately 10% of the Earth's landmass. 

![samples](imgs/aois.png)

Each AOI has a `.geojson` file associated with the geometries and ids of every image chip and the predefined train, test and val splits. You can use it to select and download only the chips of a certain dataset or region. For instance, the following illustrates the chips covering Massachusetts.

![AOIs](imgs/regionchips.png)

## Datasets

The following datasets are available

![datasets](imgs/datasets.png)

- **gunw-yyyy-mm**: ARIA Sentinel-1 Geocoded Unwrapped Interferograms on year yyyy and month mm selecting the date within each month that has most interferometric pairs as first date.

- **gunw-dateinit_dateend**: ARIA Sentinel-1 Geocoded Unwrapped Interferograms selecting within the [dateinit, datend] period the date that has most interferometric pairs as first date.

- **gunw48-yyyy-mm**: ARIA Sentinel-1 Geocoded Unwrapped Interferograms on year yyyy and month mm selecting all interferometric pairs at most 48 apart.

For gunw datasets see https://asf.alaska.edu/data-sets/derived-data-sets/sentinel-1-interferograms/

- **gssic**:

- **s1grd-yyyy**: Sentinel 1, three channels (vv, vh, vv/vh) taking the seasonal median (4 on a year) on both ascending and descending modes.

- **s1grdm-yyyy-asc**: Sentinel 1, three channels (vv, vh, vv/vh) taking the monthly meedian on ascending passes.

- **s2rgb-yyyy**: Sentinel 2, three channels (red, green, blue) seasonal cloudless median (4 on a year)

- **s2rgbm-yyyy**: Sentinel 2, three channels (red, green, blue) monthly cloudless median

- **strmdem**: NASA SRTM DEM 30m resolution, from https://developers.google.com/earth-engine/datasets/catalog/CGIAR_SRTM90_V4

- **esaworldcover-2020**: ESA WorldCover v100, from https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v100

- **esaworldcover-2021**: ESA WorldCover v200, from https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200

- **biomass-yyyy**: ESA CCI Above Ground Biomass for year yyyy. See https://climate.esa.int/en/projects/biomass/

- **modis44b006veg**: MODIS Vegetation Continuous Field Yearly Global 250m. See https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD44B

- **ghsl-built-S-yyyy**: EU JRC Global Human Settlement Layer, Builtup Surface for year yyyy. See https://ghsl.jrc.ec.europa.eu/download.php?ds=bu

- **globalfloods-2018**: Global Flood Database v1, for year 2018. See https://developers.google.com/earth-engine/datasets/catalog/GLOBAL_FLOOD_DB_MODIS_EVENTS_V1

- **firecci51**: ESA CCI Burned Area Pixel Product version 5.1. See https://developers.google.com/earth-engine/datasets/catalog/ESA_CCI_FireCCI_5_1

- **lcci-yyyy**: ESA CCI LandCover for year yyyy. See https://www.esa-landcover-cci.org/




## Example data splits

## Example notebooks

- `01 - inspect chip definitions.ipynb` to download and understand the chip definition `.geojson`
- `02 - download chips from gcp bucket` to select a region and download the chips from that region on the datasets of your selection
- `03 - inspect and visualize datasets` to open downloaded image chips, understand their channels and metadata structure and visualize them.

## Requirements

The datasets can be accessed via the Google Cloud Platform (GCP). Therefore, a working GCP account with `gsutils` installed locally is needed.
