# generalisableSAR

Multimodal multilabel wide area datasets.

Each chip covers an area 4480m x 4480m (448x448 pixels at 10/pixel)

![samples](imgs/samples.png)

## AOIs

The AOIs covered by these datasets amount for approximately 10% of the Earth landmass. 

![samples](imgs/aois.png)

Each AOI has a `.geojson` file associated with the geometries and ids of every image chip and the predefined train, test and val splits. You can use it to select and download only the chips of a certain dataset or region. For instance, the following illustrates the chips covering Massachusetts.

![AOIs](imgs/regionchips.png)

## Example notebooks

-  `01 - inspect chip definitions.ipynb` to download and understand the chip definition `.geojson`
-  `02 - download chips from gcp bucket` to select a region and download the chips from that region on the datasets of your selection
- `03 - inspect and visualize datasets` to open downloaded image chips, understand their channels and metadata structure and visualize them.

## Requirements

A working GCP account with `gsutils` install locally.
