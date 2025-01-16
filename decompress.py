from __future__ import annotations

import os

import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession

sedona = (
    SparkSession.builder.config("spark.driver.memory", "28g")
    .config("spark.executor.memory", "28g")
    .config("spark.sql.parquet.columnarReaderBatchSize", "1024")
    .appName("SedonaRaster")
    .getOrCreate()
)
sc = sedona.sparkContext
num_cores = sc.defaultParallelism
logger.info(f"Number of cores used: {num_cores}")

hadoop_version = sc._jvm.org.apache.hadoop.util.VersionInfo.getVersion()
logger.info(f"Hadoop version: {hadoop_version}")

conf = sc.getConf()
for k, v in conf.getAll():
    logger.info(f"{k} = {v}")

# As-is this script should spit the .tif/.nc files out in the same folder as the parquet files
# You might want to add some extra code at the end to remove the parquet files
# Or otherwise change the folders below

# I haven't tested this script extensively but it should be pretty close. It's lifted from another script
# in the dev repo that works so it should work or be close to working

# Stuff you need to change=======================================================================
parquet_splits_file = "wherever/europe_partitions_aschips_2f8bd3f01ddd5_splits_60bands_angle09_60-20-20_parquet.csv"
base_data_dir = "home/matt/data/m3leo/"
aoi_dir = "europe/europe_partitions_aschips_2f8bd3f01ddd5/"  # Folder for all datasets for the AOI
aoi_data_dir = os.path.join(base_data_dir, aoi_dir)

parquet_folder = "biomass-2018"  # Folder containing biomass parquet files
local_parquet_path = os.path.join(aoi_data_dir, parquet_folder)
# =================================================================================================
parquet_codec = "zstd"  # Shouldn't need to change this


splits = pd.read_csv(
    parquet_splits_file
)  # Read the splits file with the parquet names for the AOI
# E.G. https://huggingface.co/datasets/M3LEO/europe/blob/main/europe_partitions_aschips_2f8bd3f01ddd5_splits_60bands_angle09_60-20-20_parquet.csv


def save_tif(row, out_folder, test=False):  ###Saves row of a parquet file as tif
    """
    row - row from df
    out_folder e.g. ~/home/matt/data/m3leo/europe/europe_partitions_aschips_2f8bd3f01ddd5/biomass-2018 etc
    """
    file_path_in_folder = row[
        "path"
    ]  # should be 'southamerica/southamerica_partitions_aschips_3219aa0f411c2/s1grd-2020/2e0c475f0dc75.tif' etc

    fname = os.path.basename(file_path_in_folder)
    output_path = os.path.join(out_folder, fname)
    ds = row["ds"]  # dataset e.g. s1grd-2020

    if test:
        output_path = output_path.replace(ds, f"{ds}-test")

    ds_dir = os.path.dirname(output_path)  # everything except file name
    os.makedirs(ds_dir, exist_ok=True)

    tif_data = row["content"]
    with open(output_path, "wb") as f:
        f.write(tif_data)


sedona.conf.set("spark.sql.parquet.enableVectorizedReader", "false")

parquet_identifiers = splits["parquet_identifier"].unique()
for parquet_identifier in parquet_identifiers:
    logger.info(f"Decompressing {parquet_identifier}")
    target_parquet_file = os.path.join(
        local_parquet_path, f"{parquet_identifier}.{parquet_codec}.parquet"
    )
    df_loaded = sedona.read.parquet(target_parquet_file)
    df_loaded.rdd.foreach(
        lambda row: save_tif(row, out_folder=local_parquet_path, test=True)
    )

# You might want to remove the parquet files after decompressing them
