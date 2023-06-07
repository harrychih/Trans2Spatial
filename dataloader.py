import pandas as pd
import numpy as np
import pyspark
from pyspark.sql import SparkSession
from beartype import beartype

import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import csv

from vqgan_vae import VQGanVAE

# # helper functions

# def exists(val):
#     return val is not None

# def identity(t, *args, **kwargs):
#     return t

# def noop(*args, **kwargs):
#     pass

# def find_index(arr, cond):
#     for ind, el in enumerate(arr):
#         if cond(el):
#             return ind
#     return None

# def find_and_pop(arr, cond, default = None):
#     ind = find_index(arr, cond)

#     if exists(ind):
#         return arr.pop(ind)

#     if callable(default):
#         return default()

#     return default

# def cycle(dl):
#     while True:
#         for data in dl:
#             yield data

# def cast_tuple(t):
#     return t if isinstance(t, (tuple, list)) else (t,)

# def yes_or_no(question):
#     answer = input(f'{question} (y/n) ')
#     return answer.lower() in ('yes', 'y')

# def accum_log(log, new_logs):
#     for key, new_value in new_logs.items():
#         old_value = log.get(key, 0.)
#         log[key] = old_value + new_value
#     return log

# def pair(val):
#     return val if isinstance(val, tuple) else (val, val)

# def convert_image_to_fn(img_type, image):
#     if image.mode != img_type:
#         return image.convert(img_type)
#     return image

def scale_coordinates(coords, target_size):
    x_coords, y_coords = zip(*coords)

    min_x, min_y = min(x_coords), min(y_coords)
    max_x, max_y = max(x_coords), max(y_coords)

    def scale_value(val, min_val, max_val, target_size):
        return int(((val - min_val) / (max_val - min_val)) * (target_size - 1))

    scaled_coords = [(scale_value(x, min_x, max_x, target_size),
                     scale_value(y, min_y, max_y, target_size))
                     for x, y in coords]

    return scaled_coords

def get_num_cells(data_dir):

    scRNAseq_dir = data_dir + 'scRNA_count_sorted.csv'
    matrix = pd.read_csv(scRNAseq_dir, sep=',', header=0, index_col=0).values

    return matrix.shape[1]

    
class scRNAseqSTDataset(Dataset):
    def __init__(self, data_dir, image_size):
        super().__init__()
        scRNAseq_dir = data_dir + 'scRNA_count_sorted.csv'
        insitu_count_dir = data_dir + 'Insitu_count.txt'
        Location_dir = data_dir + 'Locations.txt'

        spark = SparkSession.builder \
            .appName("Large Text Files Reader") \
            .config('spark.port.maxRetries', 100) \
            .config("spark.driver.memory", "8g") \
            .getOrCreate()

        ## Spatial Transcriptomic Image generate
        self.insitu_count_df = spark.read.csv(insitu_count_dir, sep="\t", header=True)
        self.locations_df = spark.read.csv(Location_dir, sep="\t", header=True)

        print(f'{len(self.insitu_count_df.columns)} gene training samples found at {data_dir}')

        # Convert the locations_df DataFrame to a list of tuples
        self.locations = self.locations_df.rdd.map(lambda row: (float(row[0]), float(row[1]))).collect()

        # Scale the locations to fit a image-like system
        self.scaled_locations = scale_coordinates(self.locations, image_size)

        # Create an empty 3D array with dimensions (512, 512, num_genes)
        self.num_genes = len(self.insitu_count_df.columns)
        self.gene_to_idx = {gene: idx for idx, gene in enumerate(self.insitu_count_df.columns)}
        spatial_image = np.zeros((image_size, image_size, self.num_genes))

        # create pandas dataframes
        self.insitu_count = self.insitu_count_df.toPandas()

        # iterate through the rows of the insitu_count dataframe, and fill the spatial_image array
        for index, row in self.insitu_count.iterrows():
            for gene in self.insitu_count_df.columns:
                spatial_image[self.scaled_locations[index][0], self.scaled_locations[index][1], self.gene_to_idx[gene]] = row[gene]

        self.spatial_images = torch.from_numpy(np.float32(spatial_image))


        ## ScRNASeq_count preprocessing
        self.scRNAseq_df_np = pd.read_csv(scRNAseq_dir, sep=',', header=0, index_col=0).values
        self.scRNAseq_df = torch.from_numpy(np.float32(self.scRNAseq_df_np))

    def __len__(self):
        return self.num_genes
    
    def __getitem__(self, idx):
        return self.scRNAseq_df[idx,:], self.spatial_images[:,:,idx].unsqueeze(dim=0)


