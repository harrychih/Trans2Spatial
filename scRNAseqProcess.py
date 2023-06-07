import pyspark
from pyspark.sql import SparkSession

from typing import Tuple

spark = SparkSession.builder \
            .appName("Large Text Files Reader") \
            .config('spark.port.maxRetries', 100) \
            .config("spark.driver.memory", "8g") \
            .config("spark.executor.memory", "8g") \
            .getOrCreate()


def gene_seq(insitu_count_file:str) -> Tuple(dict, dict):
    insitu_count_df = spark.read.csv(insitu_count_file, sep="\t", header=True)
    num_genes = len(insitu_count_df.columns)

    print(f'{num_genes} gene training samples found at {insitu_count_file}')

    gene_to_idx = {gene: idx for idx, gene in enumerate(insitu_count_df.columns)}
    idx_to_gene = {idx: gene for idx, gene in enumerate(insitu_count_df.columns)}

    return gene_to_idx, idx_to_gene






