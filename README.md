# Trans2Spatial Project

## dataloader.py

This Python script is used for loading and preprocessing data for a spatial transcriptomics project. 

### Functions

`scale_coordinates(coords, target_size)`

This function scales a set of coordinates to a target size. It takes two arguments:

- `coords`: A list of tuples representing x, y coordinates.
- `target_size`: The size to scale the coordinates to.

The function returns a list of tuples representing the scaled coordinates.

`get_num_cells(data_dir)`

This function reads a CSV file containing scRNA count data and returns the number of cells (columns) in the data. It takes one argument:

- `data_dir`: The directory where the CSV file is located.

### Class `scRNAseqSTDataset(Dataset)`

This class represents a dataset for spatial transcriptomics and scRNA-seq data. It inherits from the PyTorch `Dataset` class.

`__init__(self, data_dir, image_size)`
The initializer for this class takes two arguments:

- `data_dir`: The directory where the data files are located.
- `image_size`: The size of the image to be generated from the spatial transcriptomics data.
The initializer reads in several data files, scales the spatial transcriptomics data to fit an image-like system, and converts the data into PyTorch tensors.

`__len__(self)`
This method returns the number of genes in the dataset.

`__getitem__(self, idx)`
This method returns a tuple containing the scRNA-seq data and spatial transcriptomics data for a given gene. It takes one argument:

- `idx`: The index of the gene to retrieve data for.

The scRNA-seq data is returned as a 1D tensor, and the spatial transcriptomics data is returned as a 2D tensor with an additional dimension for the color channel.

## scRNAseqProcess.py

This Python script is used for processing single-cell RNA sequencing (scRNA-seq) data. It uses the PySpark library to handle large data files.

### SparkSession Initialization

A SparkSession is initialized with the name "Large Text Files Reader". The configuration is set to allow a maximum of 100 retries for the port, and both the driver and executor memory are set to 8 gigabytes.

### Function

`gene_seq(insitu_count_file:str) -> Tuple(dict, dict)`

This function reads a tab-separated file containing in-situ count data and returns two dictionaries mapping genes to indices and indices to genes. It takes one argument:

- `insitu_count_file`: The path to the file containing the in-situ count data.

The function returns a tuple of two dictionaries:

- `gene_to_idx`: A dictionary mapping gene names to their corresponding indices.
- `idx_to_gene`: A dictionary mapping indices to their corresponding gene names.


## scRNASeqEmbed.py

This Python script is used for embedding single-cell RNA sequencing (scRNA-seq) data. It uses the PyTorch library to create an embedding layer and apply it to the scRNA-seq data.

### Function

`exists(val)`

This function checks if a value is not None. It takes one argument:

- `val`: The value to check.

### Class `scRNASeqEmbedding(nn.Module)`

This class represents an embedding layer for scRNA-seq data. It inherits from the PyTorch `nn.Module` class.

`__init__(self, num_cells: int, embedding_dim: int = 512)`

The initializer for this class takes two arguments:

- `num_cells`: The number of cells in the scRNA-seq data.
- `embedding_dim`: The dimension of the embedding space. The default value is 512.

The initializer creates an embedding layer and initializes its weights using the Xavier uniform initialization method. It also sets the device to use for computations.

`forward(self, scRNA_count: torch.tensor)`

This method applies the embedding layer to the scRNA-seq data. It takes one argument:

- `scRNA_count`: A tensor containing the scRNA-seq data.

The method returns a tensor containing the embedded scRNA-seq data.

### Function

`test_scRNASeqEmbedding()`

This function tests the `scRNASeqEmbedding` class by creating an instance of the class and applying it to a random scRNA-seq data matrix. The shape of the output tensor is printed to the console.


