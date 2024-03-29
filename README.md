# Trans2Spatial Project

Code adapted from https://github.com/lucidrains/muse-maskgit-pytorch

## dataloader.py

This Python script is used for loading and preprocessing for spatial transcriptomics data. 

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

## vqgan_vae.py

This Python script is used for implementing a VQGAN-VAE model. It uses the PyTorch library to create the model and its components.

### Helper Functions

The script includes several helper functions for tasks such as checking if a value exists, defaulting a value if it doesn't exist, and grouping dictionaries by key.

### Tensor Helper Functions

The script also includes several helper functions for working with tensors, such as calculating the gradient penalty, applying a leaky ReLU activation function, and calculating safe division.

### GAN Losses

The script includes functions for calculating the hinge and binary cross-entropy losses for a discriminator and a generator.

### VQGAN-VAE

The script includes several classes for implementing a VQGAN-VAE model:

- `LayerNormChan`: A layer normalization module that normalizes across the channel dimension.
- `Discriminator`: A discriminator network for the GAN.
- `ResnetEncDec`: A ResNet-based encoder-decoder network.
- `GLUResBlock`: A gated linear unit (GLU) ResNet block.
- `ResBlock`: A standard ResNet block.
- `VQGanVAE`: The main VQGAN-VAE model, which includes an encoder-decoder network, a vector quantization (VQ) layer, and optionally a discriminator and a VGG network for perceptual loss.

The VQGanVAE class includes methods for encoding and decoding images, as well as a forward method that calculates the loss for training the model. The class also includes methods for saving and loading the model's state.

## muse_maskgit_STTT.py

This Python script implements the model Muse and low/high resolution MaskGit model, which is a variant of the Transformer model, specifically designed for single-cell RNA sequencing (scRNASeq) data. The model is used for generating spatial transcriptomic data based on the scRNA-seq data.

### Helper Functions

Several helper functions are defined for operations such as checking if a value exists, normalizing tensors, getting a subset of a mask, and others.

### Model Components

Various classes are defined to build the components of the Transformer model. These include LayerNorm, GEGLU, FeedForward, Attention, TransformerBlocks, and Transformer.

### `SelfCritic` Class

This class is a wrapper around the Transformer model and is used to predict the next token.

### `MaskGitTransformer` and `TokenCritic` Classes

These are specialized versions of the Transformer model.

### `MaskGit` Class

This is the main class implementing the MaskGit model. It includes methods for saving and loading the model, as well as a forward method for training and a generate method for generating images.

### `Muse` Class

This class combines the base MaskGit model with a super-resolution MaskGit model to generate high-resolution images.

The script also includes code for handling conditioning on low-resolution images, self-conditioning, and negative prompting, as well as code for sampling and noise schedules.

## trainer.py

This python script defines two classes, `VQGanVAETrainer` and `MaskGitTrainer`, for training two different types of models: a VQGAN-VAE model and a MaskGit model. These models are used for spatial transcriptomic data generation tasks.

### Class `VQGanVAETrainer` 

The `VQGanVAETrainer` class is used to train a VQGAN-VAE model. It includes methods for training steps, saving and loading models, and logging training progress. The class also handles the creation and management of datasets and dataloaders for training and validation, as well as the setup and management of the training environment using the Accelerator class from the accelerate library.

### Class `MaskGitTrainer`

The `MaskGitTrainer` class is used to train a MaskGit model. Similar to the VQGanVAETrainer class, it includes methods for training steps, saving and loading models, and logging training progress. It also handles the creation and management of datasets and dataloaders for training and validation, as well as the setup and management of the training environment using the Accelerator class from the accelerate library.

## main.py

The provided Python script is a main execution script for training and using models for spatial transcriptomic data generation tasks. The script is divided into four steps, each of which is commented out except for the last one.

- Step 1 is for training a `VQGAN-VAE` model. It creates an instance of the `VQGanVAE` model and an instance of the `VQGanVAETrainer` class, which is used to train the model.

- Step 2 is for training a `MaskGit` model. It loads a pre-trained `VQGAN-VAE` model, creates an instance of the `MaskGitTransformer` class, and an instance of the `MaskGitTrainer` class, which is used to train the `MaskGit` model.

- Step 3 is for training a `super-resolution MaskGit` model. It loads a pre-trained `VQGAN-VAE` model, creates an instance of the `MaskGitTransformer` class, and an instance of the `MaskGitTrainer` class with the `is_superres` parameter set to `True`, which is used to train the `super-resolution MaskGit` model.

- Step 4 is for using the trained models to generate images. It loads the trained `base MaskGit` model and the trained `super-resolution MaskGit` model, creates an instance of the `Muse` class with the trained models, and uses the Muse instance to generate images from text prompts.

