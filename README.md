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
