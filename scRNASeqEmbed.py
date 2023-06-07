import logging
import torch
from torch import nn
import numpy as np

from typing import List, Union

# transformers.logging.set_verbosity_error()

def exists(val):
    return val is not None

class scRNASeqEmbedding(nn.Module):
    def __init__(self, num_cells: int, embedding_dim: int = 512):
        super(scRNASeqEmbedding, self).__init__()

        self.embedding = nn.Embedding(num_cells, embedding_dim)
        self.embedding_dim = embedding_dim
        self.embedding.weight.data = torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, scRNA_count: torch.tensor):
        num_genes, num_cells = scRNA_count.shape
        scRNA_count_tensor = torch.tensor(scRNA_count, dtype=torch.float32).to(self.device)

        embedded_cells = self.embedding(torch.arange(num_cells).to(self.device))
        embedded_cells = embedded_cells.transpose(0, 1).unsqueeze(0).repeat(num_genes, 1, 1)

        scRNA_count_expanded = scRNA_count_tensor.unsqueeze(1)
        embedding_with_counts = torch.cat((embedded_cells, scRNA_count_expanded), dim=1)

        return embedding_with_counts
    
def test_scRNASeqEmbedding():
    # Example embedding
    num_genes = 50
    num_cells = 100
    embedding_dim = 18

    # Example scRNA_count matrix
    scRNA_count = np.random.randint(0, 100, (num_genes, num_cells))
    gene_embedding = scRNASeqEmbedding(num_cells, embedding_dim)
    output_embeddings = gene_embedding(scRNA_count)

    print(output_embeddings.shape)  # Should output torch.Size([num_genes, embedding_dim+1, num_cells])

if __name__ == "__main__":
    test_scRNASeqEmbedding()