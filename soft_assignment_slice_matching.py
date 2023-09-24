import torch
from torch.nn.functional import softmax
from typing import Tuple

def dot_product_similarity(embeddings1, embeddings2):
    return (embeddings1.unsqueeze(1) * embeddings2.unsqueeze(0)).sum(dim=2)

def negative_distance_similarity(embeddings1, embeddings2):
    return -torch.norm(embeddings1.unsqueeze(1) - embeddings2.unsqueeze(0), dim=2)

def soft_assignment_matrix(embeddings1: torch.Tensor, embeddings2: torch.Tensor, sink_value,
        similarity_function=dot_product_similarity) -> torch.Tensor:
    similarities = similarity_function(embeddings1, embeddings2)
    score_matrix = torch.vstack([torch.hstack([similarities, torch.ones((similarities.shape[0], 1)).to(embeddings1.device) * sink_value]),
        torch.ones((1, similarities.shape[1]+1)).to(embeddings1.device) * sink_value])
    assignment_matrix = softmax(score_matrix, dim=0) * softmax(score_matrix, dim=1)
    if False:
        #print("Embeddings")
        #print(torch.vstack([embeddings1, embeddings2]))
        #print("Norm")
        #print(torch.linalg.norm(torch.vstack([embeddings1, embeddings2]), dim=1))
        print("Similarity")
        print(similarities)
        print("Score")
        print(score_matrix)
        print("Assignment")
        print(assignment_matrix)
    return assignment_matrix

def hard_assignment(soft_assignment_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assignment = soft_assignment_matrix.argmax(dim=1)[:-1] # exclude last row (sink)
    non_sink_matches = torch.where(assignment < soft_assignment_matrix.shape[1] - 1)[0]
    return ( non_sink_matches, assignment[non_sink_matches])

def hard_assignment_matrix(soft_assignment_matrix: torch.Tensor) -> torch.Tensor:
    assignment_matrix = torch.zeros_like(soft_assignment_matrix, dtype=int) # type: ignore
    assignment_matrix[(torch.arange(soft_assignment_matrix.shape[0]), soft_assignment_matrix.argmax(dim=1))] = 1
    return assignment_matrix

def soft_assignment_loss(soft_assignment_matrix: torch.Tensor, embeddings1_coords: torch.Tensor, embeddings2_coords: torch.Tensor,
        exponent_beta: float) -> torch.Tensor:
    # weights[i,j] = exp(-beta ||embeddings1_coords[i] - embeddings2_coords[j]||)
    distances = torch.norm(embeddings1_coords.unsqueeze(1) - embeddings2_coords.unsqueeze(0), dim=2)
    weights = torch.exp(-exponent_beta * distances)
    weights = torch.vstack([torch.hstack([weights, torch.ones((weights.shape[0], 1)).to(weights.device)]),
        torch.ones((1, weights.shape[1]+1)).to(weights.device)])
    # loss = -(weights * torch.log(soft_assignment_matrix)).sum() / weights.sum()
    loss = (-(weights * torch.log(soft_assignment_matrix)).mean()) # / weights.sum()) # / (weights.shape[0] * weights.shape[1])
    if False:
        print("Weights")
        print(weights)
        print("Loss")
        print(loss)
    return loss
