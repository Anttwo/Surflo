import torch
from surflo.utils.geometry import get_knn_index


@torch.no_grad()
def compute_chamfer_distance(
    points1: torch.Tensor, 
    points2: torch.Tensor, 
    center_normalize_points2: bool = False
) -> torch.Tensor:
    """
    Compute Chamfer Distance using KNN. 
    Faster for large point clouds and much better for memory.
    
    Args:
        points1 (torch.Tensor): (P1, 3)
        points2 (torch.Tensor): (P2, 3)
        center_normalize_points2 (bool): Center and normalize points2 before computing the Chamfer Distance.
        
    Returns:
        torch.Tensor: The Chamfer Distance between the two point clouds.
    """
    
    assert points1.ndim == 2
    assert points1.shape[-1] == 3
    
    if points1.numel() == 0 or points2.numel() == 0:
        return torch.tensor(float("nan"), device=points1.device)
    
    if center_normalize_points2:
        shift = points2.mean(dim=0)
        scale = points2.std(dim=0)
        points1 = (points1 - shift) / scale
        points2 = (points2 - shift) / scale

    knn_1_in_2 = get_knn_index(points=points1, points2=points2, k=1).squeeze(-1)  # (P1,)
    knn_2_in_1 = get_knn_index(points=points2, points2=points1, k=1).squeeze(-1)  # (P2,)
    
    dist_1_to_2 = torch.norm(points1 - points2[knn_1_in_2], dim=-1)  # (P1,)
    dist_2_to_1 = torch.norm(points2 - points1[knn_2_in_1], dim=-1)  # (P2,)
    
    return dist_1_to_2.mean() + dist_2_to_1.mean()  # ()


@torch.no_grad()
def compute_chamfer_distance_naive(
    points1: torch.Tensor, 
    points2: torch.Tensor, 
    center_normalize_points2: bool = False
) -> torch.Tensor:
    """
    Compute symmetric Chamfer Distance using direct cdist.
    Slower for large point clouds and much worse for memory.
    
    Args:
        points1 (torch.Tensor): (P1, 3)
        points2 (torch.Tensor): (P2, 3)
        center_normalize_points2 (bool): Center and normalize points2 before computing the Chamfer Distance.

    Returns:
        torch.Tensor: The Chamfer Distance between the two point clouds.
    """
    
    assert points1.ndim == 2
    assert points1.shape[-1] == 3
    
    if points1.numel() == 0 or points2.numel() == 0:
        return torch.tensor(float("nan"), device=points1.device)
    
    if center_normalize_points2:
        shift = points2.mean(dim=0)
        scale = points2.std(dim=0)
        points1 = (points1 - shift) / scale
        points2 = (points2 - shift) / scale

    cd_matrix = torch.cdist(points1[None], points2[None], p=2.0)  # (1, P1, P2)
    return cd_matrix.min(dim=1).values.mean() + cd_matrix.min(dim=2).values.mean()
