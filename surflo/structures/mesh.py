from typing import List, Union, Tuple, Optional
import torch


class Meshes(torch.nn.Module):
    """
    Meshes class for storing meshes parameters.
    """
    def __init__(
        self, 
        verts:torch.Tensor, 
        faces:torch.Tensor, 
        verts_colors:torch.Tensor=None
    ):
        super().__init__()
        assert verts_colors is None or verts_colors.shape[0] == verts.shape[0]
        self.verts = verts
        self.faces = faces.to(torch.int32)
        self.verts_colors = verts_colors
        self._edges = None
        self._faces_to_edges = None
        
    @property
    def device(self):
        return self.verts.device
        
    @property
    def face_normals(self):
        faces_verts = self.verts[self.faces]  # (F, 3, 3)
        faces_verts_normals = torch.cross(
            faces_verts[:,1] - faces_verts[:,0],  # (F, 3)
            faces_verts[:,2] - faces_verts[:,0],  # (F, 3)
            dim=-1
        )  # (F, 3)
        faces_verts_normals = torch.nn.functional.normalize(faces_verts_normals, dim=-1)  # (F, 3)
        return faces_verts_normals
    
    @property
    def vertex_normals(self):
        """Compute the vertex normals.
        Vertex normals are computed as the sum of the normals of all the faces it is part of,
        weighted by the face areas.
        """
        verts_normals = torch.zeros_like(self.verts)  # (V, 3)
        vertices_faces = self.verts[self.faces]  # (F, 3, 3)
        
        # Unnormalized faces normals.
        # Their magnitude is 2 x area of the triangle.
        faces_normals = torch.cross(
            vertices_faces[:, 2] - vertices_faces[:, 1],  # (F, 3)
            vertices_faces[:, 0] - vertices_faces[:, 1],  # (F, 3)
            dim=1,
        )  # (F, 3)
        
        # Add the faces normals to the verts normals
        verts_normals = verts_normals.index_add(
            0, self.faces[:, 0], faces_normals
        )
        verts_normals = verts_normals.index_add(
            0, self.faces[:, 1], faces_normals
        )
        verts_normals = verts_normals.index_add(
            0, self.faces[:, 2], faces_normals
        )
        
        # Normalize the verts normals
        return torch.nn.functional.normalize(
            verts_normals, eps=1e-6, dim=1
        )  # (V, 3)
    
    @property
    def edges(self):
        # Inspired from PyTorch3D: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/structures/meshes.html
        if self._edges is None:
            F = self.faces.shape[0]
            v0, v1, v2 = self.faces.chunk(3, dim=1)
            e01 = torch.cat([v0, v1], dim=1)  # (F, 2)
            e12 = torch.cat([v1, v2], dim=1)  # (F, 2)
            e20 = torch.cat([v2, v0], dim=1)  # (F, 2)
            
            # All edges including duplicates
            edges = torch.cat([e12, e20, e01], dim=0)  # (3 * F, 2)
            
            # Sort the edges in increasing vertex order to better identify duplicates
            edges, _ = edges.sort(dim=1)  # (3 * F, 2)
            
            # TODO: WARNING: Cast to long to avoid overflows
            edges = edges.to(torch.int64)
            
            # Remove duplicate edges: convert each edge (v0, v1) into an
            # integer hash = V * v0 + v1; this allows us to use the scalar version of
            # unique which is much faster than edges.unique(dim=1) which is very slow.
            # After finding the unique elements reconstruct the vertex indices as:
            # (v0, v1) = (hash / V, hash % V)
            # The inverse maps from unique_edges back to edges:
            # unique_edges[inverse_idxs] == edges
            # i.e. inverse_idxs[i] == j means that edges[i] == unique_edges[j]
            V = self.verts.shape[0]
            edges_hash = V * edges[:, 0] + edges[:, 1]  # (3 * F, )
            u, inverse_idxs = torch.unique(edges_hash, return_inverse=True)
            
            self._edges = torch.stack([u // V, u % V], dim=1)  # (E, 2)                        
            self._faces_to_edges = inverse_idxs.reshape(3, F).t()  # (F, 3)
            
            # TODO: WARNING: Cast back to int32
            self._edges = self._edges.to(torch.int32)
        
        return self._edges
    
    @property
    def faces_to_edges(self):
        if self._faces_to_edges is None:
            _ = self.edges  # Compute edges if not already computed
        return self._faces_to_edges
    
    @property
    def laplacian(self):
        """
        Inspired from PyTorch3D: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/laplacian_matrices.html
        
        Computes the laplacian matrix.
        The definition of the laplacian is
        L[i, j] =    -1       , if i == j
        L[i, j] = 1 / deg(i)  , if (i, j) is an edge
        L[i, j] =    0        , otherwise
        where deg(i) is the degree of the i-th vertex in the graph.

        Args:
            verts: tensor of shape (V, 3) containing the vertices of the graph
            edges: tensor of shape (E, 2) containing the vertex indices of each edge
        Returns:
            L: Sparse FloatTensor of shape (V, V)
        """
        V = self.verts.shape[0]

        e0, e1 = self.edges.unbind(1)

        idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
        idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
        idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

        # torch.sparse.check_sparse_tensor_invariants.enable()

        # First, we construct the adjacency matrix,
        # i.e. A[i, j] = 1 if (i,j) is an edge, or
        # A[e0, e1] = 1 &  A[e1, e0] = 1
        ones = torch.ones(idx.shape[1], dtype=torch.float32, device=self.device)
        A = torch.sparse_coo_tensor(idx, ones, (V, V), dtype=torch.float32)

        # the sum of i-th row of A gives the degree of the i-th vertex
        deg = torch.sparse.sum(A, dim=1).to_dense()

        # We construct the Laplacian matrix by adding the non diagonal values
        # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
        deg0 = deg[e0]
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
        deg1 = deg[e1]
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
        val = torch.cat([deg0, deg1])
        L = torch.sparse_coo_tensor(idx, val, (V, V), dtype=torch.float32)

        # Then we add the diagonal values L[i, i] = -1.
        idx = torch.arange(V, device=self.device)
        idx = torch.stack([idx, idx], dim=0)
        ones = torch.ones(idx.shape[1], dtype=torch.float32, device=self.device)
        L -= torch.sparse_coo_tensor(idx, ones, (V, V), dtype=torch.float32)

        return L 
    
    def cotangent_laplacian(self, eps:float=1e-12):
        """
        Inspired from PyTorch3D: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/laplacian_matrices.html
        
        Returns the Laplacian matrix with cotangent weights and the inverse of the
        face areas.

        Args:
            verts: tensor of shape (V, 3) containing the vertices of the graph
            faces: tensor of shape (F, 3) containing the vertex indices of each face
        Returns:
            2-element tuple containing
            - **L**: Sparse FloatTensor of shape (V,V) for the Laplacian matrix.
            Here, L[i, j] = cot a_ij + cot b_ij iff (i, j) is an edge in meshes.
            See the description above for more clarity.
            - **inv_areas**: FloatTensor of shape (V,) containing the inverse of sum of
            face areas containing each vertex
        """
        verts = self.verts
        faces = self.faces
        V, F = verts.shape[0], faces.shape[0]

        face_verts = verts[faces]
        v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

        # Side lengths of each triangle, of shape (F,)
        # A is the side opposite v1, B is opposite v2, and C is opposite v3
        A = (v1 - v2).norm(dim=1)
        B = (v0 - v2).norm(dim=1)
        C = (v0 - v1).norm(dim=1)

        # Area of each triangle (with Heron's formula); shape is (F,)
        s = 0.5 * (A + B + C)
        # note that the area can be negative (close to 0) causing nans after sqrt()
        # we clip it to a small positive value
        # pyre-fixme[16]: `float` has no attribute `clamp_`.
        area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=eps).sqrt()

        # Compute cotangents of angles, of shape (sum(F_n), 3)
        A2, B2, C2 = A * A, B * B, C * C
        cota = (B2 + C2 - A2) / area
        cotb = (A2 + C2 - B2) / area
        cotc = (A2 + B2 - C2) / area
        cot = torch.stack([cota, cotb, cotc], dim=1)
        cot /= 4.0

        # Construct a sparse matrix by basically doing:
        # L[v1, v2] = cota
        # L[v2, v0] = cotb
        # L[v0, v1] = cotc
        ii = faces[:, [1, 2, 0]]
        jj = faces[:, [2, 0, 1]]
        idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
        L = torch.sparse_coo_tensor(idx, cot.view(-1), (V, V), dtype=torch.float32)

        # Make it symmetric; this means we are also setting
        # L[v2, v1] = cota
        # L[v0, v2] = cotb
        # L[v1, v0] = cotc
        L += L.t()

        # For each vertex, compute the sum of areas for triangles containing it.
        idx = faces.view(-1).to(torch.int64)
        inv_areas = torch.zeros(V, dtype=torch.float32, device=verts.device)
        val = torch.stack([area] * 3, dim=1).view(-1)
        inv_areas.scatter_add_(0, idx, val)
        idx = inv_areas > 0
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        inv_areas[idx] = 1.0 / inv_areas[idx]
        inv_areas = inv_areas.view(-1, 1)

        return L, inv_areas
    
    def submesh(
        self, 
        vert_idx:Optional[torch.Tensor]=None, 
        face_idx:Optional[torch.Tensor]=None,
        vert_mask:Optional[torch.Tensor]=None,
        face_mask:Optional[torch.Tensor]=None,
    ):
        assert (
            (vert_idx is not None) or (vert_mask is not None) 
            or
            (face_idx is not None) or (face_mask is not None)
        ), "Either vert_idx, vert_mask, face_idx, or face_mask must be provided"

        if (vert_idx is not None) or (vert_mask is not None):
            if vert_mask is None:
                vert_mask = torch.zeros(self.verts.shape[0], dtype=torch.bool, device=self.verts.device)
                vert_mask[vert_idx] = True
            face_mask = vert_mask[self.faces].all(dim=1)

        elif (face_idx is not None) or (face_mask is not None):
            if face_mask is None:
                face_mask = torch.zeros(self.faces.shape[0], dtype=torch.bool, device=self.verts.device)
                face_mask[face_idx] = True
            vert_mask = torch.zeros(self.verts.shape[0], dtype=torch.bool, device=self.verts.device)
            vert_mask[self.faces[face_mask]] = True
        
        old_vert_idx_to_new_vert_idx = torch.zeros(self.verts.shape[0], dtype=self.faces.dtype, device=self.verts.device)
        old_vert_idx_to_new_vert_idx[vert_mask] = torch.arange(vert_mask.sum(), dtype=self.faces.dtype, device=self.verts.device)
        
        new_verts = self.verts[vert_mask]
        new_verts_colors = None if self.verts_colors is None else self.verts_colors[vert_mask]
        new_faces = old_vert_idx_to_new_vert_idx[self.faces][face_mask]
        
        return Meshes(verts=new_verts, faces=new_faces, verts_colors=new_verts_colors)


def combine_meshes(
    meshes:List[Meshes],
) -> Meshes:
    """Combines multiple meshes into a single mesh.

    Args:
        meshes (List[Meshes]): List of meshes to combine.

    Returns:
        Meshes: Combined mesh.
    """
    all_verts = torch.zeros(0, 3, dtype=meshes[0].verts.dtype, device=meshes[0].verts.device)
    all_faces = torch.zeros(0, 3, dtype=meshes[0].faces.dtype, device=meshes[0].faces.device)
    all_verts_colors = None if meshes[0].verts_colors is None else torch.zeros(0, 3, dtype=meshes[0].verts_colors.dtype, device=meshes[0].verts_colors.device)
    
    n_total_verts = 0
    
    for _, mesh in enumerate(meshes):
        all_verts = torch.cat([all_verts, mesh.verts], dim=0)
        all_faces = torch.cat([all_faces, mesh.faces + n_total_verts], dim=0)
        if all_verts_colors is not None:
            all_verts_colors = torch.cat([all_verts_colors, mesh.verts_colors], dim=0)
        n_total_verts += mesh.verts.shape[0]
    
    return Meshes(verts=all_verts, faces=all_faces, verts_colors=all_verts_colors)


def laplacian_smoothing_loss(
    mesh:Meshes, 
    method:str="uniform",
    reduce:bool=True,
) -> torch.Tensor:
    """Computes the Laplacian smoothing loss for a mesh.

    Args:
        mesh (Meshes): The mesh to compute the Laplacian smoothing loss for.
        method (str, optional): The method to use for the Laplacian smoothing loss.
            Defaults to "uniform", can also be "cot" or "cotcurv".
        reduce (bool, optional): Whether to reduce the loss to a scalar.
            Defaults to True.

    Raises:
        ValueError: If the method is not one of "uniform", "cot", or "cotcurv".

    Returns:
        torch.Tensor: The Laplacian smoothing loss.
    """
    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals
    with torch.no_grad():
        if method == "uniform":
            L = mesh.laplacian
        elif method in ["cot", "cotcurv"]:
            L, inv_areas = mesh.cotangent_laplacian()
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                # pyre-fixme[58]: `/` is not supported for operand types `float` and
                #  `Tensor`.
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                norm_w = 0.25 * inv_areas
        else:
            raise ValueError("Method should be one of {uniform, cot, cotcurv}")

    if method == "uniform":
        loss = L.mm(mesh.verts)
    elif method == "cot":
        # pyre-fixme[61]: `norm_w` is undefined, or not always defined.
        loss = L.mm(mesh.verts) * norm_w - mesh.verts
    elif method == "cotcurv":
        # pyre-fixme[61]: `norm_w` may not be initialized here.
        loss = (L.mm(mesh.verts) - L_sum * mesh.verts) * norm_w
    
    loss = loss.norm(dim=1)
    return loss.mean() if reduce else loss


def normal_consistency_loss(
    mesh:Meshes,
) -> torch.Tensor:
    """Computes the normal consistency loss for a mesh.
    
    Args:
        mesh (Meshes): The mesh to compute the normal consistency loss for.
        
    Returns:
        torch.Tensor: The normal consistency loss.
    """
    if False:
        edge_verts_normals = mesh.vertex_normals[mesh.edges]  # Shape (E, 2, 3)
        assert (
            edge_verts_normals.shape[0] == mesh.edges.shape[0]
            and edge_verts_normals.shape[1] == 2
            and edge_verts_normals.shape[2] == 3
        ), "edge_verts_normals should be of shape (E, 2, 3)"
        
        edge_verts_normals_dot_product = (edge_verts_normals[..., 0] * edge_verts_normals[..., 1]).sum(dim=-1)
        return (1. - edge_verts_normals_dot_product).mean()
    
    if True:
        face_normals = mesh.face_normals[:, None, :]  # Shape (F, 1, 3)
        face_verts_normals = mesh.vertex_normals[mesh.faces]  # Shape (F, 3, 3)
        face_verts_normals_dot_product = (face_normals * face_verts_normals).sum(dim=-1)  # Shape (F, 3)
        return (1. - face_verts_normals_dot_product).mean()
    
    
def get_error_quadrics(mesh: Meshes, normalization_factor: float = 100.):
    """
    Compute the error quadrics for vertices of a mesh.
    """
    
    verts = mesh.verts / normalization_factor  # (V, 3)
    
    # Compute the fundamental quadrics for each face
    face_centers = verts[mesh.faces].mean(dim=-2)  # (F, 3)
    face_normals = mesh.face_normals  # (F, 3)
    face_centers_proj = (face_normals * face_centers).sum(dim=-1, keepdim=True)  # (F, 1)
    face_planes = torch.cat(
        [
            face_normals, 
            -face_centers_proj,
        ], 
        dim=-1
    )  # (F, 4)
    fundamental_quadrics = face_planes[:, :, None] @ face_planes[:, None, :]  # (F, 4, 4)
    
    # Add the fundamental face quadrics to the verts error quadrics
    error_quadrics = torch.zeros(verts.shape[0], 4, 4, device=mesh.device)  # (V, 4, 4)
    error_quadrics = error_quadrics.index_add(
        0, mesh.faces[:, 0], fundamental_quadrics
    )
    error_quadrics = error_quadrics.index_add(
        0, mesh.faces[:, 1], fundamental_quadrics
    )
    error_quadrics = error_quadrics.index_add(
        0, mesh.faces[:, 2], fundamental_quadrics
    )
    
    return error_quadrics


def get_contraction_points(
    verts: torch.Tensor, 
    edges: torch.Tensor, 
    error_quadrics: torch.Tensor,
    return_edge_quadrics: bool = False,
):
    # Compute the edge quadrics
    edge_quadrics = error_quadrics[edges[:, 0]] + error_quadrics[edges[:, 1]]  # (E, 4, 4)
    
    M = torch.zeros_like(edge_quadrics)
    M[:, :3, :4] = edge_quadrics[:, :3, :4]
    M[:, 3, 3] = 1.
    
    b = torch.zeros(edges.shape[0], 4, device=error_quadrics.device)
    b[:, 3] = 1.
    
    # compute solvable mask
    solvable_mask = torch.linalg.det(M).abs() > 0.
    
    # Solve the linear system Mv = b
    v = torch.zeros(edges.shape[0], 4, device=error_quadrics.device)
    v[solvable_mask] = torch.linalg.solve(M[solvable_mask], b[solvable_mask])
    
    return (v[..., :3], edge_quadrics) if return_edge_quadrics else v[..., :3]
