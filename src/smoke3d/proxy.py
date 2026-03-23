"""
Proxy Mesh module for Proxy-Aware Smoke-GS.

Inspired by Proxy-GS (CVPR 2026), adapted for smoke restoration:
- Constructs lightweight proxy mesh from DUSt3R pointmap or COLMAP
- Renders proxy depth maps for occlusion awareness
- Provides surface points for guided densification
- Estimates smoke density for weighted training

Key differences from original Proxy-GS:
1. We don't need Vulkan-CUDA interop (small scenes, can pre-compute)
2. We add smoke-density awareness
3. We use PyTorch3D/nvdiffrast instead of hardware rasterization
4. Our proxy also helps identify unreliable pseudo-clean regions
"""

from __future__ import annotations

import os
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class ProxyMesh:
    """
    Lightweight proxy mesh for occlusion-aware 3DGS training.
    
    Usage:
        proxy = ProxyMesh.from_dust3r(pointmap_path, ...)
        # or
        proxy = ProxyMesh.from_colmap(colmap_path, ...)
        
        # During training:
        depth_map = proxy.render_depth(c2w, K, H, W)
        surface_pts = proxy.get_surface_points(uv, depth_map, K, c2w)
    """
    
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        device: str = "cuda",
        prefer_simple_renderer: bool = False,
    ):
        """
        Args:
            vertices: (N, 3) mesh vertices in world coordinates
            faces: (M, 3) triangle indices
            device: torch device
        """
        self.vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
        self.faces = torch.tensor(faces, dtype=torch.int64, device=device)
        self.device = device
        self.prefer_simple_renderer = bool(prefer_simple_renderer)
        
        # Pre-computed depth maps cache (view_idx -> depth_tensor)
        self._depth_cache: dict[int, torch.Tensor] = {}
        
        # Renderer (lazy init)
        self._renderer = None
        self._renderer_type = None

    @staticmethod
    def _load_pointmap_payload(pointmap_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        with np.load(pointmap_path) as payload:
            if "points" in payload.files:
                points = payload["points"]
                valid_mask = payload.get("valid_mask")
                confidences = payload.get("confidences")
                if valid_mask is None:
                    valid_mask = np.isfinite(points).all(axis=-1).astype(np.uint8)
                return points, valid_mask.astype(bool), confidences

            # Fallback for raw pointmap dumps.
            if "pts3d_world" in payload.files:
                points = payload["pts3d_world"]
            else:
                raise ValueError(f"Unsupported pointmap npz payload: {pointmap_path}")
            valid_mask = np.isfinite(points).all(axis=-1)
            return points, valid_mask.astype(bool), None

    @classmethod
    def from_pointmap_directory(
        cls,
        pointmap_dir: str,
        frame_names: list[str] | None = None,
        stride: int = 8,
        confidence_threshold: float = 0.0,
        edge_threshold_scale: float = 4.0,
        max_views: int = 0,
        device: str = "cuda",
        prefer_simple_renderer: bool = True,
    ) -> "ProxyMesh":
        """Construct a lightweight proxy mesh directly from per-view pointmap npz files.

        This avoids expensive Poisson reconstruction and stays close to the available
        DUSt3R/VGGT supervision format used by the current project.
        """
        pointmap_root = Path(pointmap_dir)
        if not pointmap_root.exists():
            raise FileNotFoundError(f"Pointmap directory not found: {pointmap_root}")

        allowed_names = None
        if frame_names:
            allowed_names = {
                name if name.endswith(".npz") else f"{name}.npz"
                for name in frame_names
            }

        pointmap_paths = sorted(pointmap_root.glob("*.npz"))
        if allowed_names is not None:
            pointmap_paths = [path for path in pointmap_paths if path.name in allowed_names]
        if max_views > 0:
            pointmap_paths = pointmap_paths[:max_views]
        if not pointmap_paths:
            raise FileNotFoundError(f"No pointmap npz files found under: {pointmap_root}")

        vertices_list: list[np.ndarray] = []
        faces_list: list[np.ndarray] = []
        vertex_offset = 0
        total_views_used = 0

        for pointmap_path in pointmap_paths:
            points, valid_mask, confidences = cls._load_pointmap_payload(str(pointmap_path))
            view_vertices, view_faces = cls._grid_mesh_from_pointmap(
                points=points,
                valid_mask=valid_mask,
                confidences=confidences,
                stride=stride,
                confidence_threshold=confidence_threshold,
                edge_threshold_scale=edge_threshold_scale,
            )
            if len(view_vertices) == 0 or len(view_faces) == 0:
                continue
            vertices_list.append(view_vertices)
            faces_list.append(view_faces + vertex_offset)
            vertex_offset += len(view_vertices)
            total_views_used += 1

        if not vertices_list or not faces_list:
            raise RuntimeError(
                f"Failed to construct proxy mesh from pointmaps under {pointmap_root}"
            )

        vertices = np.concatenate(vertices_list, axis=0)
        faces = np.concatenate(faces_list, axis=0)
        print(
            "[ProxyMesh] Built from pointmap directory: {} views, {} vertices, {} faces".format(
                total_views_used,
                len(vertices),
                len(faces),
            )
        )
        return cls(
            vertices,
            faces,
            device=device,
            prefer_simple_renderer=prefer_simple_renderer,
        )

    @staticmethod
    def _estimate_edge_threshold(
        points: np.ndarray,
        valid_mask: np.ndarray,
        edge_threshold_scale: float,
    ) -> float:
        horizontal_valid = valid_mask[:, :-1] & valid_mask[:, 1:]
        vertical_valid = valid_mask[:-1, :] & valid_mask[1:, :]
        distances: list[np.ndarray] = []
        if horizontal_valid.any():
            distances.append(
                np.linalg.norm(points[:, 1:] - points[:, :-1], axis=-1)[horizontal_valid]
            )
        if vertical_valid.any():
            distances.append(
                np.linalg.norm(points[1:, :] - points[:-1, :], axis=-1)[vertical_valid]
            )
        if not distances:
            return float("inf")
        all_distances = np.concatenate(distances, axis=0)
        median_distance = float(np.median(all_distances))
        if not np.isfinite(median_distance) or median_distance <= 0:
            return float("inf")
        return median_distance * float(edge_threshold_scale)

    @classmethod
    def _grid_mesh_from_pointmap(
        cls,
        points: np.ndarray,
        valid_mask: np.ndarray,
        confidences: np.ndarray | None,
        stride: int,
        confidence_threshold: float,
        edge_threshold_scale: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        stride = max(1, int(stride))
        sampled_points = points[::stride, ::stride]
        sampled_valid = valid_mask[::stride, ::stride].astype(bool)
        if confidences is not None:
            sampled_conf = confidences[::stride, ::stride]
            sampled_valid &= sampled_conf >= float(confidence_threshold)

        sampled_valid &= np.isfinite(sampled_points).all(axis=-1)
        if not sampled_valid.any():
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)

        vertex_ids = -np.ones(sampled_valid.shape, dtype=np.int64)
        vertices = sampled_points[sampled_valid].astype(np.float32)
        vertex_ids[sampled_valid] = np.arange(len(vertices), dtype=np.int64)

        v00 = vertex_ids[:-1, :-1]
        v01 = vertex_ids[:-1, 1:]
        v10 = vertex_ids[1:, :-1]
        v11 = vertex_ids[1:, 1:]

        edge_threshold = cls._estimate_edge_threshold(
            sampled_points,
            sampled_valid,
            edge_threshold_scale=edge_threshold_scale,
        )

        if np.isfinite(edge_threshold):
            p00 = sampled_points[:-1, :-1]
            p01 = sampled_points[:-1, 1:]
            p10 = sampled_points[1:, :-1]
            p11 = sampled_points[1:, 1:]
            diag_01 = np.linalg.norm(p00 - p11, axis=-1) <= edge_threshold
            diag_10 = np.linalg.norm(p01 - p10, axis=-1) <= edge_threshold
            edge_ok = (
                (np.linalg.norm(p00 - p01, axis=-1) <= edge_threshold)
                & (np.linalg.norm(p00 - p10, axis=-1) <= edge_threshold)
                & (np.linalg.norm(p01 - p11, axis=-1) <= edge_threshold)
                & (np.linalg.norm(p10 - p11, axis=-1) <= edge_threshold)
            )
        else:
            diag_01 = np.ones_like(v00, dtype=bool)
            diag_10 = np.ones_like(v00, dtype=bool)
            edge_ok = np.ones_like(v00, dtype=bool)

        tri1_mask = (v00 >= 0) & (v01 >= 0) & (v11 >= 0) & edge_ok & diag_01
        tri2_mask = (v00 >= 0) & (v11 >= 0) & (v10 >= 0) & edge_ok & diag_10

        faces = []
        if tri1_mask.any():
            faces.append(
                np.stack([v00[tri1_mask], v01[tri1_mask], v11[tri1_mask]], axis=-1)
            )
        if tri2_mask.any():
            faces.append(
                np.stack([v00[tri2_mask], v11[tri2_mask], v10[tri2_mask]], axis=-1)
            )
        if not faces:
            return vertices, np.zeros((0, 3), dtype=np.int64)
        return vertices, np.concatenate(faces, axis=0).astype(np.int64)
    
    @classmethod
    def from_dust3r(
        cls,
        pointmap_path: str,
        simplify_ratio: float = 0.1,
        depth_threshold: float = 50.0,
        device: str = "cuda",
        prefer_simple_renderer: bool = False,
    ) -> "ProxyMesh":
        """
        Construct proxy mesh from DUSt3R pointmap.
        
        Args:
            pointmap_path: path to DUSt3R output (pts3d_world .npy or .pt)
            simplify_ratio: target ratio for mesh simplification (0.1 = keep 10%)
            depth_threshold: max depth for filtering outliers
            device: torch device
        """
        import open3d as o3d
        
        # Load pointmap
        if pointmap_path.endswith('.npz'):
            pts, valid_mask, _ = cls._load_pointmap_payload(pointmap_path)
            pts = pts[valid_mask]
        elif pointmap_path.endswith('.npy'):
            pts = np.load(pointmap_path)
        elif pointmap_path.endswith('.pt') or pointmap_path.endswith('.pth'):
            pts = torch.load(pointmap_path, map_location='cpu')
            if isinstance(pts, torch.Tensor):
                pts = pts.numpy()
        else:
            raise ValueError(f"Unsupported pointmap format: {pointmap_path}")
        
        # Reshape if needed (H, W, 3) -> (N, 3)
        if pts.ndim == 3:
            pts = pts.reshape(-1, 3)
        
        # Filter outliers
        mask = np.linalg.norm(pts, axis=-1) < depth_threshold
        pts = pts[mask]
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
        
        # Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8, width=0, scale=1.1, linear_fit=False
        )
        
        # Remove low-density vertices (noise)
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.05)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # QEM mesh simplification (core idea from Proxy-GS)
        target_faces = int(len(mesh.triangles) * simplify_ratio)
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        print(f"[ProxyMesh] Built from DUSt3R: {len(vertices)} vertices, {len(faces)} faces")
        return cls(vertices, faces, device, prefer_simple_renderer=prefer_simple_renderer)
    
    @classmethod
    def from_colmap(
        cls,
        colmap_path: str,
        simplify_ratio: float = 0.1,
        device: str = "cuda",
        prefer_simple_renderer: bool = False,
    ) -> "ProxyMesh":
        """
        Construct proxy mesh from COLMAP dense reconstruction.
        
        Args:
            colmap_path: path to COLMAP workspace (with dense/fused.ply)
            simplify_ratio: target ratio for mesh simplification
            device: torch device
        """
        import open3d as o3d
        
        ply_path = os.path.join(colmap_path, "dense", "fused.ply")
        if not os.path.exists(ply_path):
            # Try alternative paths
            for alt in ["fused.ply", "points3D.ply", "dense/0/fused.ply"]:
                alt_path = os.path.join(colmap_path, alt)
                if os.path.exists(alt_path):
                    ply_path = alt_path
                    break
        
        pcd = o3d.io.read_point_cloud(ply_path)
        
        # Estimate normals if not present
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)
        
        # Poisson reconstruction + simplification
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8
        )
        
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.05)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        target_faces = int(len(mesh.triangles) * simplify_ratio)
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        print(f"[ProxyMesh] Built from COLMAP: {len(vertices)} vertices, {len(faces)} faces")
        return cls(vertices, faces, device, prefer_simple_renderer=prefer_simple_renderer)
    
    @classmethod
    def from_mesh_file(
        cls,
        mesh_path: str,
        simplify_ratio: float = 1.0,
        device: str = "cuda",
        prefer_simple_renderer: bool = False,
    ):
        """Load from existing mesh file (.ply / .obj)."""
        import open3d as o3d
        
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        
        if simplify_ratio < 1.0:
            target_faces = int(len(mesh.triangles) * simplify_ratio)
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
            mesh.remove_degenerate_triangles()
            mesh.remove_unreferenced_vertices()
        
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        print(f"[ProxyMesh] Loaded from file: {len(vertices)} vertices, {len(faces)} faces")
        return cls(vertices, faces, device, prefer_simple_renderer=prefer_simple_renderer)
    
    def _init_renderer(self, H: int, W: int):
        """Initialize mesh renderer (PyTorch3D or nvdiffrast)."""
        if self.prefer_simple_renderer:
            self._renderer_type = "simple"
            self._renderer = "simple"
            print("[ProxyMesh] Using simple z-buffer renderer (forced)")
            return
        try:
            # Try nvdiffrast first (faster, lighter)
            import nvdiffrast.torch as dr
            self._renderer_type = "nvdiffrast"
            self._glctx = dr.RasterizeCudaContext()
            print("[ProxyMesh] Using nvdiffrast renderer")
        except ImportError:
            try:
                # Fallback to PyTorch3D
                from pytorch3d.renderer import (
                    RasterizationSettings,
                    MeshRasterizer,
                )
                from pytorch3d.structures import Meshes
                self._renderer_type = "pytorch3d"
                print("[ProxyMesh] Using PyTorch3D renderer")
            except ImportError:
                # Ultra-fallback: simple z-buffer in pure PyTorch
                self._renderer_type = "simple"
                print("[ProxyMesh] Using simple z-buffer renderer (slow)")
    
    def render_depth(
        self,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        H: int,
        W: int,
        near: float = 0.01,
        far: float = 100.0,
    ) -> torch.Tensor:
        """
        Render proxy depth map from a given camera pose.
        
        Args:
            c2w: (4, 4) camera-to-world matrix
            intrinsics: (3, 3) or (4,) camera intrinsics [fx, fy, cx, cy]
            H, W: image dimensions
            near, far: clipping planes
            
        Returns:
            depth: (H, W) depth map in camera space, 0 for invalid pixels
        """
        if self._renderer is None:
            self._init_renderer(H, W)

        if c2w.shape == (3, 4):
            pose = torch.eye(4, dtype=torch.float32, device=self.device)
            pose[:3, :] = c2w.to(device=self.device, dtype=torch.float32)
            c2w = pose
        else:
            c2w = c2w.to(device=self.device, dtype=torch.float32)
        
        # Convert c2w to w2c
        w2c = torch.inverse(c2w)  # (4, 4)
        
        # Transform vertices to camera space
        verts_homo = torch.cat([
            self.vertices,
            torch.ones(len(self.vertices), 1, device=self.device)
        ], dim=-1)  # (N, 4)
        
        verts_cam = (w2c @ verts_homo.T).T[:, :3]  # (N, 3)
        
        if self._renderer_type == "nvdiffrast":
            return self._render_depth_nvdiffrast(verts_cam, intrinsics, H, W, near, far)
        elif self._renderer_type == "pytorch3d":
            return self._render_depth_pytorch3d(c2w, intrinsics, H, W, near, far)
        else:
            return self._render_depth_simple(verts_cam, intrinsics, H, W, near, far)

    def _project_gl_vertices(
        self,
        verts_cam: torch.Tensor,
        intrinsics: torch.Tensor,
        H: int,
        W: int,
        near: float,
        far: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if intrinsics.dim() == 1:
            fx, fy, cx, cy = intrinsics
        else:
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        depth = -verts_cam[:, 2]
        valid = torch.isfinite(depth) & (depth > near) & (depth < far)
        if not valid.any():
            empty = torch.zeros(0, device=self.device, dtype=torch.long)
            return empty, empty, torch.zeros(0, device=self.device)

        verts_cam = verts_cam[valid]
        depth = depth[valid]
        px = (fx * verts_cam[:, 0] / depth + cx).round().long()
        py = (fy * (-verts_cam[:, 1]) / depth + cy).round().long()
        valid_px = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        return px[valid_px], py[valid_px], depth[valid_px]
    
    def _render_depth_nvdiffrast(
        self, verts_cam, intrinsics, H, W, near, far
    ) -> torch.Tensor:
        """Render depth using nvdiffrast."""
        import nvdiffrast.torch as dr
        
        # Parse intrinsics
        if intrinsics.dim() == 1:
            fx, fy, cx, cy = intrinsics
        else:
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Build projection matrix (OpenGL convention)
        proj = torch.zeros(4, 4, device=self.device)
        proj[0, 0] = 2 * fx / W
        proj[1, 1] = 2 * fy / H
        proj[0, 2] = 1 - 2 * cx / W
        proj[1, 2] = 2 * cy / H - 1
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2 * far * near / (far - near)
        proj[3, 2] = -1
        
        # Transform to clip space
        verts_clip = torch.cat([
            verts_cam,
            torch.ones(len(verts_cam), 1, device=self.device)
        ], dim=-1)
        verts_clip = (proj @ verts_clip.T).T
        
        # Rasterize
        verts_clip = verts_clip.unsqueeze(0).contiguous()
        faces = self.faces.int().unsqueeze(0).contiguous()
        
        rast_out, _ = dr.rasterize(self._glctx, verts_clip, faces[0], resolution=[H, W])
        
        # Extract depth from rasterization
        # rast_out[..., 2] contains the interpolated depth
        depth = rast_out[0, :, :, 2]
        
        # Convert from NDC depth to linear camera-space depth
        mask = rast_out[0, :, :, 3] > 0  # valid triangles
        depth_linear = torch.zeros(H, W, device=self.device)
        if mask.any():
            ndc_depth = depth[mask]
            # From NDC [-1, 1] to linear depth
            depth_linear[mask] = 2 * near * far / (far + near - ndc_depth * (far - near))
        
        return depth_linear
    
    def _render_depth_simple(
        self, verts_cam, intrinsics, H, W, near, far
    ) -> torch.Tensor:
        """Simple z-buffer depth rendering (pure PyTorch, uses projected vertices)."""
        px, py, z = self._project_gl_vertices(verts_cam, intrinsics, H, W, near, far)
        if px.numel() == 0:
            return torch.zeros(H, W, device=self.device)

        flat_idx = py * W + px
        depth_flat = torch.full((H * W,), float(far), device=self.device)
        try:
            depth_flat.scatter_reduce_(0, flat_idx, z, reduce="amin", include_self=True)
        except RuntimeError:
            order = torch.argsort(z)
            for idx, depth_value in zip(flat_idx[order], z[order]):
                cur = depth_flat[idx]
                if depth_value < cur:
                    depth_flat[idx] = depth_value

        depth = depth_flat.view(H, W)
        depth[depth >= far] = 0.0
        valid = depth > 0
        if valid.any():
            approx_spacing = math.sqrt((H * W) / max(1, int(valid.sum().item())))
            fill_radius = min(8, max(1, int(round(0.35 * approx_spacing))))
            if fill_radius > 0:
                far_tensor = torch.full_like(depth, float(far))
                depth_dense = torch.where(valid, depth, far_tensor)
                kernel = 2 * fill_radius + 1
                depth_dense = -F.max_pool2d(
                    -depth_dense[None, None],
                    kernel_size=kernel,
                    stride=1,
                    padding=fill_radius,
                )[0, 0]
                valid_dense = F.max_pool2d(
                    valid.float()[None, None],
                    kernel_size=kernel,
                    stride=1,
                    padding=fill_radius,
                )[0, 0] > 0
                depth = torch.where(valid_dense, depth_dense, torch.zeros_like(depth_dense))
        return depth
    
    def _render_depth_pytorch3d(
        self, c2w, intrinsics, H, W, near, far
    ) -> torch.Tensor:
        """Render depth using PyTorch3D."""
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import (
            RasterizationSettings,
            MeshRasterizer,
            PerspectiveCameras,
        )
        
        # Build camera
        R = c2w[:3, :3].unsqueeze(0)  # (1, 3, 3)
        T = c2w[:3, 3].unsqueeze(0)   # (1, 3)
        
        # Parse intrinsics
        if intrinsics.dim() == 1:
            fx, fy, cx, cy = intrinsics
        else:
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        focal = torch.tensor([[fx, fy]], device=self.device)
        principal = torch.tensor([[cx, cy]], device=self.device)
        
        cameras = PerspectiveCameras(
            R=R.transpose(1, 2),
            T=-torch.bmm(R.transpose(1, 2), T.unsqueeze(-1)).squeeze(-1),
            focal_length=focal,
            principal_point=principal,
            image_size=torch.tensor([[H, W]]),
            device=self.device,
        )
        
        # Build mesh
        meshes = Meshes(
            verts=[self.vertices],
            faces=[self.faces],
        )
        
        raster_settings = RasterizationSettings(
            image_size=(H, W),
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        )
        
        fragments = rasterizer(meshes)
        depth = fragments.zbuf[0, :, :, 0]
        depth[depth < 0] = 0.0
        
        return depth
    
    def precompute_depths(
        self,
        c2ws: list[torch.Tensor],
        intrinsics: torch.Tensor,
        H: int,
        W: int,
        near: float = 0.01,
        far: float = 100.0,
        cache_device: str = "cpu",
    ):
        """Pre-compute proxy depth maps for all training views."""
        print(f"[ProxyMesh] Pre-computing {len(c2ws)} depth maps...")
        for i, c2w in enumerate(c2ws):
            depth = self.render_depth(c2w, intrinsics, H, W, near=near, far=far)
            if cache_device == "cpu":
                depth = depth.detach().cpu()
            self._depth_cache[i] = depth
        print("[ProxyMesh] Done pre-computing depth maps.")
    
    def get_cached_depth(self, view_idx: int) -> torch.Tensor | None:
        """Get pre-computed depth map for a view."""
        return self._depth_cache.get(view_idx, None)
    
    def get_surface_points(
        self,
        uv_coords: torch.Tensor,
        depth_map: torch.Tensor,
        intrinsics: torch.Tensor,
        c2w: torch.Tensor,
    ) -> torch.Tensor:
        """
        Back-project 2D points to 3D surface using proxy depth.
        
        Following Proxy-GS Eq. for proxy-guided densification:
        Given high-error patches, project their centers to proxy surface.
        
        Args:
            uv_coords: (K, 2) pixel coordinates [u, v]
            depth_map: (H, W) proxy depth map
            intrinsics: camera intrinsics
            c2w: camera-to-world matrix
            
        Returns:
            world_points: (K, 3) 3D points on proxy surface
        """
        # Parse intrinsics
        if intrinsics.dim() == 1:
            fx, fy, cx, cy = intrinsics
        else:
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        u = uv_coords[:, 0]
        v = uv_coords[:, 1]
        
        # Sample depth at uv coordinates
        u_idx = u.long().clamp(0, depth_map.shape[1] - 1)
        v_idx = v.long().clamp(0, depth_map.shape[0] - 1)
        z = depth_map[v_idx, u_idx]
        
        # Back-project to camera space
        x_cam = (u - cx) * z / fx
        y_cam = -(v - cy) * z / fy
        pts_cam = torch.stack([x_cam, y_cam, -z], dim=-1)  # (K, 3) in OpenGL camera convention
        
        # Transform to world space
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        pts_world = (R @ pts_cam.T).T + t
        
        return pts_world
    
    def save(self, path: str):
        """Save proxy mesh to file."""
        import open3d as o3d
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices.cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(self.faces.cpu().numpy())
        o3d.io.write_triangle_mesh(path, mesh)
        print(f"[ProxyMesh] Saved to {path}")


def estimate_smoke_density(image: torch.Tensor, patch_size: int = 15) -> torch.Tensor:
    """
    Estimate per-pixel smoke density using Dark Channel Prior.
    
    Higher values = more smoke = less reliable pseudo-clean supervision.
    
    Args:
        image: (C, H, W) or (H, W, C) normalized image [0, 1]
        patch_size: DCP patch size
        
    Returns:
        smoke_density: (H, W) in [0, 1], higher = more smoke
    """
    if image.dim() == 3 and image.shape[0] <= 4:
        # (C, H, W) -> (H, W, C)
        image = image.permute(1, 2, 0)
    
    # Dark channel = min over channels, then min over patch
    dark = image.min(dim=-1).values  # (H, W)
    
    # Min pool (erosion)
    pad = patch_size // 2
    dark_padded = F.pad(dark.unsqueeze(0).unsqueeze(0), [pad] * 4, mode='reflect')
    dark_channel = -F.max_pool2d(-dark_padded, patch_size, stride=1)  # min pool
    dark_channel = dark_channel.squeeze()
    
    # Smoke density: 1 - dark_channel (clear areas have low dark channel)
    smoke_density = 1.0 - dark_channel
    
    return smoke_density.clamp(0, 1)


def compute_proxy_guided_densify_mask(
    patch_losses: torch.Tensor,
    proxy_depth: torch.Tensor,
    smoke_density: torch.Tensor,
    patch_size: int = 16,
    error_margin: float = 1.5,
    smoke_threshold: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute which patches should trigger densification, following Proxy-GS logic.
    
    Key adaptation for smoke scenes:
    - Original Proxy-GS: densify at high-error patches on proxy surface
    - Our adaptation: additionally suppress densification in high-smoke regions
    
    Args:
        patch_losses: (P_H, P_W) per-patch L1 loss
        proxy_depth: (H, W) proxy depth map
        smoke_density: (H, W) estimated smoke density
        patch_size: size of each patch
        error_margin: multiplier for mean error threshold
        smoke_threshold: don't densify if smoke density > this
        
    Returns:
        densify_mask: (P_H, P_W) bool, which patches to densify
        surface_uvs: (K, 2) center UV coords of selected patches
    """
    mean_loss = patch_losses.mean()
    
    # High-error patches (Proxy-GS criterion)
    high_error = patch_losses > error_margin * mean_loss
    
    # Compute per-patch smoke density
    H, W = smoke_density.shape
    P_H, P_W = patch_losses.shape
    
    # Average pool smoke density to patch level
    smoke_pooled = F.avg_pool2d(
        smoke_density.unsqueeze(0).unsqueeze(0),
        kernel_size=patch_size,
        stride=patch_size,
    ).squeeze()
    
    # Crop to match patch_losses size
    smoke_pooled = smoke_pooled[:P_H, :P_W]
    
    # Don't densify in high-smoke regions
    low_smoke = smoke_pooled < smoke_threshold
    
    # Combined mask
    densify_mask = high_error & low_smoke
    
    # Get center UV coordinates of selected patches
    selected = densify_mask.nonzero()  # (K, 2) [patch_row, patch_col]
    if len(selected) > 0:
        # Convert patch indices to pixel centers
        center_u = (selected[:, 1].float() + 0.5) * patch_size
        center_v = (selected[:, 0].float() + 0.5) * patch_size
        surface_uvs = torch.stack([center_u, center_v], dim=-1)
    else:
        surface_uvs = torch.zeros(0, 2, device=patch_losses.device)
    
    return densify_mask, surface_uvs


def proxy_depth_consistency_loss(
    expected_depth: torch.Tensor,
    proxy_depth: torch.Tensor,
    smoke_density: torch.Tensor | None = None,
    smoke_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Depth consistency loss between 3DGS rendered depth and proxy depth.
    
    Following Proxy-GS insight: encourage 3DGS geometry to align with proxy surface.
    For smoke scenes: only enforce in low-smoke regions where geometry is reliable.
    
    Args:
        expected_depth: (H, W) from 3DGS rendering
        proxy_depth: (H, W) from proxy mesh
        smoke_density: (H, W) optional smoke density for masking
        smoke_threshold: ignore regions with smoke density above this
        
    Returns:
        loss: scalar depth consistency loss
    """
    # Valid mask: both depths must be positive
    valid = (expected_depth > 0) & (proxy_depth > 0)
    
    if smoke_density is not None:
        valid = valid & (smoke_density < smoke_threshold)
    
    if valid.sum() == 0:
        return torch.tensor(0.0, device=expected_depth.device)
    
    # Normalize depths to reduce scale sensitivity
    d1 = expected_depth[valid]
    d2 = proxy_depth[valid]
    
    # Log-space L1 for scale-invariant depth matching
    loss = F.l1_loss(torch.log(d1 + 1e-6), torch.log(d2 + 1e-6))
    
    return loss
