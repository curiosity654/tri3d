"""SimBEV synthetic dataset implementation for Tri3D.

SimBEV is a synthetic multi-task multi-sensor driving data generation tool.
This dataset class provides an interface to load SimBEV data in Tri3D format.
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

from .dataset import Dataset
from ..geometry import CameraProjection, RigidTransform, Rotation


class SimBEV(Dataset):
    """SimBEV synthetic driving dataset.
    
    Args:
        root: Root directory of the dataset
        split: Data split ('train', 'val', or 'test')
        **kwargs: Additional arguments passed to parent class
    """
    
    # Camera sensors following SimBEV naming convention
    cam_sensors = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
    ]

    # Image plane sensors (for calibration/alignment)
    img_sensors = [
        'IMG_FRONT_LEFT', 'IMG_FRONT', 'IMG_FRONT_RIGHT',
        'IMG_BACK_LEFT', 'IMG_BACK', 'IMG_BACK_RIGHT'
    ]
    
    # Point cloud sensors
    pcl_sensors = ['LIDAR']
    
    # Detection labels (SimBEV -> NuScenes mapping)
    LABEL_MAP = {
        'car': 'car',
        'truck': 'truck',
        'bus': 'bus',
        'motorcycle': 'motorcycle',
        'bicycle': 'bicycle',
        'pedestrian': 'pedestrian',
        'van': 'car',  # Map van to car
        'trailer': 'trailer',
    }

    # SimBEV semantic tag to object class mapping
    OBJECT_CLASSES = {
        7:  'traffic_light',
        8:  'traffic_sign',
        12: 'pedestrian',
        13: 'rider',
        14: 'car',
        15: 'truck',
        16: 'bus',
        18: 'motorcycle',
        19: 'bicycle',
        30: 'traffic_cone',
        31: 'barrier'
    }

    det_labels = list(OBJECT_CLASSES.values())
    sem_labels: List[str] = []
    
    def __init__(self, root: str, split: str = 'train', **kwargs):
        self.root = Path(root)
        self.split = split
        self._default_cam_sensor = self.cam_sensors[0]
        self._default_pcl_sensor = self.pcl_sensors[0]
        self._default_box_coords = self.pcl_sensors[0]
        
        # Load SimBEV info file from infos directory
        info_path = self.root / 'infos' / f'simbev_infos_{split}.json'
        if not info_path.exists():
            raise FileNotFoundError(f"SimBEV info file not found: {info_path}")
        
        with open(info_path) as f:
            self.infos = json.load(f)
        
        # Parse metadata
        self.metadata = self.infos.get('metadata', {})
        self._parse_calibration()
        
        # Build scene index
        self._scenes = list(self.infos.get('data', {}).keys())
        if not self._scenes:
            raise ValueError("No scenes found in SimBEV dataset")
        
        # Build frame index for each scene
        self._frames = {}
        for scene_id in self._scenes:
            scene_data = self.infos['data'][scene_id].get('scene_data', [])
            self._frames[scene_id] = list(range(len(scene_data)))

    def _calibration(self, seq_idx: int, src_sensor: str, dst_sensor: str):
        if src_sensor == dst_sensor:
            return RigidTransform(Rotation.from_euler("Z", 0.0), [0.0, 0.0, 0.0])

        if src_sensor in self.cam_sensors and dst_sensor in self.img_sensors:
            fx = float(self.camera_intrinsics[0, 0])
            fy = float(self.camera_intrinsics[1, 1])
            cx = float(self.camera_intrinsics[0, 2])
            cy = float(self.camera_intrinsics[1, 2])
            return CameraProjection("pinhole", [fx, fy, cx, cy])

        if src_sensor in self.sensor_calib and dst_sensor in self.sensor_calib:
            src2ego = self.sensor_calib[src_sensor]['transform']
            dst2ego = self.sensor_calib[dst_sensor]['transform']
            return dst2ego.inv() @ src2ego

        raise ValueError(f"Unsupported calibration: {src_sensor} -> {dst_sensor}")

    def _poses(self, seq_idx: int, sensor: str) -> RigidTransform:
        return self.poses(seq_idx, sensor)

    def _points(self, seq_idx: int, frame_idx: int, sensor: str) -> np.ndarray:
        return self.points(seq_idx, frame_idx, sensor=sensor, coords=sensor)

    def _boxes(self, seq_idx: int) -> list:
        raise NotImplementedError("SimBEV uses per-frame GT_DET; call boxes() instead.")
    
    def _parse_calibration(self):
        """Parse calibration parameters from metadata."""
        self.camera_intrinsics = np.array(
            self.metadata.get('camera_intrinsics', np.eye(3))
        )
        self.sensor_calib = {}
        
        for sensor in ['LIDAR'] + self.cam_sensors:
            if sensor in self.metadata:
                calib = self.metadata[sensor]
                trans = np.array(calib.get('sensor2ego_translation', calib.get('sensor2lidar_translation', [0, 0, 0])))
                rot = np.array(calib.get('sensor2ego_rotation', calib.get('sensor2lidar_rotation', [1, 0, 0, 0])))
                self.sensor_calib[sensor] = {
                    'translation': trans,
                    'rotation': rot,
                    'transform': self._build_transform(trans, rot)
                }

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve dataset paths across different SimBEV export layouts."""
        p = Path(path)
        if p.is_absolute() and p.exists():
            return p
        if not p.is_absolute():
            for base in (self.root, self.root.parent):
                candidate = base / p
                if candidate.exists():
                    return candidate

        parts = list(p.parts)
        if 'simbev' in parts:
            parts = parts[parts.index('simbev') + 1:]

        parts = ['samples' if x == 'sweeps' else x for x in parts]
        parts = ['ground_truth' if x == 'ground-truth' else x for x in parts]

        # Try resolving under current dataset root first.
        if len(parts) > 0:
            candidate = self.root / Path(*parts)
            if candidate.exists():
                return candidate

        # Fallback to root parent for layouts where info files and data root differ.
        if len(parts) > 0:
            candidate = self.root.parent / Path(*parts)
            if candidate.exists():
                return candidate

        return p
    
    def _build_transform(self, translation: np.ndarray, rotation: np.ndarray) -> RigidTransform:
        """Build RigidTransform from translation and quaternion rotation.
        
        Args:
            translation: 3D translation vector [x, y, z]
            rotation: Quaternion [w, x, y, z]
            
        Returns:
            RigidTransform object
        """
        rot = Rotation(rotation)
        return RigidTransform(rot, translation)
    
    # ==================== Dataset Interface Implementation ====================
    
    def sequences(self) -> List[int]:
        """Return list of sequence indices."""
        return list(range(len(self._scenes)))
    
    def sequence_name(self, seq_idx: int) -> str:
        """Return sequence name for given index."""
        return self._scenes[seq_idx]
    
    def frames(self, seq_idx: int, sensor: str) -> List[int]:
        """Return list of frame indices for given sequence and sensor."""
        scene_id = self._scenes[seq_idx]
        return self._frames[scene_id]
    
    def num_frames(self, seq_idx: int) -> int:
        """Return number of frames in sequence."""
        scene_id = self._scenes[seq_idx]
        return len(self._frames[scene_id])
    
    def timestamps(self, seq_idx: int, sensor: str) -> np.ndarray:
        """Return timestamps array in nanoseconds."""
        scene_id = self._scenes[seq_idx]
        scene_data = self.infos['data'][scene_id]['scene_data']
        return np.array([frame['timestamp'] for frame in scene_data])
    
    def keyframes(self, seq_idx: int, sensor: str) -> List[int]:
        """Return keyframe indices (all frames are keyframes in SimBEV)."""
        return self.frames(seq_idx, sensor)
    
    # ==================== Data Loading ====================
    
    def poses(self, seq_idx: int, sensor: str, timeline=None) -> RigidTransform:
        """Return sensor to world transforms.
        
        Args:
            seq_idx: Sequence index
            sensor: Sensor name
            timeline: Optional timeline for interpolation
            
        Returns:
            RigidTransform with sensor to world transforms
        """
        import torch
        scene_id = self._scenes[seq_idx]
        scene_data = self.infos['data'][scene_id]['scene_data']
        
        poses_rot = []
        poses_trans = []
        for frame in scene_data:
            # Build ego to global transform
            trans = np.array(frame['ego2global_translation'])
            rot = np.array(frame['ego2global_rotation'])
            ego2global = self._build_transform(trans, rot)
            
            # If sensor is not ego, apply sensor to ego transform
            if sensor in self.sensor_calib:
                sensor2ego = self.sensor_calib[sensor]['transform']
                sensor2global = ego2global @ sensor2ego
                pose = sensor2global
            else:
                pose = ego2global

            poses_rot.append(pose.rotation.quat)
            poses_trans.append(pose.translation.vec)

        return RigidTransform(np.array(poses_rot), np.array(poses_trans))
    
    def image(self, seq_idx: int, frame_idx: int, sensor: str):
        """Load camera image.
        
        Args:
            seq_idx: Sequence index
            frame_idx: Frame index
            sensor: Camera sensor name
            
        Returns:
            numpy array of image (H, W, 3) in BGR format
        """
        from PIL import Image
        scene_id = self._scenes[seq_idx]
        frame = self.infos['data'][scene_id]['scene_data'][frame_idx]
        
        img_key = f'RGB-{sensor}'
        if img_key in frame:
            img_path = self._resolve_path(frame[img_key])
            if img_path.exists():
                return Image.open(img_path)
        return None
    
    def points(self, seq_idx: int, frame_idx: int, sensor: str = 'LIDAR', 
               coords: str = 'world') -> np.ndarray:
        """Load point cloud data.
        
        Args:
            seq_idx: Sequence index
            frame_idx: Frame index
            sensor: Point cloud sensor name
            coords: Coordinate frame ('world', 'sensor', 'vehicle')
            
        Returns:
            numpy array of points (N, 5+) with [x, y, z, intensity, timestamp/ring]
        """
        scene_id = self._scenes[seq_idx]
        frame = self.infos['data'][scene_id]['scene_data'][frame_idx]
        
        if sensor not in frame:
            return np.empty((0, 5))
        
        pts_path = self._resolve_path(frame[sensor])
        if not pts_path.exists():
            return np.empty((0, 5))
        
        # Load SimBEV point cloud (.npz format)
        try:
            pts = np.load(pts_path)
            if isinstance(pts, np.lib.npyio.NpzFile):
                # .npz file - find the point data
                if 'points' in pts.files:
                    points = pts['points']
                elif 'data' in pts.files:
                    points = pts['data']
                elif 'arr_0' in pts.files:
                    points = pts['arr_0']
                else:
                    points = pts[pts.files[0]]
            else:
                points = pts
        except Exception as e:
            print(f"Error loading point cloud {pts_path}: {e}")
            return np.empty((0, 5))
        
        # Ensure points have at least 4 dimensions [x, y, z, intensity]
        if points.ndim == 1:
            points = points.reshape(-1, 4)
        elif points.shape[1] < 4:
            # Pad with zeros if needed
            padding = np.zeros((points.shape[0], 4 - points.shape[1]))
            points = np.hstack([points, padding])
        
        # Add timestamp/ring column if needed (mmdet3d expects 5 dims)
        if points.shape[1] < 5:
            time_col = np.zeros((points.shape[0], 1))
            points = np.hstack([points, time_col])
        
        return points[:, :5]  # Return [x, y, z, intensity, timestamp/ring]
    
    def boxes(self, seq_idx: int, frame_idx: int, coords: str = 'world') -> list:
        """Load 3D bounding box annotations.
        
        Args:
            seq_idx: Sequence index
            frame_idx: Frame index
            coords: Coordinate frame ('world', 'sensor', 'vehicle')
            
        Returns:
            List of Box objects with 3D bounding box annotations
        """
        from .dataset import Box
        
        scene_id = self._scenes[seq_idx]
        frame = self.infos['data'][scene_id]['scene_data'][frame_idx]
        
        gt_det_path = frame.get('GT_DET')
        if not gt_det_path:
            return []
        
        det_path = self._resolve_path(gt_det_path)
        if not det_path.exists():
            return []
        
        boxes = []
        try:
            det_data = np.load(det_path, allow_pickle=True)
            for det in det_data:
                if det.get('valid_flag') is False:
                    continue

                label = det.get('class')
                if label is None:
                    for tag in det.get('semantic_tags', []):
                        if tag in self.OBJECT_CLASSES:
                            label = self.OBJECT_CLASSES[tag]
                            break

                if label is None:
                    continue

                corners = np.asarray(det.get('bounding_box'))
                if corners.shape != (8, 3):
                    continue

                center, size, heading = self._corners_to_box(corners)
                if coords is not None and coords != 'world':
                    sensor2world = self.poses(seq_idx, coords)[frame_idx]
                    world2sensor = sensor2world.inv()
                    center = world2sensor.apply(center)
                    forward = np.array([np.cos(heading), np.sin(heading), 0.0])
                    forward = world2sensor.apply(forward) - world2sensor.apply(np.zeros(3))
                    heading = float(np.arctan2(forward[1], forward[0]))
                transform = RigidTransform(Rotation.from_euler("Z", heading), center)
                uid = int(det.get('id', len(boxes)))

                boxes.append(
                    Box(
                        frame=frame_idx,
                        uid=uid,
                        center=center,
                        size=size,
                        heading=heading,
                        transform=transform,
                        label=label,
                    )
                )
        except Exception as e:
            print(f"Warning: Could not load detection annotations from {det_path}: {e}")
            print("Returning empty box list. Detection evaluation may not work.")
        
        return boxes

    @staticmethod
    def _corners_to_box(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Estimate center/size/heading from 8 box corners (upright boxes)."""
        corners = np.asarray(corners, dtype=float)
        center_xy = corners[:, :2].mean(axis=0)
        z_min = float(corners[:, 2].min())
        z_max = float(corners[:, 2].max())
        center = np.array([center_xy[0], center_xy[1], (z_min + z_max) / 2.0], dtype=float)
        height = z_max - z_min

        pts = corners[:, :2] - center_xy
        if np.allclose(pts, 0):
            heading = 0.0
            length = 0.0
            width = 0.0
        else:
            cov = np.cov(pts.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            main = eigvecs[:, int(np.argmax(eigvals))]
            heading = float(np.arctan2(main[1], main[0]))

            proj_main = pts @ main
            perp = np.array([-main[1], main[0]])
            proj_perp = pts @ perp
            length = float(proj_main.max() - proj_main.min())
            width = float(proj_perp.max() - proj_perp.min())

            if width > length:
                length, width = width, length
                heading = float(heading + np.pi / 2)

        size = np.array([length, width, height], dtype=float)
        return center, size, heading
    
    def calibration(self, seq_idx: int, sensor: str) -> dict:
        """Get calibration parameters for sensor.
        
        Args:
            seq_idx: Sequence index
            sensor: Sensor name
            
        Returns:
            Dictionary with calibration parameters
        """
        if sensor in self.sensor_calib:
            return self.sensor_calib[sensor]
        return {}
    
    def image_size(self, seq_idx: int, sensor: str) -> Tuple[int, int]:
        """Get image size for camera sensor.
        
        Args:
            seq_idx: Sequence index
            sensor: Camera sensor name
            
        Returns:
            Tuple of (width, height)
        """
        # SimBEV uses standard camera intrinsics
        # Image size can be inferred from camera matrix
        # cx = width / 2, cy = height / 2
        cx = self.camera_intrinsics[0, 2]
        cy = self.camera_intrinsics[1, 2]
        return (int(cx * 2), int(cy * 2))
    
    @property
    def num_sequences(self) -> int:
        """Return total number of sequences."""
        return len(self._scenes)
    
    def sequence_info(self, seq_idx: int) -> dict:
        """Get information about a sequence.
        
        Args:
            seq_idx: Sequence index
            
        Returns:
            Dictionary with sequence information
        """
        scene_id = self._scenes[seq_idx]
        return self.infos['data'][scene_id].get('scene_info', {})
