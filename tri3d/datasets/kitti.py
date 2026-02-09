import dataclasses
import pathlib
from typing import Sequence

import numpy as np
from PIL import Image

from .. import geometry
from .dataset import Box, Dataset


@dataclasses.dataclass(frozen=True)
class KittiBox(Box):
    truncated: float
    occluded: int
    alpha: float
    difficulty: int


class KITTI(Dataset):
    """KITTI 3D Object Detection dataset (data_object_* format)."""

    _default_cam_sensor = "CAM2"
    _default_pcl_sensor = "LIDAR"
    _default_box_coords = "LIDAR"

    cam_sensors = [
        "CAM2",
        "CAM3",
    ]

    img_sensors = [
        "IMG2",
        "IMG3",
    ]

    pcl_sensors = [
        "LIDAR",
    ]

    sensors = cam_sensors + pcl_sensors
    det_labels = [
        "Car",
        "Van",
        "Truck",
        "Pedestrian",
        "Person_sitting",
        "Cyclist",
        "Tram",
        "Misc",
    ]
    sem_labels = []
    sem2d_labels = []

    def __init__(self, root, split="training") -> None:
        self.root = pathlib.Path(root)
        self.split = split
        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"KITTI split not found: {split_dir}")

        lidar_dir = split_dir / "velodyne"
        image_dir = split_dir / "image_2"
        if not lidar_dir.exists():
            raise FileNotFoundError(f"KITTI velodyne dir not found: {lidar_dir}")
        if not image_dir.exists():
            raise FileNotFoundError(f"KITTI image_2 dir not found: {image_dir}")

        self.frame_ids = sorted(p.stem for p in lidar_dir.glob("*.bin"))
        if not self.frame_ids:
            raise FileNotFoundError(f"No KITTI frames found in {lidar_dir}")

        self._calib_cache = {}
        self._image_size_cache = {}

    def sequences(self):
        return [0]

    def timestamps(self, seq, sensor):
        return np.arange(len(self.frame_ids), dtype=np.float64)

    def _frame_id(self, frame: int) -> str:
        return self.frame_ids[frame]

    def _load_calib(self, frame: int):
        if frame in self._calib_cache:
            return self._calib_cache[frame]

        frame_id = self._frame_id(frame)
        calib_path = self.root / self.split / "calib" / f"{frame_id}.txt"
        if not calib_path.exists():
            raise FileNotFoundError(f"KITTI calib file not found: {calib_path}")

        calib = {}
        with open(calib_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                data = np.array([float(x) for x in value.strip().split()], dtype=np.float32)
                calib[key] = data

        def reshape(mat, shape):
            return calib[mat].reshape(shape)

        p2 = reshape("P2", (3, 4))
        p3 = reshape("P3", (3, 4))
        r0 = reshape("R0_rect", (3, 3))
        tr_velo = reshape("Tr_velo_to_cam", (3, 4))

        r0_4 = np.eye(4, dtype=np.float32)
        r0_4[:3, :3] = r0
        tr_velo_4 = np.eye(4, dtype=np.float32)
        tr_velo_4[:3, :] = tr_velo

        self._calib_cache[frame] = {
            "P2": p2,
            "P3": p3,
            "R0_rect": r0_4,
            "Tr_velo_to_cam": tr_velo_4,
        }
        return self._calib_cache[frame]

    def _get_image_size(self, frame: int, sensor: str):
        key = (frame, sensor)
        if key in self._image_size_cache:
            return self._image_size_cache[key]
        frame_id = self._frame_id(frame)
        if sensor == "IMG3":
            img_path = self.root / self.split / "image_3" / f"{frame_id}.png"
        else:
            img_path = self.root / self.split / "image_2" / f"{frame_id}.png"
        with Image.open(img_path) as img:
            w, h = img.size
        self._image_size_cache[key] = (w, h)
        return w, h

    def _calibration(self, seq, src_sensor, dst_sensor):
        if src_sensor in self.pcl_sensors or src_sensor == "boxes":
            src_sensor = "LIDAR"
        if dst_sensor in self.pcl_sensors or dst_sensor == "boxes":
            dst_sensor = "LIDAR"

        if src_sensor == dst_sensor:
            return geometry.Translation([0.0, 0.0, 0.0])

        frame = 0
        calib = self._load_calib(frame)
        r0 = calib["R0_rect"]
        tr_velo = calib["Tr_velo_to_cam"]

        lidar_to_cam = geometry.RigidTransform.from_matrix(r0 @ tr_velo)

        if dst_sensor in self.img_sensors:
            p_key = "P2" if dst_sensor == "IMG2" else "P3"
            proj = calib[p_key]
            fx, fy = proj[0, 0], proj[1, 1]
            cx, cy = proj[0, 2], proj[1, 2]
            tx = proj[0, 3] / fx
            ty = proj[1, 3] / fy
            tz = proj[2, 3]
            w, h = self._get_image_size(frame, dst_sensor)
            cam2img = geometry.CameraProjection(
                "pinhole", (fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0), w, h
            )
            cam2img = cam2img @ geometry.Translation([tx, ty, tz])
            cam_sensor = self.cam_sensors[self.img_sensors.index(dst_sensor)]
            src2cam = self._calibration(seq, src_sensor, cam_sensor)
            return cam2img @ src2cam

        if src_sensor in self.cam_sensors and dst_sensor == "LIDAR":
            return lidar_to_cam.inv()

        if src_sensor == "LIDAR" and dst_sensor in self.cam_sensors:
            return lidar_to_cam

        raise ValueError("invalid or unsupported coords combination")

    def _poses(self, seq, sensor) -> geometry.RigidTransform:
        if sensor == "boxes":
            sensor = "LIDAR"
        num = len(self.frame_ids)
        quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (num, 1))
        trans = np.zeros((num, 3), dtype=np.float64)
        return geometry.RigidTransform(quat, trans)

    def _points(self, seq, frame, sensor):
        frame_id = self._frame_id(frame)
        pts_path = self.root / self.split / "velodyne" / f"{frame_id}.bin"
        pts = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 4)
        return pts

    def _boxes(self, seq) -> Sequence[KittiBox]:
        if self.split != "training":
            return []
        boxes = []
        for frame, frame_id in enumerate(self.frame_ids):
            label_path = self.root / self.split / "label_2" / f"{frame_id}.txt"
            if not label_path.exists():
                continue

            calib = self._load_calib(frame)
            r0 = calib["R0_rect"]
            tr_velo = calib["Tr_velo_to_cam"]
            cam_rect_to_velo = np.linalg.inv(r0 @ tr_velo)

            with open(label_path, "r", encoding="utf-8") as f:
                for uid, line in enumerate(f):
                    fields = line.strip().split()
                    if len(fields) < 15:
                        continue
                    label = fields[0]
                    if label == "DontCare":
                        continue

                    truncated = float(fields[1])
                    occluded = int(fields[2])
                    alpha = float(fields[3])
                    h, w, l = (float(fields[8]), float(fields[9]), float(fields[10]))
                    x, y, z = (float(fields[11]), float(fields[12]), float(fields[13]))
                    rotation_y = float(fields[14])

                    center_cam = np.array([x, y, z, 1.0], dtype=np.float32)
                    center_velo = (cam_rect_to_velo @ center_cam)[:3]
                    center_velo[2] += h / 2.0

                    rot_cam = geometry.Rotation.from_euler("Y", rotation_y).as_matrix()[:3, :3]
                    rot_velo = cam_rect_to_velo[:3, :3] @ rot_cam
                    heading = float(np.arctan2(rot_velo[1, 0], rot_velo[0, 0]))

                    transform = geometry.RigidTransform(
                        geometry.Rotation.from_euler("Z", heading),
                        center_velo,
                    )

                    boxes.append(
                        KittiBox(
                            frame=frame,
                            uid=uid,
                            center=center_velo,
                            size=np.array([l, w, h], dtype=np.float32),
                            heading=heading,
                            transform=transform,
                            label=label,
                            truncated=truncated,
                            occluded=occluded,
                            alpha=alpha,
                            difficulty=0,
                        )
                    )
        return boxes

    def image(self, seq, frame, sensor):
        frame_id = self._frame_id(frame)
        if sensor == "CAM2":
            img_path = self.root / self.split / "image_2" / f"{frame_id}.png"
        elif sensor == "CAM3":
            img_path = self.root / self.split / "image_3" / f"{frame_id}.png"
        else:
            raise ValueError(f"Unknown camera sensor: {sensor}")
        return Image.open(img_path)
