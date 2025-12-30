import dataclasses
import json
import os
import pathlib
import uuid

import numpy as np
import PIL.Image

from ..geometry import (
    CameraProjection,
    RigidTransform,
    Rotation,
    Transformation,
    Translation,
)
from ..misc import memoize_method
from .dataset import Box, Dataset

train_detect = \
    ['scene-0001', 'scene-0002', 'scene-0041', 'scene-0042', 'scene-0043', 'scene-0044', 'scene-0045', 'scene-0046',
     'scene-0047', 'scene-0048', 'scene-0049', 'scene-0050', 'scene-0051', 'scene-0052', 'scene-0053', 'scene-0054',
     'scene-0055', 'scene-0056', 'scene-0057', 'scene-0058', 'scene-0059', 'scene-0060', 'scene-0061', 'scene-0062',
     'scene-0063', 'scene-0064', 'scene-0065', 'scene-0066', 'scene-0067', 'scene-0068', 'scene-0069', 'scene-0070',
     'scene-0071', 'scene-0072', 'scene-0073', 'scene-0074', 'scene-0075', 'scene-0076', 'scene-0161', 'scene-0162',
     'scene-0163', 'scene-0164', 'scene-0165', 'scene-0166', 'scene-0167', 'scene-0168', 'scene-0170', 'scene-0171',
     'scene-0172', 'scene-0173', 'scene-0174', 'scene-0175', 'scene-0176', 'scene-0190', 'scene-0191', 'scene-0192',
     'scene-0193', 'scene-0194', 'scene-0195', 'scene-0196', 'scene-0199', 'scene-0200', 'scene-0202', 'scene-0203',
     'scene-0204', 'scene-0206', 'scene-0207', 'scene-0208', 'scene-0209', 'scene-0210', 'scene-0211', 'scene-0212',
     'scene-0213', 'scene-0214', 'scene-0254', 'scene-0255', 'scene-0256', 'scene-0257', 'scene-0258', 'scene-0259',
     'scene-0260', 'scene-0261', 'scene-0262', 'scene-0263', 'scene-0264', 'scene-0283', 'scene-0284', 'scene-0285',
     'scene-0286', 'scene-0287', 'scene-0288', 'scene-0289', 'scene-0290', 'scene-0291', 'scene-0292', 'scene-0293',
     'scene-0294', 'scene-0295', 'scene-0296', 'scene-0297', 'scene-0298', 'scene-0299', 'scene-0300', 'scene-0301',
     'scene-0302', 'scene-0303', 'scene-0304', 'scene-0305', 'scene-0306', 'scene-0315', 'scene-0316', 'scene-0317',
     'scene-0318', 'scene-0321', 'scene-0323', 'scene-0324', 'scene-0347', 'scene-0348', 'scene-0349', 'scene-0350',
     'scene-0351', 'scene-0352', 'scene-0353', 'scene-0354', 'scene-0355', 'scene-0356', 'scene-0357', 'scene-0358',
     'scene-0359', 'scene-0360', 'scene-0361', 'scene-0362', 'scene-0363', 'scene-0364', 'scene-0365', 'scene-0366',
     'scene-0367', 'scene-0368', 'scene-0369', 'scene-0370', 'scene-0371', 'scene-0372', 'scene-0373', 'scene-0374',
     'scene-0375', 'scene-0382', 'scene-0420', 'scene-0421', 'scene-0422', 'scene-0423', 'scene-0424', 'scene-0425',
     'scene-0426', 'scene-0427', 'scene-0428', 'scene-0429', 'scene-0430', 'scene-0431', 'scene-0432', 'scene-0433',
     'scene-0434', 'scene-0435', 'scene-0436', 'scene-0437', 'scene-0438', 'scene-0439', 'scene-0457', 'scene-0458',
     'scene-0459', 'scene-0461', 'scene-0462', 'scene-0463', 'scene-0464', 'scene-0465', 'scene-0467', 'scene-0468',
     'scene-0469', 'scene-0471', 'scene-0472', 'scene-0474', 'scene-0475', 'scene-0476', 'scene-0477', 'scene-0478',
     'scene-0479', 'scene-0480', 'scene-0566', 'scene-0568', 'scene-0570', 'scene-0571', 'scene-0572', 'scene-0573',
     'scene-0574', 'scene-0575', 'scene-0576', 'scene-0577', 'scene-0578', 'scene-0580', 'scene-0582', 'scene-0583',
     'scene-0665', 'scene-0666', 'scene-0667', 'scene-0668', 'scene-0669', 'scene-0670', 'scene-0671', 'scene-0672',
     'scene-0673', 'scene-0674', 'scene-0675', 'scene-0676', 'scene-0677', 'scene-0678', 'scene-0679', 'scene-0681',
     'scene-0683', 'scene-0684', 'scene-0685', 'scene-0686', 'scene-0687', 'scene-0688', 'scene-0689', 'scene-0739',
     'scene-0740', 'scene-0741', 'scene-0744', 'scene-0746', 'scene-0747', 'scene-0749', 'scene-0750', 'scene-0751',
     'scene-0752', 'scene-0757', 'scene-0758', 'scene-0759', 'scene-0760', 'scene-0761', 'scene-0762', 'scene-0763',
     'scene-0764', 'scene-0765', 'scene-0767', 'scene-0768', 'scene-0769', 'scene-0868', 'scene-0869', 'scene-0870',
     'scene-0871', 'scene-0872', 'scene-0873', 'scene-0875', 'scene-0876', 'scene-0877', 'scene-0878', 'scene-0880',
     'scene-0882', 'scene-0883', 'scene-0884', 'scene-0885', 'scene-0886', 'scene-0887', 'scene-0888', 'scene-0889',
     'scene-0890', 'scene-0891', 'scene-0892', 'scene-0893', 'scene-0894', 'scene-0895', 'scene-0896', 'scene-0897',
     'scene-0898', 'scene-0899', 'scene-0900', 'scene-0901', 'scene-0902', 'scene-0903', 'scene-0945', 'scene-0947',
     'scene-0949', 'scene-0952', 'scene-0953', 'scene-0955', 'scene-0956', 'scene-0957', 'scene-0958', 'scene-0959',
     'scene-0960', 'scene-0961', 'scene-0975', 'scene-0976', 'scene-0977', 'scene-0978', 'scene-0979', 'scene-0980',
     'scene-0981', 'scene-0982', 'scene-0983', 'scene-0984', 'scene-0988', 'scene-0989', 'scene-0990', 'scene-0991',
     'scene-1011', 'scene-1012', 'scene-1013', 'scene-1014', 'scene-1015', 'scene-1016', 'scene-1017', 'scene-1018',
     'scene-1019', 'scene-1020', 'scene-1021', 'scene-1022', 'scene-1023', 'scene-1024', 'scene-1025', 'scene-1074',
     'scene-1075', 'scene-1076', 'scene-1077', 'scene-1078', 'scene-1079', 'scene-1080', 'scene-1081', 'scene-1082',
     'scene-1083', 'scene-1084', 'scene-1085', 'scene-1086', 'scene-1087', 'scene-1088', 'scene-1089', 'scene-1090',
     'scene-1091', 'scene-1092', 'scene-1093', 'scene-1094', 'scene-1095', 'scene-1096', 'scene-1097', 'scene-1098',
     'scene-1099', 'scene-1100', 'scene-1101', 'scene-1102', 'scene-1104', 'scene-1105']

train_track = \
    ['scene-0004', 'scene-0005', 'scene-0006', 'scene-0007', 'scene-0008', 'scene-0009', 'scene-0010', 'scene-0011',
     'scene-0019', 'scene-0020', 'scene-0021', 'scene-0022', 'scene-0023', 'scene-0024', 'scene-0025', 'scene-0026',
     'scene-0027', 'scene-0028', 'scene-0029', 'scene-0030', 'scene-0031', 'scene-0032', 'scene-0033', 'scene-0034',
     'scene-0120', 'scene-0121', 'scene-0122', 'scene-0123', 'scene-0124', 'scene-0125', 'scene-0126', 'scene-0127',
     'scene-0128', 'scene-0129', 'scene-0130', 'scene-0131', 'scene-0132', 'scene-0133', 'scene-0134', 'scene-0135',
     'scene-0138', 'scene-0139', 'scene-0149', 'scene-0150', 'scene-0151', 'scene-0152', 'scene-0154', 'scene-0155',
     'scene-0157', 'scene-0158', 'scene-0159', 'scene-0160', 'scene-0177', 'scene-0178', 'scene-0179', 'scene-0180',
     'scene-0181', 'scene-0182', 'scene-0183', 'scene-0184', 'scene-0185', 'scene-0187', 'scene-0188', 'scene-0218',
     'scene-0219', 'scene-0220', 'scene-0222', 'scene-0224', 'scene-0225', 'scene-0226', 'scene-0227', 'scene-0228',
     'scene-0229', 'scene-0230', 'scene-0231', 'scene-0232', 'scene-0233', 'scene-0234', 'scene-0235', 'scene-0236',
     'scene-0237', 'scene-0238', 'scene-0239', 'scene-0240', 'scene-0241', 'scene-0242', 'scene-0243', 'scene-0244',
     'scene-0245', 'scene-0246', 'scene-0247', 'scene-0248', 'scene-0249', 'scene-0250', 'scene-0251', 'scene-0252',
     'scene-0253', 'scene-0328', 'scene-0376', 'scene-0377', 'scene-0378', 'scene-0379', 'scene-0380', 'scene-0381',
     'scene-0383', 'scene-0384', 'scene-0385', 'scene-0386', 'scene-0388', 'scene-0389', 'scene-0390', 'scene-0391',
     'scene-0392', 'scene-0393', 'scene-0394', 'scene-0395', 'scene-0396', 'scene-0397', 'scene-0398', 'scene-0399',
     'scene-0400', 'scene-0401', 'scene-0402', 'scene-0403', 'scene-0405', 'scene-0406', 'scene-0407', 'scene-0408',
     'scene-0410', 'scene-0411', 'scene-0412', 'scene-0413', 'scene-0414', 'scene-0415', 'scene-0416', 'scene-0417',
     'scene-0418', 'scene-0419', 'scene-0440', 'scene-0441', 'scene-0442', 'scene-0443', 'scene-0444', 'scene-0445',
     'scene-0446', 'scene-0447', 'scene-0448', 'scene-0449', 'scene-0450', 'scene-0451', 'scene-0452', 'scene-0453',
     'scene-0454', 'scene-0455', 'scene-0456', 'scene-0499', 'scene-0500', 'scene-0501', 'scene-0502', 'scene-0504',
     'scene-0505', 'scene-0506', 'scene-0507', 'scene-0508', 'scene-0509', 'scene-0510', 'scene-0511', 'scene-0512',
     'scene-0513', 'scene-0514', 'scene-0515', 'scene-0517', 'scene-0518', 'scene-0525', 'scene-0526', 'scene-0527',
     'scene-0528', 'scene-0529', 'scene-0530', 'scene-0531', 'scene-0532', 'scene-0533', 'scene-0534', 'scene-0535',
     'scene-0536', 'scene-0537', 'scene-0538', 'scene-0539', 'scene-0541', 'scene-0542', 'scene-0543', 'scene-0544',
     'scene-0545', 'scene-0546', 'scene-0584', 'scene-0585', 'scene-0586', 'scene-0587', 'scene-0588', 'scene-0589',
     'scene-0590', 'scene-0591', 'scene-0592', 'scene-0593', 'scene-0594', 'scene-0595', 'scene-0596', 'scene-0597',
     'scene-0598', 'scene-0599', 'scene-0600', 'scene-0639', 'scene-0640', 'scene-0641', 'scene-0642', 'scene-0643',
     'scene-0644', 'scene-0645', 'scene-0646', 'scene-0647', 'scene-0648', 'scene-0649', 'scene-0650', 'scene-0651',
     'scene-0652', 'scene-0653', 'scene-0654', 'scene-0655', 'scene-0656', 'scene-0657', 'scene-0658', 'scene-0659',
     'scene-0660', 'scene-0661', 'scene-0662', 'scene-0663', 'scene-0664', 'scene-0695', 'scene-0696', 'scene-0697',
     'scene-0698', 'scene-0700', 'scene-0701', 'scene-0703', 'scene-0704', 'scene-0705', 'scene-0706', 'scene-0707',
     'scene-0708', 'scene-0709', 'scene-0710', 'scene-0711', 'scene-0712', 'scene-0713', 'scene-0714', 'scene-0715',
     'scene-0716', 'scene-0717', 'scene-0718', 'scene-0719', 'scene-0726', 'scene-0727', 'scene-0728', 'scene-0730',
     'scene-0731', 'scene-0733', 'scene-0734', 'scene-0735', 'scene-0736', 'scene-0737', 'scene-0738', 'scene-0786',
     'scene-0787', 'scene-0789', 'scene-0790', 'scene-0791', 'scene-0792', 'scene-0803', 'scene-0804', 'scene-0805',
     'scene-0806', 'scene-0808', 'scene-0809', 'scene-0810', 'scene-0811', 'scene-0812', 'scene-0813', 'scene-0815',
     'scene-0816', 'scene-0817', 'scene-0819', 'scene-0820', 'scene-0821', 'scene-0822', 'scene-0847', 'scene-0848',
     'scene-0849', 'scene-0850', 'scene-0851', 'scene-0852', 'scene-0853', 'scene-0854', 'scene-0855', 'scene-0856',
     'scene-0858', 'scene-0860', 'scene-0861', 'scene-0862', 'scene-0863', 'scene-0864', 'scene-0865', 'scene-0866',
     'scene-0992', 'scene-0994', 'scene-0995', 'scene-0996', 'scene-0997', 'scene-0998', 'scene-0999', 'scene-1000',
     'scene-1001', 'scene-1002', 'scene-1003', 'scene-1004', 'scene-1005', 'scene-1006', 'scene-1007', 'scene-1008',
     'scene-1009', 'scene-1010', 'scene-1044', 'scene-1045', 'scene-1046', 'scene-1047', 'scene-1048', 'scene-1049',
     'scene-1050', 'scene-1051', 'scene-1052', 'scene-1053', 'scene-1054', 'scene-1055', 'scene-1056', 'scene-1057',
     'scene-1058', 'scene-1106', 'scene-1107', 'scene-1108', 'scene-1109', 'scene-1110']

train = list(sorted(set(train_detect + train_track)))

val = \
    ['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
     'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095',
     'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099', 'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103',
     'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0221',
     'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
     'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
     'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524',
     'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559',
     'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565', 'scene-0625', 'scene-0626',
     'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633', 'scene-0634', 'scene-0635', 'scene-0636',
     'scene-0637', 'scene-0638', 'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780',
     'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784', 'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797',
     'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802', 'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907',
     'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
     'scene-0916', 'scene-0917', 'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924',
     'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962',
     'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972', 'scene-1059',
     'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1065', 'scene-1066', 'scene-1067',
     'scene-1068', 'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073']

val_mini = \
    ['scene-0003']

@dataclasses.dataclass(frozen=True)
class NuScenesBox(Box):
    attributes: list[str]
    visibility: str | None
    num_lidar_pts: int
    num_radar_pts: int
    velocity: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )


@dataclasses.dataclass()
class Scene:
    data: dict[str, list[str]]
    calibration: dict[str, np.ndarray]
    ego_poses: dict[str, RigidTransform]
    keyframes: dict[str, np.ndarray]
    sample_tokens: list[str]
    timestamps: dict[str, np.ndarray]
    boxes: list[NuScenesBox]


class NuScenes(Dataset):
    """`NuScenes <https://www.nuscenes.org>`_ dataset.

    .. note::

       Notable differences with original NuScenes data:

       * Size encoded as length, width, height instead of width, length, height.
       * Lidar pcl is rotated by 90° so x axis points forward.
       * Annotations are automatically interpolated between keyframes.

    The :meth:`keyframes` method returns the indices of the keyframes for each
    sensor. Keyframes aggregate a sample for each sensor around a timestamps
    at around 2Hz.

    .. run::

        import matplotlib.pyplot as plt
        import numpy as np
        from tri3d.datasets import NuScenes

        plt.switch_backend("Agg")

        dataset = NuScenes("datasets/nuscenes", "v1.0-mini")
        name = "tri3d.datasets.NuScenes"
        camera, imgcoords, lidar = "CAM_FRONT", "IMG_FRONT", "LIDAR_TOP"
        seq, frame, cam_frame = 5, 130, 77
    """

    scenes: list[Scene]
    _default_cam_sensor = "CAM_FRONT"
    _default_pcl_sensor = "LIDAR_TOP"
    _default_box_coords = "LIDAR_TOP"

    def __init__(
        self,
        root,
        subset="v1.0-mini",
        split=None,
        det_label_map=None,
        sem_label_map=None,
    ):
        root = pathlib.Path(root)
        self.root = root
        self.subset = subset
        self.split = split
        self.det_label_map = det_label_map
        self.sem_label_map = sem_label_map

        # load original data
        with open(root / subset / "attribute.json", "rb") as f:
            attribute = json.load(f)
        with open(root / subset / "calibrated_sensor.json", "rb") as f:
            calibrated_sensor = json.load(f)
        with open(root / subset / "category.json", "rb") as f:
            category = json.load(f)
        with open(root / subset / "ego_pose.json", "rb") as f:
            ego_pose = json.load(f)
        with open(root / subset / "instance.json", "rb") as f:
            instance = json.load(f)
        with open(root / subset / "sample.json", "rb") as f:
            sample = json.load(f)
        with open(root / subset / "sample_annotation.json", "rb") as f:
            sample_annotation = json.load(f)
        with open(root / subset / "sample_data.json", "rb") as f:
            sample_data = json.load(f)
        with open(root / subset / "scene.json", "rb") as f:
            scene = json.load(f)
        with open(root / subset / "sensor.json", "rb") as f:
            sensor = json.load(f)
        with open(root / subset / "visibility.json", "rb") as f:
            visibility = json.load(f)
        if os.path.exists(root / subset / "lidarseg.json"):
            with open(root / subset / "lidarseg.json", "rb") as f:
                lidarseg = json.load(f)
        else:
            lidarseg = []
        if os.path.exists(root / subset / "panoptic.json"):
            with open(root / subset / "panoptic.json", "rb") as f:
                panoptic = json.load(f)
        else:
            panoptic = []

        # convert into dictionaries
        attribute = {v["token"]: v for v in attribute}
        calibrated_sensor = {v["token"]: v for v in calibrated_sensor}
        category = {v["token"]: v for v in category}
        ego_pose = {v["token"]: v for v in ego_pose}
        instance = {v["token"]: v for v in instance}
        sample = {v["token"]: v for v in sample}
        sample_data = {v["token"]: v for v in sample_data}
        scene = {v["token"]: v for v in scene}
        sensor = {v["token"]: v for v in sensor}

        if self.split is not None:
            if self.split == "train":
                valid_scenes = set(train)
            elif self.split == "val":
                valid_scenes = set(val)
            elif self.split == "val_mini":
                valid_scenes = set(val_mini)
            elif isinstance(self.split, (list, tuple, set)):
                valid_scenes = set(self.split)
            else:
                valid_scenes = None

            if valid_scenes is not None:
                scene = {k: v for k, v in scene.items() if v["name"] in valid_scenes}

        # extract sensor names
        self.cam_sensors = []
        self.img_sensors = []
        self.pcl_sensors = []
        for s in sensor.values():
            if s["modality"] == "camera":
                self.cam_sensors.append(s["channel"])
                self.img_sensors.append(s["channel"].replace("CAM", "IMG"))
            elif s["modality"] == "lidar":
                self.pcl_sensors.append(s["channel"])

        # extract label names
        self.det_labels = [c["name"] for c in category.values()]
        self.sem_labels = self.det_labels

        # categories
        if "index" in next(iter(category.values())):
            self.categories = [None] * (max(c["index"] for c in category.values()) + 1)
            for c in category.values():
                self.categories[c["index"]] = c["name"]
        else:
            self.categories = [c["name"] for c in category.values()]

        # merge channel name into sample_data
        for sample_data_v in sample_data.values():
            calibrated_sensor_t = sample_data_v["calibrated_sensor_token"]
            calibrated_sensor_v = calibrated_sensor[calibrated_sensor_t]
            sensor_v = sensor[calibrated_sensor_v["sensor_token"]]
            sample_data_v["channel"] = sensor_v["channel"]

        # merge ego pose into sample_data
        for sample_data_v in sample_data.values():
            sample_data_v["ego_pose"] = ego_pose[sample_data_v["ego_pose_token"]]

        # merge lidarseg and panoptic into sample_data
        for lidarseg_v in lidarseg:
            sample_data_t = lidarseg_v["sample_data_token"]
            sample_data[sample_data_t]["lidarseg_filename"] = lidarseg_v["filename"]

        for panoptic_v in panoptic:
            sample_data_t = panoptic_v["sample_data_token"]
            sample_data[sample_data_t]["panoptic_filename"] = panoptic_v["filename"]

        # group sample tokens by scenes
        scene_samples = {scene_t: [] for scene_t in scene.keys()}
        sample_idx = {}
        for scene_t, scene_v in scene.items():
            sample_t = scene_v["first_sample_token"]
            i = 0
            while sample_t != "":
                scene_samples[scene_t].append(sample_t)
                sample_idx[sample_t] = i
                sample_t = sample[sample_t]["next"]
                i += 1

        # Group sample data by scene
        scene_data = {scene_t: {} for scene_t in scene.keys()}

        for sample_data_v in sample_data.values():
            sample_t = sample_data_v["sample_token"]
            sample_v = sample[sample_t]
            scene_t = sample_v["scene_token"]
            if scene_t not in scene_data:
                continue
            channel = sample_data_v["channel"]

            if channel not in scene_data[scene_t]:
                scene_data[scene_t][channel] = []

            scene_data[scene_t][channel].append(sample_data_v)

        # sort sample data by timestamps
        for scene_data_v in scene_data.values():
            for channel in list(scene_data_v.keys()):
                scene_data_v[channel] = sorted(
                    scene_data_v[channel], key=lambda d: d["timestamp"]
                )

        # Group box annotations by scene
        scene_annotations = {scene_t: [] for scene_t in scene.keys()}
        for sample_annotation_v in sample_annotation:
            sample_t = sample_annotation_v["sample_token"]
            if sample_t not in sample_idx:
                continue
            scene_t = sample[sample_t]["scene_token"]
            instance_t = sample_annotation_v["instance_token"]
            center = sample_annotation_v["translation"]
            rotation = sample_annotation_v["rotation"]
            transform = RigidTransform(rotation, center)
            label = category[instance[instance_t]["category_token"]]["name"]
            annotation_attributes = [
                attribute[t]["name"] for t in sample_annotation_v["attribute_tokens"]
            ]
            if "visibility_t" in sample_annotation_v is None:
                box_visibility = visibility[sample_annotation_v["visibility_t"]][
                    "level"
                ]
            else:
                box_visibility = None
            width, length, height = sample_annotation_v["size"]

            scene_annotations[scene_t].append(
                NuScenesBox(
                    frame=sample_idx[sample_t],
                    uid=instance_t,
                    center=center,
                    size=np.array([length, width, height]),
                    heading=transform.rotation.as_euler("ZYX")[0],
                    transform=transform,
                    label=label,
                    attributes=annotation_attributes,
                    visibility=box_visibility,
                    num_lidar_pts=sample_annotation_v["num_lidar_pts"],
                    num_radar_pts=sample_annotation_v["num_radar_pts"],
                )
            )

        scene_annotations = {  # sort by frame and uid
            scene_t: sorted(boxes, key=lambda b: (b.frame, b.uid))
            for scene_t, boxes in scene_annotations.items()
        }

        self.scenes = []
        for scene_t in sorted(scene.keys()):
            # calib
            scene_calibration = {}
            scene_ego_poses = {}
            scene_keyframes = {}
            scene_timestamps = {}
            scene_data_ = {}

            for channel, channel_data_v in scene_data[scene_t].items():
                if channel not in self.pcl_sensors and channel not in self.cam_sensors:
                    continue

                calib = calibrated_sensor[channel_data_v[0]["calibrated_sensor_token"]]

                if channel in self.pcl_sensors:
                    calib["rotation"] = (
                        Rotation(calib["rotation"])
                        @ Rotation.from_euler("Z", np.pi / 2)
                    ).as_quat()  # type: ignore

                scene_calibration[channel] = calib

                scene_ego_poses[channel] = RigidTransform(
                    [d["ego_pose"]["rotation"] for d in channel_data_v],
                    [d["ego_pose"]["translation"] for d in channel_data_v],
                )

                scene_keyframes[channel] = np.array(
                    [i for i, d in enumerate(channel_data_v) if d["is_key_frame"]]
                )

                scene_timestamps[channel] = np.array(
                    [d["timestamp"] for d in channel_data_v]
                )

                scene_data_[channel] = [d["filename"] for d in channel_data_v]

                if channel in self.pcl_sensors:
                    scene_data_["lidarseg"] = [
                        d.get("lidarseg_filename", None) for d in channel_data_v
                    ]
                    scene_data_["panoptic"] = [
                        d.get("panoptic_filename", None) for d in channel_data_v
                    ]

            scene_timestamps["sample"] = np.array(
                [sample[sample_t]["timestamp"] for sample_t in scene_samples[scene_t]]
            )

            self.scenes.append(
                Scene(
                    data=scene_data_,
                    calibration=scene_calibration,
                    ego_poses=scene_ego_poses,
                    keyframes=scene_keyframes,
                    sample_tokens=scene_samples[scene_t],
                    timestamps=scene_timestamps,
                    boxes=scene_annotations[scene_t],
                )
            )

        self.annotations_cache = {}
        self._add_velocity()

    def _add_velocity(self):
        """Compute velocities for all boxes in all scenes."""
        for scene in self.scenes:
            # Group boxes by instance UID
            instance_boxes = {}
            for box in scene.boxes:
                if box.uid not in instance_boxes:
                    instance_boxes[box.uid] = []
                instance_boxes[box.uid].append(box)

            # Map from (uid, frame) to velocity
            vel_map = {}
            for uid, boxes in instance_boxes.items():
                boxes.sort(key=lambda b: b.frame)
                for i in range(len(boxes)):
                    current_box = boxes[i]
                    if i > 0:
                        prev_box = boxes[i - 1]
                        t_curr = scene.timestamps["sample"][current_box.frame] / 1e6
                        t_prev = scene.timestamps["sample"][prev_box.frame] / 1e6
                        dt = t_curr - t_prev
                        if dt > 0:
                            vel = (
                                np.array(current_box.center) - np.array(prev_box.center)
                            ) / dt
                        else:
                            vel = np.array([0.0, 0.0, 0.0])
                    else:
                        vel = np.array([0.0, 0.0, 0.0])
                    vel_map[(uid, current_box.frame)] = vel

            # Update boxes in scene
            for i, box in enumerate(scene.boxes):
                vel = vel_map.get((box.uid, box.frame), np.array([0.0, 0.0, 0.0]))
                scene.boxes[i] = dataclasses.replace(box, velocity=vel)

    @memoize_method(maxsize=100)
    def _calibration(self, seq, src_sensor, dst_sensor) -> Transformation:
        if src_sensor == dst_sensor:
            return Translation([0.0, 0.0, 0.0])

        if dst_sensor in self.img_sensors:
            cam_sensor = self.cam_sensors[self.img_sensors.index(dst_sensor)]
            intrinsic = self.scenes[seq].calibration[cam_sensor]["camera_intrinsic"]
            intrinsic = (
                intrinsic[0][0],
                intrinsic[1][1],
                intrinsic[0][2],
                intrinsic[1][2],
            )
            cam2img = CameraProjection("pinhole", intrinsic)

            cam = self.cam_sensors[self.img_sensors.index(dst_sensor)]
            src2cam = self._calibration(seq, src_sensor, cam)

            return cam2img @ src2cam

        if src_sensor == "ego":
            return self._calibration(seq, dst_sensor, src_sensor).inv()

        if src_sensor not in self.cam_sensors and src_sensor not in self.pcl_sensors:
            raise ValueError()

        src_calib = self.scenes[seq].calibration[src_sensor]
        src_calib = RigidTransform(src_calib["rotation"], src_calib["translation"])

        if dst_sensor == "ego":
            dst_calib = Translation([0, 0, 0])
        else:
            dst_calib = self.scenes[seq].calibration[dst_sensor]
            dst_calib = RigidTransform(dst_calib["rotation"], dst_calib["translation"])

        return dst_calib.inv() @ src_calib

    @memoize_method(maxsize=10)
    def _poses(self, seq, sensor):
        if sensor == "boxes":
            N = len(self.scenes[seq].sample_tokens)
            return RigidTransform.from_matrix(np.tile(np.eye(4), (N, 1, 1)))

        if sensor in self.img_sensors:
            sensor = self.cam_sensors[self.img_sensors.index(sensor)]

        sensor2ego: RigidTransform = self._calibration(seq, sensor, "ego")  # type: ignore
        ego2world = self.scenes[seq].ego_poses[sensor]
        return ego2world @ sensor2ego

    def _points(self, seq, frame, sensor):
        filename = self.root / self.scenes[seq].data[sensor][frame]
        pcl = np.fromfile(filename, dtype=np.float32).reshape(-1, 5)

        # Rotate pcl to make x point forward
        pcl[:, 0], pcl[:, 1] = pcl[:, 1], -pcl[:, 0]
        return pcl

    def _boxes(self, seq):
        return self.scenes[seq].boxes

    def sequences(self):
        return list(range(len(self.scenes)))

    def timestamps(self, seq, sensor):
        if sensor == "boxes":
            sensor = "sample"
        elif sensor in self.img_sensors:
            sensor = self.cam_sensors[self.img_sensors.index(sensor)]

        return self.scenes[seq].timestamps[sensor]

    def image(self, seq, frame, sensor="CAM_FRONT"):
        if sensor in self.img_sensors:
            sensor = self.cam_sensors[self.img_sensors.index(sensor)]

        filename = self.root / self.scenes[seq].data[sensor][frame]
        return PIL.Image.open(filename)

    def rectangles(self, seq: int, frame: int):
        raise NotImplementedError

    def semantic(self, seq, frame, sensor="LIDAR_TOP"):
        filename = self.scenes[seq].data["lidarseg"][frame]
        if filename is None:
            raise ValueError(f"frame {frame} has no segmentation")

        semantic = np.fromfile(self.root / filename, dtype=np.uint8)

        if self.sem_label_map is not None:
            semantic = self.sem_label_map[semantic]

        return semantic

    def instances(self, seq, frame, sensor="LIDAR_TOP"):
        filename = self.scenes[seq].data["panoptic"][frame]
        if filename is None:
            raise ValueError(f"frame {frame} has no panoptic")

        panoptic = np.load(self.root / filename)["data"] % 1000

        return panoptic

    def keyframes(self, seq, sensor):
        """The indices of the keyframes within the frames of each sensor."""
        if sensor in self.img_sensors:
            sensor = self.cam_sensors[self.img_sensors.index(sensor)]

        return self.scenes[seq].keyframes[sensor]

    def sample_tokens(self, seq):
        """The sample token for each keyframe."""
        return self.scenes[seq].sample_tokens


def dump_nuscene_boxes(
    dataset: NuScenes,
    seq_indices: list[int],
    sensor: str,
    boxes: list[NuScenesBox],
    keyframes: bool = False,
):
    """Convert boxes to the NuScene format (`sample_annotation.json`).

    Args:
        dataset: the nuscene dataset
        seq_indices: *for each box* the index of its sequence
        sensor: the coordinate system in which box poses are expressed
        boxes: the boxes to export
        keyframes:
            Whether boxes[*].frame indexes the keyframes or the sensor timeline.

    Note: Boxes on frames other than keyframes are skipped.

    Warning:
        New annotation token are generated so the `{first,last}_annotation_token`
        foreign keys in `instance.json` won't work with the exported boxes.
        These fields are not used in tri3D.
    """
    out = []
    prev = None
    token = uuid.uuid4().hex
    next = uuid.uuid4().hex

    assert len(seq_indices) == len(boxes)

    for i, s, b in zip(range(len(boxes)), seq_indices, boxes):
        if keyframes:
            frame = dataset.keyframes(s, sensor)[b.frame]
            sample_token = dataset.sample_tokens(s)[b.frame]
        else:
            frame = b.frame
            keyframe_idx = np.searchsorted(dataset.keyframes(s, sensor), b.frame)
            if dataset.keyframes(s, sensor)[keyframe_idx] != b.frame:
                continue
            sample_token = dataset.sample_tokens(s)[keyframe_idx]

        ego_pose = dataset.poses(s, sensor)[frame]
        box_pose: RigidTransform = ego_pose @ b.transform  # type: ignore
        length, width, height = b.size

        out.append(
            {
                "token": token,
                "sample_token": sample_token,
                "instance_token": b.uid,
                "attribute_token": [],  # TODO
                "visibility_token": None,  # TODO
                "translation": box_pose.translation.vec,
                "size": [width, length, height],
                "rotation": box_pose.rotation.quat,
                "num_lidar_pts": b.num_lidar_pts,
                "num_radar_pts": b.num_radar_pts,
                "prev": prev,
                "next": next,
            }
        )

        prev = token
        token = next
        next = uuid.uuid4().hex

    out[-1]["next"] = None

    return out
