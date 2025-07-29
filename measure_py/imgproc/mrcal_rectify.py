import mrcal
from pathlib import Path
import numpy as np
from logging import info, error
import math
import re
import cv2
from icecream import ic

from config import ConfigFile, ROOT_DIR

CONFIG = ConfigFile()
USE_LATLON = CONFIG.get_config("Rectify/latlon")

folder = Path()


def load_cameramodel(
        camera00: str = "",
        camera01: str = "",
        ):
    """
    Load the camera models from the specified files.
    if no camera models are provided, the default camera models are loaded.
    """
    if not camera00 and not camera01:
        camera00 = ROOT_DIR / "cameramodel" / "camera0-0.cameramodel"
        camera01 = ROOT_DIR / "cameramodel" / "camera0-1.cameramodel"
        
    global folder
    folder = Path(camera00).parent

    models = [mrcal.cameramodel(str(f)) for f in (camera00, camera01)]
    return models


def get_camera_fov(cm:mrcal.cameramodel):
    model, intrin = cm.intrinsics()
    w, h = cm.imagersize()

    info("Loaded model type: " + model)
    if 'LENSMODEL_OPENCV' in model:
        fx, fy = intrin[:2]

        # Calculate the arctangent in degrees
        result_rad = 2 * math.atan(w / 2 / fx)
        hfov = math.degrees(result_rad)

        # Calculate the arctangent in degrees
        result_rad = 2 * math.atan(h / 2 / fy)
        vfov = math.degrees(result_rad)
        print("\tH/VFOV:", hfov, vfov)
    elif 'LENSMODEL_SPLINED' in model:
        # get fov
        pattern = r"fov_x_deg=([\d.]+)"
        match = re.search(pattern, model)

        # Check if the pattern is found
        if match:
            hfov = float(match.group(1))
            vfov = hfov / w * h
            print("\tH/VFOV:", hfov, vfov)
        else:
            error("FOV not found!")

    return hfov, vfov


def rectify_image(
        image00: str,
        image01: str,
        models,
        az_fov_deg = -1,
        el_fov_deg = -1,
        ):
    """
    Rectify the images using the provided models.
    """
    images = [mrcal.load_image(f) for f in (image00, image01)]
    MODEL = "LENSMODEL_PINHOLE" if not USE_LATLON else "LENSMODEL_LATLON"

    if az_fov_deg == -1 or el_fov_deg == -1:
        hfov, vfov = get_camera_fov(models[0])
        az_fov_deg = int(hfov) + 2
        el_fov_deg = int(vfov) + 2
    models_rectified = mrcal.rectified_system(models,
                                            az_fov_deg=az_fov_deg,
                                            el_fov_deg=el_fov_deg,
                                            rectification_model=MODEL)
    
    global folder
    save_path = folder / f"rectification_{az_fov_deg}_{el_fov_deg}_{MODEL}.npy"
    # check if rectification maps are already saved
    if save_path.exists():
        info("Rectification maps found. Loading rectification maps.")
        rectification_maps = np.load(save_path, allow_pickle=True)
    else:
        info("Rectification maps not found. Creating new rectification maps.")
        rectification_maps = mrcal.rectification_maps(models, models_rectified)
        np.save(save_path, np.array(rectification_maps))

    images_rectified = [mrcal.transform_image(images[i], rectification_maps[i]) for i in range(2)]
    cv2.imwrite("left.jpg", images_rectified[0])
    cv2.imwrite("right.jpg", images_rectified[1])
    if len(images_rectified[0].shape) == 2:
        images_rectified = [np.dstack([img]*3) for img in images_rectified]

    # cv2.imshow("test", images_rectified[0])
    return images_rectified, models_rectified


def unproject_to_3dcoords(
        point,
        d,
        models,
        models_rectified,
        ):
    """
    Unproject the points to 3D coordinates using the provided models.
    """
    selected_points = np.array(point).reshape(-1, 2)
    p_rect0 = mrcal.stereo_unproject( d,
                            models_rectified,
                            disparity_scale = 1,
                            qrect0 = selected_points)
    Rt_cam0_rect0 = mrcal.compose_Rt( models[0].extrinsics_Rt_fromref(),
                                    models_rectified[0].extrinsics_Rt_toref() )
    p_cam0 = mrcal.transform_point_Rt(Rt_cam0_rect0, p_rect0)
    # ic(p_rect0, p_cam0)
    return p_cam0[0]


if __name__ == "__main__":
    import sys
    import logging

    logFormatter = logging.Formatter("[%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    cam_path = (str(ROOT_DIR / "cameramodel" / "camera0-0.cameramodel"),
                str(ROOT_DIR / "cameramodel" / "camera0-1.cameramodel"))
    img_paths = (
        str(ROOT_DIR / "A_17109260688908724.jpg"),
        str(ROOT_DIR / "D_17109260688908724.jpg")
    )
    models = load_cameramodel(*cam_path)
    images_rectified, models_rectified = rectify_image(*img_paths, 
                                        models)
