import os
import pytest
from backend.utils_geotiff import load_and_inspect_image

def test_load_optical_geotiff():
    path = "dataset/sample_data/single_optical.tif"
    if os.path.exists(path):
        rgb, meta = load_and_inspect_image(path)
        assert rgb.shape[2] == 3
        assert meta["width"] > 0
        assert meta["height"] > 0
        assert meta["sensor_type"] in ["Optical", "SAR"]

def test_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_and_inspect_image("non_existent_file.tif")
