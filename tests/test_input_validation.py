import pytest
from backend.perception import PerceptionLayer

def test_single_image_validation():
    perception = PerceptionLayer()
    meta = [{"format": ".tif", "sensor_type": "Optical", "modality": "Optical", "height": 512, "width": 512}]
    res = perception.inspect_and_validate(meta)
    assert res["passed"] is True
    assert res["image_count"] == 1

def test_missing_second_image_change_query_rejection():
    perception = PerceptionLayer()
    meta = [{"format": ".tif", "sensor_type": "Optical", "modality": "Optical", "height": 512, "width": 512}]
    res = perception.inspect_and_validate(meta, requested_task="change_vqa")
    assert res["passed"] is False
    assert any("Two spatially corresponding images are required" in err for err in res["errors"])

def test_missing_sar_optical_sar_query_rejection():
    perception = PerceptionLayer()
    meta = [
        {"format": ".tif", "sensor_type": "Optical", "modality": "Optical", "height": 512, "width": 512},
        {"format": ".tif", "sensor_type": "Optical", "modality": "Optical", "height": 512, "width": 512}
    ]
    res = perception.inspect_and_validate(meta, requested_task="optical_sar")
    assert res["passed"] is False
    assert any("requires one optical/multispectral image and one SAR image" in err for err in res["errors"])

def test_valid_optical_sar_pair():
    perception = PerceptionLayer()
    meta = [
        {"format": ".tif", "sensor_type": "Optical", "modality": "Optical", "height": 512, "width": 512},
        {"format": ".tif", "sensor_type": "SAR", "modality": "SAR", "height": 512, "width": 512}
    ]
    res = perception.inspect_and_validate(meta, requested_task="optical_sar")
    assert res["passed"] is True
