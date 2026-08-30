import numpy as np
import pytest
from backend.models.change_vqa_tool import ChangeVQATool

def test_change_execution():
    tool = ChangeVQATool()
    img1 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    res = tool.execute(img1, img2, "What changed?", {}, {})
    assert "change_percentage" in res
    assert "changed_pixels" in res
    assert res["is_baseline"] is False
