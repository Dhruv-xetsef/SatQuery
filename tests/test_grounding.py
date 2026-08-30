import numpy as np
from backend.models.grounding_tool import GroundingTool

def test_grounding_execution():
    tool = GroundingTool()
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    res = tool.execute(dummy_img, "Highlight the water body", {})
    assert "bbox_normalized" in res
    assert len(res["bbox_normalized"]) == 4
    assert res["is_baseline"] is False
