import numpy as np
from backend.models.optical_sar_tool import OpticalSARTool

def test_optical_sar_execution():
    tool = OpticalSARTool()
    opt = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    sar = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    res = tool.execute(opt, sar, "Use optical and SAR together", {}, {})
    assert "cloud_percentage" in res
    assert "builtup_percentage" in res
    assert res["is_baseline"] is False
