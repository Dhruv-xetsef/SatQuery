import numpy as np
from backend.models.vqa_caption_tool import VQACaptionTool

def test_vqa_execution():
    tool = VQACaptionTool()
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    res = tool.execute(dummy_img, "What features are visible?", {})
    assert "text_response" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert res["is_baseline"] is False
