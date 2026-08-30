from backend.query_engine import QueryUnderstandingEngine

def test_vqa_query_parsing():
    engine = QueryUnderstandingEngine()
    plan = engine.parse_query("Describe the land-cover", num_images=1)
    assert plan["task_type"] == "vqa_caption"
    assert plan["structured_query"]["requires_temporal_pair"] is False

def test_change_query_parsing():
    engine = QueryUnderstandingEngine()
    plan = engine.parse_query("What changed between these two dates?", num_images=2)
    assert plan["task_type"] == "change_vqa"
    assert plan["structured_query"]["requires_temporal_pair"] is True

def test_optical_sar_query_parsing():
    engine = QueryUnderstandingEngine()
    plan = engine.parse_query("Use optical and SAR together", num_images=2)
    assert plan["task_type"] == "optical_sar"
    assert plan["structured_query"]["requires_cross_modal_pair"] is True
