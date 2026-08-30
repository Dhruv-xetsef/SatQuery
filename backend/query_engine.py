import re
from typing import Dict, Any, List

class QueryUnderstandingEngine:
    """
    2. QUERY UNDERSTANDING ENGINE
    Parses natural language query intent into a structured representation:
    - task
    - requires_temporal_pair
    - requires_cross_modal_pair
    - requires_grounding
    - target_entities
    - spatial_relations
    - temporal_relations
    - requested_output
    """
    def __init__(self):
        self.change_keywords = [
            "change", "changed", "dates", "time", "temporal", "between", 
            "increase", "decrease", "deforestation", "expansion", "before and after",
            "bitemporal", "converted", "remained", "trend"
        ]
        self.grounding_keywords = [
            "highlight", "locate", "where is", "bounding box", "find the", 
            "grounding", "box", "show me where", "region", "outline", "delineate"
        ]
        self.crossmodal_keywords = [
            "sar", "optical", "together", "both", "cross-modal", "multisensor", 
            "radar", "fusion", "combine optical and sar", "penetrate", "cloud"
        ]
        self.entity_keywords = {
            "water": ["water", "river", "lake", "reservoir", "stream", "pond", "coastal"],
            "builtup": ["built-up", "urban", "building", "structure", "city", "settlement", "roof", "road", "impervious"],
            "vegetation": ["forest", "tree", "canopy", "vegetation", "grassland", "crop", "agriculture"],
            "wetland": ["marsh", "swamp", "wetland"]
        }

    def parse_query(self, query: str, num_images: int = 1, force_task: str = "auto") -> Dict[str, Any]:
        q_lower = query.lower()

        is_change = any(k in q_lower for k in self.change_keywords) or (num_images == 2 and ("change" in q_lower or "date" in q_lower or "vs" in q_lower))
        is_crossmodal = any(k in q_lower for k in self.crossmodal_keywords) or (num_images == 2 and ("optical" in q_lower or "sar" in q_lower))
        is_grounding = any(k in q_lower for k in self.grounding_keywords)

        if force_task and force_task != "auto":
            task_type = force_task
            intent = force_task.upper()
            reasoning = f"Task type explicitly forced to '{force_task}'."
        else:
            if is_change and num_images >= 2:
                task_type = "change_vqa"
                intent = "BI_TEMPORAL_CHANGE_ANALYSIS"
                reasoning = "Query requests bi-temporal change detection and change VQA."
            elif is_crossmodal and num_images >= 2:
                task_type = "optical_sar"
                intent = "CROSS_MODAL_FUSION"
                reasoning = "Query requests joint optical-SAR multisensor analysis."
            elif is_grounding:
                task_type = "grounding"
                intent = "REGION_GROUNDING"
                reasoning = "Query requests target region spatial grounding and bounding box."
            elif is_change and num_images < 2:
                task_type = "change_vqa"
                intent = "BI_TEMPORAL_CHANGE_ANALYSIS"
                reasoning = "Query requests change analysis (bi-temporal image pair required)."
            elif is_crossmodal and num_images < 2:
                task_type = "optical_sar"
                intent = "CROSS_MODAL_FUSION"
                reasoning = "Query requests optical-SAR fusion (optical and SAR images required)."
            else:
                task_type = "vqa_caption"
                intent = "SCENE_VQA_CAPTIONING"
                reasoning = "Query requests VQA or scene description."

        # Extract entities
        extracted_entities = []
        for category, words in self.entity_keywords.items():
            if any(w in q_lower for w in words):
                extracted_entities.append(category)
        if not extracted_entities:
            extracted_entities = ["general_land_cover"]

        # Structured query representation required by Section 13
        structured_representation = {
            "task": task_type,
            "requires_temporal_pair": (task_type == "change_vqa"),
            "requires_cross_modal_pair": (task_type == "optical_sar"),
            "requires_grounding": (task_type == "grounding"),
            "target_entities": extracted_entities,
            "requested_output": "text_and_visual_evidence"
        }

        return {
            "query": query,
            "task_type": task_type,
            "intent": intent,
            "interpretation_reasoning": reasoning,
            "structured_query": structured_representation,
            "extracted_entities": extracted_entities,
            "spatial_relations": ["Global Scene Extent"] if not is_grounding else ["Target Region Grounding"],
            "temporal_relations": ["Bi-Temporal Shift (T1 -> T2)"] if is_change else ["Single Observation"],
            "required_evidences": ["Natural Language Synthesis", "Visual Evidence Overlay"]
        }
