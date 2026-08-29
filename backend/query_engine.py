import re
from typing import Dict, Any, List

class QueryUnderstandingEngine:
    """
    2. QUERY UNDERSTANDING ENGINE
    Extracts query intent, target entities/objects, spatial relationships,
    temporal relationships, and identifies required evidence types.
    """
    def __init__(self):
        self.change_keywords = [
            "change", "changed", "dates", "time", "temporal", "between", 
            "increase", "decrease", "deforestation", "expansion", "before and after",
            "bitemporal", "converted", "remained"
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

    def parse_query(self, query: str, num_images: int = 1, force_task: str = None) -> Dict[str, Any]:
        q_lower = query.lower()

        # 1. Intent Extraction
        if force_task and force_task != "auto":
            task_type = force_task
            reasoning = f"Task type explicitly set to '{force_task}' by user."
            intent = force_task.upper()
        else:
            is_change = any(k in q_lower for k in self.change_keywords) or (num_images == 2 and ("change" in q_lower or "date" in q_lower or "vs" in q_lower))
            is_crossmodal = any(k in q_lower for k in self.crossmodal_keywords) or (num_images == 2 and ("optical" in q_lower or "sar" in q_lower))
            is_grounding = any(k in q_lower for k in self.grounding_keywords)

            if is_change and num_images == 2:
                task_type = "change_vqa"
                intent = "BI_TEMPORAL_CHANGE_ANALYSIS"
                reasoning = "Query requests bi-temporal change detection, land-cover conversion description, and spatial change map."
            elif is_crossmodal and num_images == 2:
                task_type = "optical_sar"
                intent = "CROSS_MODAL_FUSION"
                reasoning = "Query requests joint optical-SAR multisensor feature extraction and cloud-resilient analysis."
            elif is_grounding:
                task_type = "grounding"
                intent = "REGION_GROUNDING"
                reasoning = "Query requests spatial localization, bounding box coordinates, and region delineation."
            elif is_change and num_images == 1:
                task_type = "change_vqa"
                intent = "SINGLE_IMAGE_CHANGE_QUERY"
                reasoning = "Query asks about changes (bi-temporal image pair recommended)."
            else:
                task_type = "vqa_caption"
                intent = "SCENE_VQA_CAPTIONING"
                reasoning = "Query requests visual question answering and general remote-sensing scene description."

        # 2. Entity / Object Extraction
        extracted_entities = []
        for category, words in self.entity_keywords.items():
            if any(w in q_lower for w in words):
                extracted_entities.append(category)
        if not extracted_entities:
            extracted_entities = ["general_land_cover"]

        # 3. Spatial Relationship Extraction
        spatial_relations = []
        if any(w in q_lower for w in ["near", "adjacent", "surrounding", "close to"]):
            spatial_relations.append("Proximity / Adjacency")
        if any(w in q_lower for w in ["where", "location", "quadrant", "region", "box"]):
            spatial_relations.append("Spatial Bounding / Region Grounding")
        if not spatial_relations:
            spatial_relations.append("Global Scene Extent")

        # 4. Temporal Relationship Extraction
        temporal_relations = []
        if any(w in q_lower for w in ["between", "from t1 to t2", "dates", "before and after"]):
            temporal_relations.append("Bi-Temporal Shift (T1 -> T2)")
        if any(w in q_lower for w in ["increase", "decrease", "grew", "shrank"]):
            temporal_relations.append("Area Quantified Trend Shift")
        if not temporal_relations:
            temporal_relations.append("Single Observation Moment")

        # 5. Required Evidence Identification
        required_evidences = ["Natural Language Synthesis", "Model Confidence Score"]
        if task_type == "grounding":
            required_evidences.extend(["Bounding Box Coordinates", "Spatial Segmentation Overlay"])
        elif task_type == "change_vqa":
            required_evidences.extend(["Binary Change Mask", "Heatmap Change Map Overlay", "Land-Cover Trend Quantifier"])
        elif task_type == "optical_sar":
            required_evidences.extend(["Optical Cloud Penetration Mask", "SAR Backscatter Double-Bounce Overlay", "Fused FCC Overlay"])
        else:
            required_evidences.extend(["BigEarthNet Class Distribution Graph", "High-Probability Region Highlights"])

        return {
            "query": query,
            "task_type": task_type,
            "intent": intent,
            "interpretation_reasoning": reasoning,
            "extracted_entities": extracted_entities,
            "spatial_relations": spatial_relations,
            "temporal_relations": temporal_relations,
            "required_evidences": required_evidences
        }
