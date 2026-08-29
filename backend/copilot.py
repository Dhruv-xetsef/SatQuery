from typing import Dict, Any, List

class InvestigationCopilot:
    """
    9. INVESTIGATION COPILOT
    Generates contextual, evidence-driven "Suggested Next Questions" enabling analysts
    to perform interactive follow-up investigations and deep-dive drills.
    """
    def __init__(self):
        pass

    def generate_suggestions(self, query_plan: Dict[str, Any], tool_output: Dict[str, Any], discovery_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        task_type = query_plan["task_type"]
        q_lower = query_plan["query"].lower()

        suggestions = []

        if task_type == "change_vqa":
            suggestions = [
                {"query": "Show exactly where the change occurred on a high-contrast mask.", "action": "FILTER_CHANGE_MAP"},
                {"query": "Does SAR radar imagery confirm structural building changes in this zone?", "action": "RUN_SAR_VERIFICATION"},
                {"query": "What is the total area (in hectares) of converted land?", "action": "CALCULATE_AREA_STATS"},
                {"query": "Was the urban expansion within 500 meters of a water body?", "action": "SPATIAL_PROXIMITY_ANALYSIS"},
                {"query": "Show only high-confidence changes above 90% certainty.", "action": "HIGH_CONFIDENCE_FILTER"}
            ]
        elif task_type == "optical_sar":
            suggestions = [
                {"query": "Show the SAR microwave backscatter image independently.", "action": "VIEW_SAR_ONLY"},
                {"query": "Delineate built-up structures hidden under cloud cover.", "action": "CLOUD_PENETRATION_ZOOM"},
                {"query": "Highlight all water bodies extracted using SAR specular reflection.", "action": "SAR_WATER_MASK"},
                {"query": "Compare optical vegetation index (NDVI) with SAR structural roughness.", "action": "NDVI_SAR_CORRELATION"}
            ]
        elif task_type == "grounding":
            suggestions = [
                {"query": "Show exact bounding box coordinates in GeoTIFF lat/lon CRS.", "action": "GET_GEO_COORDS"},
                {"query": "Ground the nearest road or infrastructure network.", "action": "GROUND_INFRASTRUCTURE"},
                {"query": "Calculate distance from grounded region to nearest water body.", "action": "CALCULATE_DISTANCE"}
            ]
        else:
            suggestions = [
                {"query": "Highlight the water body referred to in the query.", "action": "RUN_GROUNDING"},
                {"query": "What is the breakdown of land-cover classes across BigEarthNet taxonomy?", "action": "SHOW_TAXONOMY"},
                {"query": "Run cross-modal SAR check on structural features in this image.", "action": "RUN_SAR_CHECK"},
                {"query": "Are there any unqueried anomalies in this satellite scene?", "action": "SCAN_ANOMALIES"}
            ]

        return suggestions
