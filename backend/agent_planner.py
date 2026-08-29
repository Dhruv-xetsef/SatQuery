from typing import Dict, Any, List

class AgenticMissionPlanner:
    """
    3. AGENTIC MISSION PLANNER
    Generates dynamic task graphs, selects specialist tools from model registry,
    verifies input-task compatibility, configures execution parameters, and orders tool pipeline.
    """
    def __init__(self):
        pass

    def plan_mission(self, query_plan: Dict[str, Any], perception_plan: Dict[str, Any]) -> Dict[str, Any]:
        task_type = query_plan["task_type"]
        num_images = perception_plan["image_count"]
        relationship = perception_plan["relationship_type"]

        # 1. Input ↔ Task Compatibility Validation & Tool Selection
        if task_type == "change_vqa":
            selected_tool_id = "change_vqa_tool"
            tool_name = "Bi-Temporal Change Understanding & CDVQA Specialist"
            params = {"threshold": 35, "blur_kernel": [5, 5], "contour_min_area": 100}
            if num_images < 2:
                compatibility_note = "Warning: 1 image provided for Change Analysis. Synthetic second frame generated for baseline differencing."
            else:
                compatibility_note = "Optimal: Bi-temporal pair provided. Full change detection pipeline engaged."
        elif task_type == "optical_sar":
            selected_tool_id = "optical_sar_tool"
            tool_name = "Cross-Modal Optical-SAR Fusion Specialist"
            params = {"sar_double_bounce_thresh": 160, "water_threshold": 40, "cloud_thresh": 220}
            if num_images < 2:
                compatibility_note = "Warning: Single image provided for Cross-Modal analysis. Synthetic SAR structural layer generated."
            else:
                compatibility_note = "Optimal: Co-registered Optical and SAR images provided."
        elif task_type == "grounding":
            selected_tool_id = "grounding_tool"
            tool_name = "RS Text-Guided Region Grounding Specialist"
            params = {"iou_threshold": 0.5, "target_entities": query_plan["extracted_entities"]}
            compatibility_note = "Single image region grounding ready."
        else: # Default vqa_caption
            selected_tool_id = "vqa_caption_tool"
            tool_name = "RS-VQA & Scene Description Specialist (BigEarthNet Adapted)"
            params = {"top_k_classes": 5, "confidence_thresh": 0.15}
            compatibility_note = "Single image VQA & Scene description ready."

        # 2. Query Decomposition into Action Nodes
        task_graph_nodes = [
            {"id": "node_1", "stage": "PERCEPTION", "name": "Input Inspection & CRS Validation", "status": "COMPLETED"},
            {"id": "node_2", "stage": "QUERY_UNDERSTANDING", "name": f"Intent Parsing ({query_plan['intent']})", "status": "COMPLETED"},
            {"id": "node_3", "stage": "SPECIALIST_EXECUTION", "name": f"Execute {tool_name}", "status": "PENDING"},
            {"id": "node_4", "stage": "EVIDENCE_FUSION", "name": "Multi-Modal Evidence Alignment", "status": "PENDING"},
            {"id": "node_5", "stage": "TRUST_EVALUATION", "name": "Calculate Reliability & Uncertainty Map", "status": "PENDING"},
            {"id": "node_6", "stage": "AUTONOMOUS_DISCOVERY", "name": "Run Secondary Scan for Beyond-Query Insights", "status": "PENDING"},
            {"id": "node_7", "stage": "OUTPUT_SYNTHESIS", "name": "Generate Evidence Answer & PDF Audit Trace", "status": "PENDING"}
        ]

        task_graph_edges = [
            {"from": "node_1", "to": "node_2"},
            {"from": "node_2", "to": "node_3"},
            {"from": "node_3", "to": "node_4"},
            {"from": "node_4", "to": "node_5"},
            {"from": "node_5", "to": "node_6"},
            {"from": "node_6", "to": "node_7"}
        ]

        mission_summary = (
            f"Agent Mission Plan created: Task='{task_type}', Tool='{tool_name}'. "
            f"Pipeline contains {len(task_graph_nodes)} sequential execution nodes. "
            f"Compatibility: {compatibility_note}"
        )

        return {
            "selected_tool_id": selected_tool_id,
            "tool_name": tool_name,
            "parameters": params,
            "compatibility_note": compatibility_note,
            "task_graph_nodes": task_graph_nodes,
            "task_graph_edges": task_graph_edges,
            "mission_summary": mission_summary
        }
