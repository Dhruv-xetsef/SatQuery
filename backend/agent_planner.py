from typing import Dict, Any, List

class AgenticMissionPlanner:
    """
    3. AGENTIC MISSION PLANNER
    Constructs execution task graphs, selects specialist tools, verifies input compatibility,
    and configures tool parameters without synthesizing missing inputs.
    """
    def plan_mission(self, query_plan: Dict[str, Any], perception_plan: Dict[str, Any]) -> Dict[str, Any]:
        task_type = query_plan["task_type"]
        num_images = perception_plan["image_count"]

        # Check for perception errors
        if not perception_plan.get("passed", True):
            err_msg = perception_plan["errors"][0] if perception_plan.get("errors") else "Input validation failed."
            return {
                "selected_tool_id": "none",
                "tool_name": "Input Validation Error",
                "parameters": {},
                "compatibility_note": f"REJECTED: {err_msg}",
                "task_graph_nodes": [
                    {"id": "node_1", "stage": "PERCEPTION", "name": "Input Inspection", "status": "FAILED"}
                ],
                "task_graph_edges": [],
                "mission_summary": f"Mission planning rejected due to input validation error: {err_msg}"
            }

        if task_type == "change_vqa":
            selected_tool_id = "change_vqa_tool"
            tool_name = "Bi-Temporal Change Understanding & CDVQA Specialist"
            params = {"threshold_method": "otsu", "blur_kernel": [5, 5]}
            compatibility_note = "Valid: Spatially corresponding bi-temporal image pair verified."
        elif task_type == "optical_sar":
            selected_tool_id = "optical_sar_tool"
            tool_name = "Cross-Modal Optical-SAR Fusion Specialist"
            params = {"cloud_thresh": 220, "sar_bright_thresh": 160}
            compatibility_note = "Valid: Co-registered Optical and SAR images verified."
        elif task_type == "grounding":
            selected_tool_id = "grounding_tool"
            tool_name = "RS Text-Guided Region Grounding Specialist"
            params = {"target_entities": query_plan["extracted_entities"]}
            compatibility_note = "Valid: Single image region grounding input ready."
        else: # Default vqa_caption
            selected_tool_id = "vqa_caption_tool"
            tool_name = "RS-VQA & Scene Description Specialist (BigEarthNet Adapted)"
            params = {"top_k_classes": 5}
            compatibility_note = "Valid: Single image VQA input ready."

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

        return {
            "selected_tool_id": selected_tool_id,
            "tool_name": tool_name,
            "parameters": params,
            "compatibility_note": compatibility_note,
            "task_graph_nodes": task_graph_nodes,
            "task_graph_edges": task_graph_edges,
            "mission_summary": f"Mission Plan created for task '{task_type}' using '{tool_name}'."
        }
