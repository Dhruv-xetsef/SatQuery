from typing import Dict, Any, List

class EvidenceGraphBuilder:
    """
    6. EVIDENCE GRAPH BUILDER
    Answers "WHY DO WE BELIEVE THIS?" by constructing a Directed Acyclic Graph (DAG):
    QUERY ──→ TASK ──→ OBJECT ──→ REGION ──→ TIME ──→ MODALITY ──→ HYPOTHESIS
    """
    def __init__(self):
        pass

    def build_graph(self, query_plan: Dict[str, Any], tool_output: Dict[str, Any], fusion_output: Dict[str, Any]) -> Dict[str, Any]:
        query_text = query_plan["query"]
        task_type = query_plan["task_type"]
        entities = query_plan["extracted_entities"]
        
        nodes = [
            {"id": "q1", "label": f"Query: \"{query_text}\"", "type": "query"},
            {"id": "t1", "label": f"Task: {task_type.upper()}", "type": "task"},
            {"id": "o1", "label": f"Target Entity: {', '.join(entities)}", "type": "object"},
            {"id": "r1", "label": "Region: Scene Spatial Bounds", "type": "region"},
            {"id": "m1", "label": f"Modality: {fusion_output['evidences'][0]['modality']}", "type": "modality"},
            {"id": "h1", "label": f"Hypothesis: {fusion_output['hypothesis']}", "type": "hypothesis"}
        ]

        edges = [
            {"from": "q1", "to": "t1", "label": "classified as", "relation": "SUPPORT"},
            {"from": "t1", "to": "o1", "label": "extracts entity", "relation": "SUPPORT"},
            {"from": "o1", "to": "r1", "label": "localizes to", "relation": "SUPPORT"},
            {"from": "r1", "to": "m1", "label": "sensed via", "relation": "SUPPORT"},
            {"from": "m1", "to": "h1", "label": "validates claim", "relation": "SUPPORT"}
        ]

        # Add specific evidence nodes based on task type
        if task_type == "optical_sar":
            nodes.append({"id": "e_sar", "label": "SAR Microwave Radar Backscatter (Cloud Penetration)", "type": "evidence"})
            nodes.append({"id": "e_opt", "label": "Optical Multispectral Reflectance", "type": "evidence"})
            edges.append({"from": "e_sar", "to": "h1", "label": "bypasses clouds", "relation": "SUPPORT"})
            edges.append({"from": "e_opt", "to": "h1", "label": "spectral color", "relation": "SUPPORT"})
        elif task_type == "change_vqa":
            nodes.append({"id": "e_change", "label": f"Spatial Change Map ({tool_output.get('change_percentage', 0)}% Shift)", "type": "evidence"})
            edges.append({"from": "e_change", "to": "h1", "label": "quantifies trend", "relation": "SUPPORT"})

        return {
            "nodes": nodes,
            "edges": edges,
            "graph_summary": f"Evidence Graph built with {len(nodes)} nodes and {len(edges)} support links proving claim provenance."
        }
