from typing import Dict, Any

class AnswerGenerator:
    """
    11. ANSWER GENERATOR
    Synthesizes natural language answer, evidence-linked explanation, trust & uncertainty summary,
    spatial statistics, and autonomous discovery highlights into a cohesive response payload.
    """
    def __init__(self):
        pass

    def synthesize(self, query_plan: Dict[str, Any], tool_output: Dict[str, Any], fusion_output: Dict[str, Any], trust_output: Dict[str, Any], discovery_output: Dict[str, Any]) -> Dict[str, Any]:
        text_response = tool_output.get("text_response", "")
        reliability_score = trust_output["reliability_score"]
        rating = trust_output["reliability_rating"]

        # Synthesize evidence-linked explanation
        ev_summary = " | ".join([f"[{ev['modality']}] {ev['claim']}" for ev in fusion_output["evidences"][:3]])
        
        linked_explanation = (
            f"Evidence-Linked Justification:\n"
            f"The answer is backed by multi-modal evidence provenance: {ev_summary}. "
            f"Conflict Resolution: {fusion_output.get('conflict_resolution', 'None')}"
        )

        return {
            "final_answer": text_response,
            "evidence_explanation": linked_explanation,
            "reliability_score": reliability_score,
            "reliability_rating": rating,
            "discoveries_summary": discovery_output["summary"],
            "discoveries": discovery_output["discoveries"]
        }
