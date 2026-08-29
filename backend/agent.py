import time
import os

from backend.utils_geotiff import load_and_inspect_image
from backend.perception import PerceptionLayer
from backend.query_engine import QueryUnderstandingEngine
from backend.agent_planner import AgenticMissionPlanner
from backend.registry import ToolRegistry
from backend.evidence_fusion import MultiModalEvidenceFusion
from backend.evidence_graph import EvidenceGraphBuilder
from backend.trust_uncertainty import TrustAndUncertaintyEngine
from backend.discovery_engine import DiscoveryEngine
from backend.copilot import InvestigationCopilot
from backend.evidence_generator import EvidenceGenerator
from backend.answer_generator import AnswerGenerator
from backend.audit_trace import AuditTraceGenerator

class SatQueryAgent:
    """
    SATQUERY AI MASTER AGENT
    Orchestrates end-to-end multimodal remote sensing query intelligence across all 14 workflow steps.
    """
    def __init__(self, exports_dir: str = "exports"):
        self.perception = PerceptionLayer()
        self.query_engine = QueryUnderstandingEngine()
        self.planner = AgenticMissionPlanner()
        self.registry = ToolRegistry()
        self.fusion = MultiModalEvidenceFusion()
        self.graph_builder = EvidenceGraphBuilder()
        self.trust_engine = TrustAndUncertaintyEngine()
        self.discovery_engine = DiscoveryEngine()
        self.copilot = InvestigationCopilot()
        self.evidence_generator = EvidenceGenerator(exports_dir=exports_dir)
        self.answer_generator = AnswerGenerator()
        self.audit_tracer = AuditTraceGenerator()
        self.exports_dir = exports_dir

    def process_query(self, query: str, image_paths: list, force_task: str = "auto") -> dict:
        start_time = time.time()

        # Step 1: Input Load & Perception Layer
        images_rgb = []
        image_metadatas = []
        for path in image_paths:
            rgb, meta = load_and_inspect_image(path)
            images_rgb.append(rgb)
            image_metadatas.append(meta)

        perception_plan = self.perception.inspect_and_validate(image_metadatas)
        perception_plan["metadatas"] = image_metadatas

        # Step 2: Query Understanding Engine
        query_plan = self.query_engine.parse_query(query, num_images=len(image_paths), force_task=force_task)

        # Step 3: Agentic Mission Planner
        mission_plan = self.planner.plan_mission(query_plan, perception_plan)

        # Step 4: Model Registry Execution
        tool_output = self.registry.route_and_execute(query_plan["task_type"], images_rgb, query, image_metadatas)

        # Step 5: Multi-Modal Evidence Fusion
        fusion_output = self.fusion.fuse_evidence(query_plan, tool_output, perception_plan)

        # Step 6: Evidence Graph Builder
        graph_output = self.graph_builder.build_graph(query_plan, tool_output, fusion_output)

        # Step 7: Trust & Uncertainty Engine
        trust_output = self.trust_engine.evaluate_trust(tool_output, perception_plan, query_plan)

        # Step 8: Discovery Engine ("Beyond Your Query")
        discovery_output = self.discovery_engine.run_secondary_scan(tool_output, query_plan, perception_plan)

        # Step 9: Investigation Copilot
        copilot_suggestions = self.copilot.generate_suggestions(query_plan, tool_output, discovery_output)

        # Step 10: Visual Evidence Generator
        visual_artifacts = self.evidence_generator.generate_and_save_artifacts(tool_output, trust_output)

        # Step 11: Answer Generator
        answer_output = self.answer_generator.synthesize(query_plan, tool_output, fusion_output, trust_output, discovery_output)

        execution_time_ms = round((time.time() - start_time) * 1000, 2)

        # Step 12: Observable Audit Trace & PDF Report
        trace_data = self.audit_tracer.build_trace(
            query, perception_plan, query_plan, mission_plan, tool_output,
            fusion_output, trust_output, discovery_output, answer_output, execution_time_ms
        )

        pdf_filename = f"SatQuery_Audit_Report_{int(start_time*1000)}.pdf"
        pdf_path = os.path.join(self.exports_dir, pdf_filename)
        self.audit_tracer.generate_pdf_report(trace_data, pdf_path)
        pdf_url = f"/exports/{pdf_filename}"

        return {
            "query": query,
            "task_type": query_plan["task_type"],
            "intent": query_plan["intent"],
            "specialist_tool": tool_output["specialist_tool"],
            "confidence": tool_output["confidence"],
            "reliability_score": trust_output["reliability_score"],
            "reliability_rating": trust_output["reliability_rating"],
            "text_response": answer_output["final_answer"],
            "evidence_explanation": answer_output["evidence_explanation"],
            "execution_time_ms": execution_time_ms,
            "perception": perception_plan,
            "query_analysis": query_plan,
            "mission_plan": mission_plan,
            "evidence_fusion": fusion_output,
            "evidence_graph": graph_output,
            "trust_and_uncertainty": {
                "reliability_score": trust_output["reliability_score"],
                "reliability_rating": trust_output["reliability_rating"],
                "model_confidence_pct": trust_output["model_confidence_pct"],
                "cross_model_agreement_pct": trust_output["cross_model_agreement_pct"],
                "spatial_consistency_pct": trust_output["spatial_consistency_pct"],
                "temporal_consistency_pct": trust_output["temporal_consistency_pct"],
                "conflict_flags": trust_output["conflict_flags"]
            },
            "discoveries": discovery_output["discoveries"],
            "copilot_suggestions": copilot_suggestions,
            "visual_artifacts": visual_artifacts,
            "pdf_report_url": pdf_url,
            "execution_trace": trace_data["execution_trace"]
        }
