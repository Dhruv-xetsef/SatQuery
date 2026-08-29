import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

class AuditTraceGenerator:
    """
    12. AUDIT / EXECUTION TRACE & REPORT GENERATOR
    Produces a fully observable, auditable execution trace and generates downloadable PDF reports.
    """
    def __init__(self):
        pass

    def build_trace(self, query: str, perception_plan: dict, query_plan: dict, mission_plan: dict, tool_output: dict, fusion_output: dict, trust_output: dict, discovery_output: dict, answer_output: dict, execution_time_ms: float) -> dict:
        trace_logs = [
            {"stage": "Perception Layer", "detail": f"Format & CRS validated. Mode: {perception_plan['relationship_type']} ({perception_plan['image_count']} image(s)). Status: {perception_plan['status']}"},
            {"stage": "Query Engine", "detail": f"Classified intent: '{query_plan['intent']}'. Extracted entities: {query_plan['extracted_entities']}. Task: '{query_plan['task_type']}'"},
            {"stage": "Mission Planner", "detail": f"Selected tool: '{mission_plan['tool_name']}' with params {mission_plan['parameters']}. Generated {len(mission_plan['task_graph_nodes'])} node task-graph."},
            {"stage": "Specialist Tool Execution", "detail": f"Executed specialist tool '{tool_output['specialist_tool']}' with model confidence {tool_output['confidence']*100:.1f}%."},
            {"stage": "Evidence Fusion", "detail": f"Aligned {len(fusion_output['evidences'])} evidence streams. {fusion_output['conflict_resolution']}"},
            {"stage": "Trust & Uncertainty", "detail": f"Calculated Reliability Score: {trust_output['reliability_score']}% ({trust_output['reliability_rating']}). Cross-model agreement: {trust_output['cross_model_agreement_pct']}%."},
            {"stage": "Discovery Engine", "detail": f"Autonomous scan completed: {discovery_output['summary']}"},
            {"stage": "Answer Generator", "detail": f"Final evidence-linked response synthesized in {execution_time_ms:.1f} ms."}
        ]

        return {
            "query": query,
            "task_type": query_plan["task_type"],
            "task_label": mission_plan["tool_name"],
            "specialist_tool": tool_output["specialist_tool"],
            "confidence": tool_output["confidence"],
            "reliability_score": trust_output["reliability_score"],
            "reliability_rating": trust_output["reliability_rating"],
            "execution_time_ms": execution_time_ms,
            "text_response": answer_output["final_answer"],
            "image_metadatas": perception_plan.get("metadatas", []),
            "execution_trace": trace_logs
        }

    def generate_pdf_report(self, trace_data: dict, output_filepath: str) -> str:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        doc = SimpleDocTemplate(output_filepath, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        story = []

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=20, leading=24, textColor=colors.HexColor("#0f172a"), spaceAfter=8)
        subtitle_style = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor("#64748b"), spaceAfter=14)
        heading2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=13, leading=16, textColor=colors.HexColor("#1e293b"), spaceBefore=12, spaceAfter=6)
        body_style = ParagraphStyle('BodyTextCustom', parent=styles['BodyText'], fontSize=9.5, leading=13, textColor=colors.HexColor("#334155"))

        # Title
        story.append(Paragraph("SatQuery AI - Multimodal Remote Sensing Evidence Report", title_style))
        story.append(Paragraph("Autonomous Agentic Execution & Evidence Provenance Trace", subtitle_style))
        story.append(Spacer(1, 4))

        # Table Summary
        summary_table_data = [
            [Paragraph("<b>Query:</b>", body_style), Paragraph(f"<i>\"{trace_data.get('query', 'N/A')}\"</i>", body_style)],
            [Paragraph("<b>Task Category:</b>", body_style), Paragraph(str(trace_data.get('task_type', 'N/A')).upper(), body_style)],
            [Paragraph("<b>Specialist Tool:</b>", body_style), Paragraph(trace_data.get('specialist_tool', 'N/A'), body_style)],
            [Paragraph("<b>Reliability Score:</b>", body_style), Paragraph(f"<b>{trace_data.get('reliability_score', 90)}% ({trace_data.get('reliability_rating', 'HIGH TRUST')})</b>", body_style)],
            [Paragraph("<b>Execution Time:</b>", body_style), Paragraph(f"{trace_data.get('execution_time_ms', 0):.1f} ms", body_style)]
        ]
        t = Table(summary_table_data, colWidths=[140, 390])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

        # Text Answer
        story.append(Paragraph("1. Agent Synthesis & Visual-Language Answer", heading2))
        resp_text = trace_data.get("text_response", "").replace("\n", "<br/>")
        story.append(Paragraph(resp_text, body_style))
        story.append(Spacer(1, 10))

        # Observable Pipeline Trace Log
        story.append(Paragraph("2. Auditable Agent Pipeline Execution Trace", heading2))
        log_text = "<br/>".join([f"• <b>[{log.get('stage')}]</b> {log.get('detail')}" for log in trace_data.get("execution_trace", [])])
        story.append(Paragraph(log_text, body_style))

        doc.build(story)
        return output_filepath
