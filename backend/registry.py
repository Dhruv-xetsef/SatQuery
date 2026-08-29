from typing import List, Dict, Any
from backend.models.vqa_caption_tool import VQACaptionTool
from backend.models.grounding_tool import GroundingTool
from backend.models.change_vqa_tool import ChangeVQATool
from backend.models.optical_sar_tool import OpticalSARTool

class ToolRegistry:
    """
    4. MODEL / TOOL REGISTRY
    Maintains available specialist remote-sensing tools and routes dynamic execution.
    """
    def __init__(self):
        self.vqa_tool = VQACaptionTool()
        self.grounding_tool = GroundingTool()
        self.change_tool = ChangeVQATool()
        self.optical_sar_tool = OpticalSARTool()

    def get_available_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "vqa_caption_tool",
                "name": "RS-VQA & Scene Description Specialist",
                "benchmark": "BigEarthNet & RSVQA",
                "input_scope": "Single Optical or SAR image",
                "description": "Answers natural language queries and extracts land-cover taxonomy using BigEarthNet fine-tuned vision backbone."
            },
            {
                "id": "grounding_tool",
                "name": "RS Text-Guided Region Grounding Specialist",
                "benchmark": "VRSBench",
                "input_scope": "Single Optical or SAR image",
                "description": "Localizes query target entities into normalized bounding boxes and spatial segmentation masks."
            },
            {
                "id": "change_vqa_tool",
                "name": "Bi-Temporal Change Understanding & CDVQA Specialist",
                "benchmark": "CDVQA & ISRO Change Set",
                "input_scope": "Bi-temporal image pair (T1, T2)",
                "description": "Detects land-cover conversion over time, quantifies area shifts, and outputs Spatial Change Maps."
            },
            {
                "id": "optical_sar_tool",
                "name": "Cross-Modal Optical-SAR Fusion Specialist",
                "benchmark": "BigEarthNet & ISRO Cartosat/RISAT",
                "input_scope": "Co-registered Optical + SAR image pair",
                "description": "Combines optical spectral indices with SAR microwave structural backscatter for cloud-resilient joint reasoning."
            }
        ]

    def route_and_execute(self, task_type: str, images_rgb: list, query: str, metadatas: list) -> Dict[str, Any]:
        if task_type == "grounding":
            res = self.grounding_tool.execute(images_rgb[0], query, metadatas[0])
        elif task_type == "change_vqa":
            img1 = images_rgb[0]
            img2 = images_rgb[1] if len(images_rgb) > 1 else images_rgb[0]
            m1 = metadatas[0]
            m2 = metadatas[1] if len(metadatas) > 1 else metadatas[0]
            res = self.change_tool.execute(img1, img2, query, m1, m2)
        elif task_type == "optical_sar":
            img1 = images_rgb[0]
            img2 = images_rgb[1] if len(images_rgb) > 1 else images_rgb[0]
            m1 = metadatas[0]
            m2 = metadatas[1] if len(metadatas) > 1 else metadatas[0]
            res = self.optical_sar_tool.execute(img1, img2, query, m1, m2)
        else: # Default vqa_caption
            res = self.vqa_tool.execute(images_rgb[0], query, metadatas[0])

        return res
