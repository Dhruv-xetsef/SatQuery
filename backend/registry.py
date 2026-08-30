from typing import List, Dict, Any
from backend.models.vqa_caption_tool import VQACaptionTool
from backend.models.grounding_tool import GroundingTool
from backend.models.change_vqa_tool import ChangeVQATool
from backend.models.optical_sar_tool import OpticalSARTool
from backend.baselines.classical_change import ClassicalChangeBaseline
from backend.baselines.classical_optical_sar import ClassicalOpticalSARBaseline

class ToolRegistry:
    """
    4. MODEL / TOOL REGISTRY
    Maintains available specialist remote-sensing tools & classical baselines,
    routing dynamic execution based on task type and modality requirements.
    """
    def __init__(self):
        self.vqa_tool = VQACaptionTool()
        self.grounding_tool = GroundingTool()
        self.change_tool = ChangeVQATool()
        self.optical_sar_tool = OpticalSARTool()
        self.classical_change = ClassicalChangeBaseline()
        self.classical_optical_sar = ClassicalOpticalSARBaseline()

    def get_available_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "vqa_caption_tool",
                "name": "RS-VQA & Scene Description Specialist",
                "benchmark": "BigEarthNet & RSVQA",
                "input_scope": "Single Optical or SAR image",
                "type": "vision-language",
                "description": "Answers natural language queries using BigEarthNet fine-tuned visual adapter."
            },
            {
                "id": "grounding_tool",
                "name": "RS Text-Guided Region Grounding Specialist",
                "benchmark": "VRSBench",
                "input_scope": "Single Optical or SAR image",
                "type": "grounding",
                "description": "Localizes query target entities into bounding boxes and spatial segmentation masks."
            },
            {
                "id": "change_vqa_tool",
                "name": "Bi-Temporal Change Understanding & CDVQA Specialist",
                "benchmark": "CDVQA & ISRO Change Set",
                "input_scope": "Bi-temporal image pair (T1, T2)",
                "type": "temporal",
                "description": "Extracts bi-temporal visual feature differences, quantifies change area, and produces Spatial Change Maps."
            },
            {
                "id": "optical_sar_tool",
                "name": "Cross-Modal Optical-SAR Fusion Specialist",
                "benchmark": "BigEarthNet & ISRO Cartosat/RISAT",
                "input_scope": "Co-registered Optical + SAR pair",
                "type": "multimodal-fusion",
                "description": "Fuses optical spectral reflectance with SAR microwave radar backscatter for cloud-resilient analysis."
            },
            {
                "id": "classical_change_baseline",
                "name": "Classical Image Differencing Baseline",
                "benchmark": "Classical CV Baseline",
                "input_scope": "Bi-temporal image pair",
                "type": "baseline",
                "description": "Simple pixel-level intensity differencing and Otsu thresholding."
            },
            {
                "id": "classical_optical_sar_baseline",
                "name": "Classical Optical-SAR Threshold Baseline",
                "benchmark": "Classical CV Baseline",
                "input_scope": "Optical + SAR pair",
                "type": "baseline",
                "description": "Simple channel brightness thresholding for optical/SAR feature delineation."
            }
        ]

    def route_and_execute(self, task_type: str, images_rgb: list, query: str, metadatas: list, use_baseline: bool = False) -> Dict[str, Any]:
        num_images = len(images_rgb)

        if task_type == "grounding":
            return self.grounding_tool.execute(images_rgb[0], query, metadatas[0])
        elif task_type == "change_vqa":
            if num_images < 2:
                raise ValueError("Two spatially corresponding images are required for bi-temporal analysis.")
            if use_baseline:
                return self.classical_change.execute(images_rgb[0], images_rgb[1], query, metadatas[0], metadatas[1])
            return self.change_tool.execute(images_rgb[0], images_rgb[1], query, metadatas[0], metadatas[1])
        elif task_type == "optical_sar":
            if num_images < 2:
                raise ValueError("Cross-modal analysis requires one optical/multispectral image and one SAR image.")
            if use_baseline:
                return self.classical_optical_sar.execute(images_rgb[0], images_rgb[1], query, metadatas[0], metadatas[1])
            return self.optical_sar_tool.execute(images_rgb[0], images_rgb[1], query, metadatas[0], metadatas[1])
        else: # Default vqa_caption
            return self.vqa_tool.execute(images_rgb[0], query, metadatas[0])
