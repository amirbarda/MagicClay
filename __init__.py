import threestudio
from packaging.version import Version

if hasattr(threestudio, "__version__") and Version(threestudio.__version__) >= Version(
    "0.2.0"
):
    pass
else:
    if hasattr(threestudio, "__version__"):
        print(f"[INFO] threestudio version: {threestudio.__version__}")
    raise ValueError(
        "threestudio version must be >= 0.2.0, please update threestudio by pulling the latest version from github"
    )

from .system.hybrid_base import HybridSystem
from .system.magicclay import MagicClay
from .data.hybrid import HybridRandomCameraDataModule
from .geometry.dynamic_mesh import DynamicMesh
from .geometry.implicit_sdf import ImplicitSDF
from .renderer.neus_volume_renderer2 import NeuSVolumeRenderer
from .renderer.mesh_renderer import NVDiffRasterizer
from .materials.diffuse_with_point_light_material import DiffuseWithPointLightMaterial
