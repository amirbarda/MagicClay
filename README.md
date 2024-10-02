# MagicClay: Sculpting Meshes with Generative Neural Fields


[Project Page](https://amir90.github.io/MagicClay.github.io/) | [Paper](https://arxiv.org/pdf/2403.02460.pdf) |

This is the official implementation of MagicClay, in the form of a [Threestudio](https://github.com/threestudio-project/threestudio) plugin.

![alt text](https://github.com/amir90/MagicClay/blob/main/assets/teaser.png?raw=true)


Installation

1. follow the installation instruction for [Threestudio](https://github.com/threestudio-project/threestudio)
2. install torch-scatter
```pip install torch-scatter``` 
3. download demo data [here](https://drive.google.com/drive/folders/1FT6CuIwp2qA9JKN2SA6mqg7jabMrbDaf?usp=sharing)
4. to run:
```python launch.py --config custom/threestudio-magicclay/configs/magicclay-sd.yaml --gpu 7 --train "system.implicit.system.prompt_processor.prompt=a 3d model of a minotaur in t-pose" \
"system.explicit.system.geometry.allowed_vertices_path=custom/threestudio-magicclay/demo_data/textured_man/allowed_vertices_ids.txt" \
"system.explicit.system.geometry.initial_mesh_path=custom/threestudio-magicclay/demo_data/textured_man/base_mesh.obj" \
"system.explicit.system.geometry.initial_mtl_path=custom/threestudio-magicclay/demo_data/textured_man/base_mesh.mtl" \
"system.explicit.system.geometry.animation_weights_path=custom/threestudio-magicclay/demo_data/textured_man/weights_kicking.txt" \
"system.implicit.system.geometry.sdf_bias=mesh:custom/threestudio-magicclay/demo_data/textured_man/base_mesh.obj"
```

