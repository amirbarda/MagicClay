# minotaur
python launch.py --config custom/threestudio-magicclay/configs/magicclay-sd.yaml --gpu 1 --train "system.implicit.system.prompt_processor.prompt=a 3d model of a minotaur in t-pose" \
"system.explicit.system.geometry.allowed_vertices_path=custom/threestudio-magicclay/demo_data/textured_man/allowed_vertices_ids.txt" \
"system.explicit.system.geometry.initial_mesh_path=custom/threestudio-magicclay/demo_data/textured_man/base_mesh.obj" \
"system.explicit.system.geometry.initial_mtl_path=custom/threestudio-magicclay/demo_data/textured_man/base_mesh.mtl" \
"system.explicit.system.geometry.animation_weights_path=custom/threestudio-magicclay/demo_data/textured_man/weights_kicking.txt" \
"system.implicit.system.geometry.sdf_bias=mesh:custom/threestudio-magicclay/demo_data/textured_man/base_mesh.obj"

# ninja
python launch.py --config custom/threestudio-magicclay/configs/magicclay-sd.yaml --gpu 2 --train "system.implicit.system.prompt_processor.prompt=a 3d model of a ninja in t-pose" \
"system.explicit.system.geometry.allowed_vertices_path=custom/threestudio-magicclay/demo_data/textured_man/allowed_vertices_ids.txt" \
"system.explicit.system.geometry.initial_mesh_path=custom/threestudio-magicclay/demo_data/textured_man/base_mesh.obj" \
"system.explicit.system.geometry.initial_mtl_path=custom/threestudio-magicclay/demo_data/textured_man/base_mesh.mtl" \
"system.explicit.system.geometry.animation_weights_path=custom/threestudio-magicclay/demo_data/textured_man/weights_kicking.txt" \
"system.implicit.system.geometry.sdf_bias=mesh:custom/threestudio-magicclay/demo_data/textured_man/base_mesh.obj" \
"trainer.max_steps=7200"

# spot with angel wings
python launch.py --config custom/threestudio-magicclay/configs/magicclay-sd.yaml --gpu 3 --train "system.implicit.system.prompt_processor.prompt=a cow with angel wings" \
"system.explicit.system.geometry.allowed_vertices_path=custom/threestudio-magicclay/demo_data/spot_wings/allowed_vertices_ids.txt" \
"system.explicit.system.geometry.initial_mesh_path=custom/threestudio-magicclay/demo_data/spot_wings/spot.obj" \
"system.explicit.system.geometry.initial_mtl_path=custom/threestudio-magicclay/demo_data/spot_wings/spot.mtl" \
"system.implicit.system.geometry.sdf_bias=mesh:custom/threestudio-magicclay/demo_data/spot_wings/spot.obj"