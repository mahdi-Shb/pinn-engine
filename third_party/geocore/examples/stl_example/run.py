"""Every NN input of a 3D file, with part.output().

    python examples/stl_example/run.py [file.stl]
"""
import os
import sys

import geocore as gc
from geocore import render

here = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "PicoGK_Modeler_preview.stl")
out = os.path.join(here, "output")

part = gc.load_geometry(path)                 # Mesh (repaired only if it has holes)
print(part, "volume %.1f mm^3" % part.volume())

part_mg = part.output("MESH_GRAPH")                         # every MESH_GRAPH array
part_pc = part.output("POINT_CLOUD", n_points=4096, mode="mixed")
part_vg = part.output("VOXEL_GRID", shape=64, channels=["occupancy", "sdf", "boundary"])
part_xyz = part.output("MESH_GRAPH", coordinates=True, node_features=True)   # only two arrays

for o in (part_mg, part_pc, part_vg, part_xyz):
    print(o, "\n")

for o in (part_mg, part_pc, part_vg):
    print("saved", o.save(out + os.sep))
render.render_mesh(part, os.path.join(out, "object.png"), views=4, width=600, height=500,
                   title=os.path.basename(path))
render.render_point_cloud(part_pc, os.path.join(out, "point_cloud.png"))
render.render_voxel_grid(part_vg, os.path.join(out, "voxel_grid.png"))
