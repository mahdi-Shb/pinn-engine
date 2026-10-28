# Geometry values

## mesh_graph.npz  -  MESH_GRAPH

| array | dtype | shape | min | max | value |
|---|---|---|---|---|---|
| `schema_version` | string | [] |  |  | 1.0 |
| `representation` | string | [] |  |  | MESH_GRAPH |
| `coordinate_system` | string | [] |  |  | cartesian_xyz_right_handed |
| `units` | string | [] |  |  | mm/normalized |
| `normalization_id` | string | [] |  |  | unit_cube:842dae469695 |
| `normalization_center` | float32 | [3] | -0.15 | 21.9 | [-0.15000000596046448, 0.0, 21.899999618530273] |
| `normalization_scale` | float32 | [] | 31.65 | 31.65 | 31.649999618530273 |
| `coordinates` | float32 | [122044, 3] | -0.9431 | 0.9525 |  |
| `node_features` | float32 | [122044, 6] | -18.85 | 189.6 |  |
| `edge_index` | int64 | [2, 732276] | 0 | 1.22e+05 |  |
| `edge_features` | float32 | [732276, 4] | -0.01165 | 0.01674 |  |
| `batch` | int64 | [122044] | 0 | 0 |  |
| `node_feature_names` | string | [6] |  |  | ['nx', 'ny', 'nz', 'vertex_area', 'mean_curvature', 'tag:... |
| `edge_feature_names` | string | [4] |  |  | ['dx', 'dy', 'dz', 'distance'] |
| `faces` | int64 | [244092, 3] | 0 | 1.22e+05 |  |

## point_cloud.npz  -  POINT_CLOUD

| array | dtype | shape | min | max | value |
|---|---|---|---|---|---|
| `schema_version` | string | [] |  |  | 1.0 |
| `representation` | string | [] |  |  | POINT_CLOUD |
| `coordinate_system` | string | [] |  |  | cartesian_xyz_right_handed |
| `units` | string | [] |  |  | mm/normalized |
| `normalization_id` | string | [] |  |  | centroid_unit_sphere:5cd9c71d7121 |
| `normalization_center` | float32 | [3] | -7.996 | 12.3 | [-7.995813846588135, 0.022046100348234177, 12.29640007019... |
| `normalization_scale` | float32 | [] | 43.86 | 43.86 | 43.856781005859375 |
| `coordinates` | float32 | [1, 4096, 3] | -0.5017 | 0.8663 |  |
| `point_features` | float32 | [1, 4096, 5] | -1 | 1 |  |
| `point_feature_names` | string | [5] |  |  | ['nx', 'ny', 'nz', 'sdf', 'is_boundary'] |

## voxel_grid.npz  -  VOXEL_GRID

| array | dtype | shape | min | max | value |
|---|---|---|---|---|---|
| `schema_version` | string | [] |  |  | 1.0 |
| `representation` | string | [] |  |  | VOXEL_GRID |
| `coordinate_system` | string | [] |  |  | cartesian_xyz_right_handed |
| `units` | string | [] |  |  | mm/normalized |
| `normalization_id` | string | [] |  |  | unit_cube:d4010736150e |
| `normalization_center` | float32 | [3] | -0.15 | 21.9 | [-0.15000000596046448, 0.0, 21.899999618530273] |
| `normalization_scale` | float32 | [] | 34.81 | 34.81 | 34.814998626708984 |
| `voxel_fields` | float32 | [1, 3, 48, 64, 96] | -0.1486 | 1.161 |  |
| `channel_names` | string | [3] |  |  | ['occupancy', 'sdf', 'boundary'] |
| `origin` | float32 | [3] | -0.9896 | -0.7002 | [-0.9895833134651184, -0.7002167701721191, -0.75557410717... |
| `spacing` | float32 | [3] | 0.02083 | 0.03215 | [0.02083333395421505, 0.02222910337150097, 0.032152090221... |
| `bounds` | float32 | [2, 3] | -1 | 1 | [[-1.0, -0.711331307888031, -0.7716501355171204], [1.0, 0... |

## voxel_grid_tagged.npz  -  VOXEL_GRID

| array | dtype | shape | min | max | value |
|---|---|---|---|---|---|
| `schema_version` | string | [] |  |  | 1.0 |
| `representation` | string | [] |  |  | VOXEL_GRID |
| `coordinate_system` | string | [] |  |  | cartesian_xyz_right_handed |
| `units` | string | [] |  |  | mm/normalized |
| `normalization_id` | string | [] |  |  | unit_cube:c5adb08588c8 |
| `normalization_center` | float32 | [3] | 0 | 22 | [0.0, 0.0, 22.0] |
| `normalization_scale` | float32 | [] | 33 | 33 | 33.0 |
| `voxel_fields` | float32 | [1, 7, 48, 64, 96] | -0.1446 | 1.144 |  |
| `channel_names` | string | [7] |  |  | ['occupancy', 'sdf', 'tag:base', 'tag:upright', 'tag:boss... |
| `origin` | float32 | [3] | -0.9896 | -0.6861 | [-0.9895833134651184, -0.6860795617103577, -0.74179291725... |
| `spacing` | float32 | [3] | 0.02083 | 0.03157 | [0.02083333395421505, 0.02178030274808407, 0.031565655022... |
| `bounds` | float32 | [2, 3] | -1 | 1 | [[-1.0, -0.6969696879386902, -0.7575757503509521], [1.0, ... |

## parametric_surface.npz  -  PARAMETRIC_SURFACE

| array | dtype | shape | min | max | value |
|---|---|---|---|---|---|
| `schema_version` | string | [] |  |  | 1.0 |
| `representation` | string | [] |  |  | PARAMETRIC_SURFACE |
| `coordinate_system` | string | [] |  |  | cartesian_xy_2d |
| `units` | string | [] |  |  | mm |
| `normalization_id` | string | [] |  |  | none |
| `normalization_center` | float32 | [3] | 0 | 0 | [0.0, 0.0, 0.0] |
| `normalization_scale` | float32 | [] | 1 | 1 | 1.0 |
| `degree` | int64 | [2] | 3 | 3 | [3, 3] |
| `knot_vector_u` | float32 | [16] | 0 | 1 |  |
| `knot_vector_v` | float32 | [14] | 0 | 1 |  |
| `control_points` | float32 | [12, 10, 2] | -4 | 4 |  |
| `weights` | float32 | [12, 10] | 0.8536 | 1 |  |
| `admissible_coefficients` | float32 | [12, 10] | 0 | 1 |  |
| `lifting_coefficients` | float32 | [12, 10, 2] | -1 | 1 |  |
| `parametric_coordinates` | float32 | [1, 4096, 2] | 0.000106 | 0.9999 |  |
| `physical_coordinates` | float32 | [1, 4096, 2] | -4 | 3.999 |  |

## Geometry

- **name**: bracket (filleted)
- **backend**: picogk
- **voxel_size_mm**: 0.3
- **bounds_mm**: [[-30.0, -20.0012, -0.0], [29.9977, 20.0012, 44.0]]
- **volume_mm3**: 30863.769
- **surface_area_mm2**: 10360.71
- **surface_vertices**: 122044
- **surface_faces**: 244092
- **watertight**: True
- **boundary_tags**: ['base', 'upright', 'boss', 'bolt_hole', 'shaft_bore']
