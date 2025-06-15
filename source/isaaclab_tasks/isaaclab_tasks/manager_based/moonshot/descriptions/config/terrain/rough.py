# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from .terrain_generator_cfg import TerrainGeneratorCfg

"""Rough terrains configuration."""

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(6, 6),
    border_width=20.0,
    num_rows = 64,
    num_cols = 64,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    curriculum=False,
    difficulty_range=(0.0,0.0),
    sub_terrains={
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.001, 0.03), noise_step=0.01, border_width=0.25
        ),
        "moonlike_noise": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.002, 0.02),   # valori piccoli per sabbia fine
            noise_step=0.005,            # più piccolo = più liscio
            border_width=0.0,
        )
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=0.1, border_width=0.25
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=0.1, border_width=0.25
        # ),
    },
)


# MOONLIKE_TERRAIN_CFG = TerrainGeneratorCfg(
#     size=(10.0, 10.0),
#     border_width=20.0,
#     num_rows=10,
#     num_cols=20,
#     horizontal_scale=0.1,      # quanto si estende il pattern
#     vertical_scale=0.01,       # quanto è "morbido" o ondulato
#     slope_threshold=0.75,
#     use_cache=False,
#     curriculum=False,
#     difficulty_range=(0.0, 0.0),  # disattiva la variazione
#     sub_terrains={
#         "perlin_smooth": terrain_gen.HfPerlinTerrainCfg(
#             proportion=1.0,
#             frequency=0.2,           # più basso = ondulazioni più larghe
#             octaves=3,
#             border_width=0.25,
#         )
#     },
# )

"""Rough and hilly terrains configuration."""

ROUGH_HILLY_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=100,
    num_cols=100,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    difficulty_range=(0.0,1.0),
    sub_terrains={
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.4, noise_range=(0.001, 0.03), noise_step=0.01, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.3, slope_range=(0.1, 0.4), platform_width=0.1, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.3, slope_range=(0.1, 0.4), platform_width=0.1, border_width=0.25
        ),
    },
)
