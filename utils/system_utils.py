#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os


def mkdir_p(folder_path):
    os.makedirs(folder_path, exist_ok=True)


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)


def check_gs_model(model_path, saving_iterations, fix_pcd=False):
    file_name = "point_cloud_fix_pcd" if fix_pcd else "point_cloud"
    p = os.path.join(model_path, file_name)
    if not os.path.exists(model_path):
        return False
    if not os.path.exists(p):
        return False
    if len(os.listdir(p)) <= 0:
        return False
    if not searchForMaxIteration(p) == saving_iterations[-1]:
        return False
    else:
        max_iter = searchForMaxIteration(p)
        print(f"Find the model: {p}, iterations: {max_iter}")
        return True
