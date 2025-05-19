import torch
from pxr import Usd, UsdPhysics, Sdf
import omni.usd
import omni.kit.commands
import logging

log = logging.getLogger("grasp_manager")


def create_fixed_joint(prim_path, parent_path, child_path, break_force=1e6, break_torque=1e6):
    stage = omni.usd.get_context().get_stage()
    joint_prim = stage.DefinePrim(prim_path, "PhysicsFixedJoint")
    joint_prim.GetReferences().ClearReferences()

    UsdPhysics.Joint(joint_prim).CreateBody0Rel().SetTargets([parent_path])
    UsdPhysics.Joint(joint_prim).CreateBody1Rel().SetTargets([child_path])

    # Optional: set limits
    if break_force is not None:
        UsdPhysics.Joint(joint_prim).CreateBreakForceAttr().Set(break_force)
    if break_torque is not None:
        UsdPhysics.Joint(joint_prim).CreateBreakTorqueAttr().Set(break_torque)


def delete_prim(prim_path):
    omni.kit.commands.execute("DeletePrims", paths=[prim_path])


def does_prim_exist(prim_path):
    stage = omni.usd.get_context().get_stage()
    return stage.GetPrimAtPath(prim_path).IsValid()


class SimpleGraspManager:
    def __init__(self, env, gripper_prim_paths, object_prim_paths):
        self.env = env
        self.num_envs = env.num_envs
        self.gripper_prim_paths = gripper_prim_paths
        self.object_prim_paths = object_prim_paths
        self.joint_names = [f"/World/envs/env_{i}/grasp_fixed_joint" for i in range(self.num_envs)]
        self.active_mask = torch.zeros((self.num_envs,), dtype=torch.bool, device="cuda")

    def attach(self, env_ids):
        for i in env_ids:
            env_index = int(i)
            joint_path = self.joint_names[env_index]
            if does_prim_exist(joint_path):
                continue
            create_fixed_joint(
                prim_path=joint_path,
                parent_path=self.gripper_prim_paths[env_index],
                child_path=self.object_prim_paths[env_index],
                break_force=1e6,
                break_torque=1e6,
            )
            log.info(f"[GraspManager] ✅ ATTACH fixed joint in env {env_index}")
            self.active_mask[env_index] = True

    def detach(self, env_ids):
        for i in env_ids:
            env_index = int(i)
            joint_path = self.joint_names[env_index]
            if does_prim_exist(joint_path):
                delete_prim(joint_path)
                log.info(f"[GraspManager] ❌ DETACH fixed joint in env {env_index}")
            self.active_mask[env_index] = False

    def reset(self):
        self.detach(torch.nonzero(self.active_mask).flatten())
