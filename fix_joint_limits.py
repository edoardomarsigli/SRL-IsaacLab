from pxr import Usd, UsdPhysics, PhysxSchema

# CAMBIA il path se serve
usd_path = "/home/edomrl/SRL-IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/moonshot/descriptions/usd/robot/Hero_dragon_grip/hero_dragon_gripW.usd"

stage = Usd.Stage.Open(usd_path)
joint_path = "/World/hero_dragon/leg2joint7"
joint_prim = stage.GetPrimAtPath(joint_path)

if not joint_prim.IsValid():
    print("❌ Joint prim not found:", joint_path)
else:
    print("✅ Trovato il giunto:", joint_path)

    joint = UsdPhysics.RevoluteJoint(joint_prim)
    joint.CreateLowerLimitAttr(-3.1)
    joint.CreateUpperLimitAttr(3.1)

    # Applica PhysX API per garantire che i limiti siano rispettati
    physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
    physx_joint.CreateMaxJointVelocityAttr().Set(5.0)

    print("✅ Limiti impostati correttamente.")

    # Salva il file
    stage.GetRootLayer().Save()
    print("✅ File salvato:", usd_path)
