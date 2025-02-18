from pxr import Usd, UsdGeom, UsdPhysics

def change_root_link(usd_file_path, new_root_link_name, output_file_path):
    """
    Reconfigures a USD file to set a new root link inside /hero_vehicle.

    Args:
        usd_file_path (str): Path to the input USD file.
        new_root_link_name (str): Name of the new root link (e.g., "chassis").
        output_file_path (str): Path to save the modified USD file.
    """
    stage = Usd.Stage.Open(usd_file_path)

    # Get the main vehicle prim
    hero_vehicle = stage.GetPrimAtPath("/hero_vehicle")
    if not hero_vehicle or not hero_vehicle.IsValid():
        raise ValueError("Could not find '/hero_vehicle' in the USD file.")

    # Ensure the new root exists
    new_root_prim = stage.GetPrimAtPath(f"/hero_vehicle/{new_root_link_name}")
    if not new_root_prim or not new_root_prim.IsValid():
        raise ValueError(f"New root link '{new_root_link_name}' not found under '/hero_vehicle'.")

    # Get the current root link (first child under hero_vehicle)
    children = hero_vehicle.GetChildren()
    if not children:
        raise RuntimeError("No root link found under /hero_vehicle.")
    
    current_root = children[0]  # Get the first child as the current root

    # Ensure new root is different
    if current_root.GetName() == new_root_link_name:
        print(f"'{new_root_link_name}' is already the root. No changes needed.")
        return

    print(f"Changing root from '{current_root.GetName()}' to '{new_root_link_name}'...")

    # Move all children of hero_vehicle under new_root_prim
    for child in children:
        if child != new_root_prim:
            new_child_path = f"{new_root_prim.GetPath()}/{child.GetName()}"
            stage.DefinePrim(new_child_path, child.GetTypeName())  # Define a new prim under new_root
            new_child_prim = stage.GetPrimAtPath(new_child_path)
            
            # Copy all attributes from old to new
            for attr in child.GetAttributes():
                new_attr = new_child_prim.CreateAttribute(attr.GetName(), attr.GetTypeName())
                new_attr.Set(attr.Get())

    # Ensure new root is directly under hero_vehicle
    new_root_prim_path = f"/hero_vehicle/{new_root_link_name}"
    if new_root_prim.GetPath() != new_root_prim_path:
        stage.DefinePrim(new_root_prim_path, new_root_prim.GetTypeName())

    # Reset transforms (prevent unwanted shifts)
    UsdGeom.Xformable(new_root_prim).ClearXformOpOrder()

    # Update physics joints (if any joint references the old root, update it)
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            body0_rel = joint.GetPrim().GetRelationship("physics:body0")
            if body0_rel and body0_rel.HasAuthoredTargets():
                targets = body0_rel.GetTargets()
                if targets and targets[0] == current_root.GetPath():
                    body0_rel.SetTargets([new_root_prim.GetPath()])

    # Save the modified USD file
    stage.Flatten()
    for prim in stage.Traverse():
        print(prim.GetPath())
    stage.GetRootLayer().Export(output_file_path)
    print(f"Modified USD saved to: {output_file_path}")

# Example usage:
change_root_link("hero_vehicle.usd", "leg1link4", "hero_vehicle_modified.usd")
