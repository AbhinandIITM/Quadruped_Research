import mujoco

# Load your existing PyBullet URDF
model = mujoco.MjModel.from_xml_path(r"unitree_go2\urdf\go2_description_bullet_cleaned.urdf")

# Export and save it into a MuJoCo MJCF XML file
mujoco.mj_saveLastXML(r"unitree_go2\go2_mujoco_c.xml", model)
print("Conversion complete!")
