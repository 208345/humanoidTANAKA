import pybullet as p
import pybullet_data
import time

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("model/humanoid.urdf", [0, 0, 0.5], useFixedBase=False)

print("Simulating 100 steps...")
for i in range(100):
    p.stepSimulation()
    time.sleep(1./240.)

pos, orn = p.getBasePositionAndOrientation(robotId)
print(f"Final Z position: {pos[2]}")
p.disconnect()
