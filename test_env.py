import gymnasium as gym
import ale_py

print("ALE-py version:", ale_py.__version__)
print("Gymnasium version:", gym.__version__)

try:
    env = gym.make("ALE/MontezumaRevenge-v5")
    obs, info = env.reset()
    print("Successfully created ALE/MontezumaRevenge-v5!")
    print("Obs shape:", obs.shape)
    print("Action space:", env.action_space)
    env.close()
except Exception as e:
    print("Error creating ALE/MontezumaRevenge-v5:", e)

try:
    env2 = gym.make("MontezumaRevengeNoFrameskip-v4")
    obs2, info2 = env2.reset()
    print("Successfully created MontezumaRevengeNoFrameskip-v4!")
    print("Obs shape:", obs2.shape)
    print("Action space:", env2.action_space)
    env2.close()
except Exception as e:
    print("Error creating MontezumaRevengeNoFrameskip-v4:", e)
