from transformers import pipeline

model_path = "Bbyaz/robotics-deep-rl-grasping"

pipe = pipeline(model=model_path)

print(pipe.model.config)