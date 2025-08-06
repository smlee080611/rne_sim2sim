import torch
import os
# 모델 파일의 실제 경로를 지정해야 합니다.
# 이 스크립트를 rl_player.py와 같은 위치에서 실행한다면 아래 경로는 틀릴 수 있습니다.
# 가장 간단한 방법은 .pt 파일의 절대 경로를 쓰는 것입니다.
model_path = '/home/simon/ros2_ws/src/rne_sim2sim/models/trot_v0.pt'

print(f"Loading model from: {model_path}")

try:
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    if isinstance(checkpoint, dict):
        print("\nFile is a dictionary. Here are the keys:")
        for key in checkpoint.keys():
            print(f"- {key}")
    else:
        print(f"\nFile is not a dictionary. It is of type: {type(checkpoint)}")

except Exception as e:
    print(f"\nAn error occurred: {e}")