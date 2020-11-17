import os
from PIL import Image

def inputShotSuccess():
    directory_in_str = 'input'
    directory = os.fsencode(directory_in_str)
    shot_successes = []

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        file_path = os.path.join(directory_in_str, filename)
        if file_path.endswith(".png"):
            im = Image.open(file_path)
            im.show()
            successful_shot = input()
            if successful_shot == 'y' or successful_shot == "Y":
                shot_successes.append(True)
            else:
                shot_successes.append(False)
        else:
            continue

    return shot_successes

print(inputShotSuccess())