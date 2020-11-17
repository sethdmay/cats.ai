import os
from PIL import Image

# TODO Use Numpy for this!

# Given 2D positions of various body parts from an OpenPose array for a single image,
# calculates angles of shoulders, elbows, knees, hips. Uses LShoulder, Neck, RShoulder, LElbow, RElbow, LWrist, RWrist,
# LAnkle, RAnkle, LKnee, RKnee, LHip, MidHip, RHip to make these calculations
# Outputs list of these angles [leftShoulder, rightShoulder, leftElbow, rightElbow, leftKnee, rightKnee, leftHip,
# rightHip]
def getAngles(open_pose_array):
    angles = []
    return angles

# Given list of list of angles from getAngles, returns a list of the angles averaged by index. Essentially, we take
# averages down the columns.
def averageAngles(angles_list):
    if len(angles_list) == 0:
        return []

    averages = [0] * len(angles_list[0])

    return averages

def groupOpenPoseArrays(open_pose_arrays, shot_successes):
    successful_array = []
    miss_array = []

    for it in range(len(open_pose_arrays)):
        if shot_successes[it]:
            successful_array.append(open_pose_arrays[it])
        else:
            miss_array.append(open_pose_arrays[it])

    return successful_array, miss_array

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

#print(inputShotSuccess())
'''a = [[1,2,3],[4,5,6]]
b = [[7,8,9],[10,11,12]]
d = [[13,14,15],[16,17,18]]
c = [a, b, d]

successes = [True, True, False]

successful_array, miss_array = groupOpenPoseArrays(c, successes)
print(successful_array, miss_array)
print(successful_array == c)'''
a = [0] * 5
b = [1,2,3,4,5]
a += b
print(a)