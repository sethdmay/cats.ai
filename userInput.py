import os
import numpy as np
from PIL import Image
from openpose.process_images import get_keypoints

num_angles = 8
num_vectors = 12
directory_in_str = './openpose/img_to_process'


# Given 2D positions of various body parts from an OpenPose array for a single image,
# calculates angles of shoulders, elbows, knees, hips. Uses LShoulder, Neck, RShoulder, LElbow, RElbow, LWrist, RWrist,
# LAnkle, RAnkle, LKnee, RKnee, LHip, MidHip, RHip to make these calculations
# Outputs numpy array of these angles [rightShoulder, rightElbow, rightHip, rightKnee,leftShoulder, leftElbow, leftHip,
# leftKnee]
def getAngles(open_pose_array):
    angles = np.zeros(num_angles) # [rightShoulder, rightElbow, rightHip, rightKnee,
    # leftShoulder, leftElbow, leftHip, leftKnee]
    vectors = np.zeros((num_vectors, 2)) # [rightShoulderBlade, rightUpperArm, rightForearm, rightPelvis, rightUpperLeg,
    # rightLowerLeg, leftShoulderBlade, leftUpperArm, leftForearm, leftPelvis, leftUpperLeg, leftLowerLeg]

    # Vector calculations
    vectors[0] = getVector(open_pose_array[1], open_pose_array[2])  # right shoulder blade
    vectors[1] = getVector(open_pose_array[2], open_pose_array[3])  # right upper arm
    vectors[2] = getVector(open_pose_array[3], open_pose_array[4])  # right forearm
    vectors[3] = getVector(open_pose_array[8], open_pose_array[9])  # right pelvis
    vectors[4] = getVector(open_pose_array[9], open_pose_array[10])  # right upper le g
    vectors[5] = getVector(open_pose_array[10], open_pose_array[11])  # right lower leg

    vectors[6] = getVector(open_pose_array[1], open_pose_array[5])  # left shoulder blade
    vectors[7] = getVector(open_pose_array[5], open_pose_array[6])  # left upper arm
    vectors[8] = getVector(open_pose_array[6], open_pose_array[7]) # left forearm
    vectors[9] = getVector(open_pose_array[8], open_pose_array[12])  # left pelvis
    vectors[10] = getVector(open_pose_array[12], open_pose_array[13])  # left upper leg
    vectors[11] = getVector(open_pose_array[13], open_pose_array[14])  # left lower leg

    # Angle calculations
    angles[0] = getAngle(vectors[0], vectors[1]) # right shoulder
    angles[1] = getAngle(vectors[1], vectors[2])  # right elbow
    angles[2] = getAngle(vectors[3], vectors[4])  # right hip
    angles[3] = getAngle(vectors[4], vectors[5])  # right knee

    angles[4] = getAngle(vectors[6], vectors[7])  # left shoulder
    angles[5] = getAngle(vectors[7], vectors[8])  # left elbow
    angles[6] = getAngle(vectors[9], vectors[10])  # left hip
    angles[7] = getAngle(vectors[10], vectors[11])  # left knee

    return angles


# Takes in 2 positions, returns numpy array representing vector of pos2 - pos1
def getVector(pos1, pos2):
    return np.array([pos2[0] - pos1[0], pos2[1] - pos1[1]])


# Takes in 2 vectors, returns angle (in radians) between the vectors calculated using dot product
def getAngle(vec1, vec2):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.dot(vec1, vec2)
    return np.arccos(dot_product)


# Given list of list of angles from getAngles, returns a list of the angles averaged by index. Essentially, we take
# averages down the columns.
def averageAngles(angles_list):
    if len(angles_list) == 0:
        return np.array([])

    averages = np.sum(angles_list, axis=0)
    return averages / len(angles_list)


def groupAngleArrays(angle_arrays, shot_successes):
    successful_array = np.array([])
    miss_array = np.array([])
    for it in range(len(angle_arrays)):
        if shot_successes[it]:
            successful_array = np.append(successful_array, angle_arrays[it])
        else:
            miss_array = np.append(miss_array, angle_arrays[it])

    return np.reshape(successful_array, (-1, num_angles)), np.reshape(miss_array, (-1, num_angles))


def processAllOpenPoses(open_pose_array):
    angles = np.zeros((len(open_pose_array), num_angles))

    for i in range(len(open_pose_array)):
        angles[i] = getAngles(open_pose_array[i])

    return angles


def inputShotSuccess():
    directory = os.fsencode(directory_in_str)
    shot_successes = np.array([])

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        file_path = os.path.join(directory_in_str, filename)
        if file_path.lower().endswith((".png",".jpeg",".jpg")):
            im = Image.open(file_path)
            im.show()
            successful_shot = input()
            if successful_shot == 'y' or successful_shot == "Y":
                shot_successes = np.append(shot_successes, True)
            else:
                shot_successes = np.append(shot_successes, False)
        else:
            continue

    return shot_successes


# Takes in OpenPose positions, reads user input for each shot
# Returns average angle of each joint for successful shots and average angle of each joint for missed shots
def analyzeOpenPoses(open_pose_array, shot_successes_array):
    angles = processAllOpenPoses(open_pose_array)
    successful_angles, miss_angles = groupAngleArrays(angles, shot_successes_array)
    outputAnalysis(averageAngles(successful_angles), averageAngles(miss_angles))


def outputAnalysis(successful_angles, miss_angles):
    joints = ["right shoulder", "right elbow", "right hip", "right knee",
              "left shoulder", "left elbow", "left hip", "left knee"]

    i = 0
    length = min(len(successful_angles), len(miss_angles))

    while i < length:
        if successful_angles[i] < miss_angles[i]:
            print("Bend your ", joints[i], " more")
        else:
            print("Bend your ", joints[i], " less")
        i += 1


kps, success_array = get_keypoints(directory_in_str)
print(kps)

analyzeOpenPoses(kps, success_array)
