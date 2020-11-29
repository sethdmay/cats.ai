import os
import numpy as np
from PIL import Image
from openpose.process_images import get_keypoints

num_angles = 8
num_vectors = 12
directory_in_str = './openpose/img_to_process'

# TODO Use Numpy for this!

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

#just to make sure I don't mess up the above function, I will make a duplicate version here
#groupOpenPoseArrays convverted to numpy version


	


	
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
    #shot_successes = inputShotSuccess()
    print(shot_successes_array)
    successful_angles, miss_angles = groupAngleArrays(angles, shot_successes_array)
    return averageAngles(successful_angles), averageAngles(miss_angles)


#print(inputShotSuccess())
'''a = [[1,2,3],[4,5,6]]
b = [[7,8,9],[10,11,12]]
d = [[13,14,15],[16,17,18]]
c = [a, b, d]

successes = [True, True, False]

successful_array, miss_array = groupOpenPoseArrays(c, successes)
print(successful_array, miss_array)
print(successful_array == c)
a = [0] * 5
b = [1,2,3,4,5]
a += b

print(a) '''



ex_input = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2, 1]])

def average_angles(angles_array):
	running_total = np.zeros(9)
	curr_part = 0
	for sub_list in angles_array:
		for e in sub_list:
			running_total[curr_part] += e
			curr_part += 1
		curr_part = 0

	counter = 0
	for e in running_total:
		running_total[counter] = e/(len(angles_array))
		counter+=1
	return running_total
#print(np.zeros(8))
print(average_angles(ex_input))




print(a)

'''

test1 = np.array([[[6.29239807e+02, 5.98466919e+02, 9.56953824e-01],
  [6.04586975e+02, 6.66049622e+02, 7.69139051e-01],
  [6.04623962e+02, 6.69112122e+02, 8.71934175e-01],
  [6.99743286e+02, 6.72054260e+02, 8.93443167e-01],
  [6.99838318e+02, 6.01571228e+02, 8.86620343e-01],
  [6.04589600e+02, 6.59911499e+02, 6.43684864e-01],
  [6.99651367e+02, 6.72076294e+02, 3.68231565e-01],
  [6.99725098e+02, 6.10784851e+02, 3.93413872e-01],
  [5.70944214e+02, 8.62202942e+02, 7.07278371e-01],
  [5.77091797e+02, 8.65277649e+02, 7.37012982e-01],
  [6.16945312e+02, 1.01562201e+03, 8.89406562e-01],
  [5.61683899e+02, 1.12003625e+03, 1.73740849e-01],
  [5.58713684e+02, 8.59152039e+02, 5.61030149e-01],
  [5.92374695e+02, 1.00947424e+03, 8.39329600e-01],
  [5.37177185e+02, 1.11999854e+03, 5.32564163e-01],
  [6.16826843e+02, 5.92390747e+02, 9.16968405e-01],
  [6.26184082e+02, 5.86320312e+02, 1.52191162e-01],
  [5.83258606e+02, 6.13854553e+02, 9.06941116e-01],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [5.28015564e+02, 1.12001575e+03, 2.93556333e-01],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]])

kps, success_array = get_keypoints(directory_in_str)
print(kps)

print(analyzeOpenPoses(kps, success_array))
