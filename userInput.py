import os
import numpy as np
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

#convert to numpy arrays
def groupOpenPoseArrays(open_pose_arrays, shot_successes):
    successful_array = []
    miss_array = []

    for it in range(len(open_pose_arrays)):
        if shot_successes[it]:
            successful_array.append(open_pose_arrays[it])
        else:
            miss_array.append(open_pose_arrays[it])

    return successful_array, miss_array

#just to make sure I don't mess up the above function, I will make a duplicate version here
#groupOpenPoseArrays convverted to numpy version

def groupOpenPoseArrays2(open_pose_arrays, shot_successes):
	


	
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


#originally start with a numpy array
 
#make an empty numpy array
#make an empty list
#np.array([])
#append to that later 


# Outputs list of these angles [leftShoulder, rightShoulder, leftElbow, rightElbow, leftKnee, rightKnee, leftHip,
# rightHip]
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



#report 

#3 functions
#write a function that converts lists to numpy arrays
#converting two points to a vector #david done
#angle between two vectors 
#done-> average angles function -> return list of average angles (2d array -> going down one column -> all of the angles for one body part)