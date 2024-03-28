import os
import numpy as np

degrees_to_radians = np.pi / 180

def pitchyaw_to_vector(angles, is_degrees=True):
    n = angles.shape[0]
    out = np.empty((n, 3))
    if is_degrees:
        angles = angles * degrees_to_radians
    out[:, 0] = np.sin(angles[:, 1])  # Yaw
    out[:, 1] = -np.sin(angles[:, 0])  # Pitch, assuming pitch is up/down and should be applied to Y
    out[:, 2] = np.cos(angles[:, 1]) * np.cos(angles[:, 0])  # Adjusted for combined pitch/yaw influence on Z
    out = np.apply_along_axis(lambda vec: vec / np.linalg.norm(vec), 1, out)
    return out

def process_gaze_dataset(dataset_path):
    data_dict = {}

    # Iterate through each subject's folder
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            # Iterate through each image in the subject's folder
            for image_filename in os.listdir(folder_path):
                parts = image_filename.split('_')
                if len(parts) >= 5:
                    subject_id, _, head_pose, vertical_disp, horizontal_disp = parts
                    # Convert displacements to integer
                    horizontal_disp = int(horizontal_disp.split('.')[0][:-1])
                    vertical_disp = int(vertical_disp[:-1])  # Remove the 'V' and convert to int
                    unique_id = f"{subject_id}_{head_pose}"

                    # Prepare angles array for vector conversion
                    angles_array = np.array([[vertical_disp, horizontal_disp]])
                    vector = pitchyaw_to_vector(angles_array)[0]  # Convert angles to vector

                    # Initialize the list for this unique ID if it doesn't already exist
                    if unique_id not in data_dict:
                        data_dict[unique_id] = []

                    # Append image path and vector to the list for this unique ID
                    data_dict[unique_id].append({
                        'image_path': os.path.join(folder_path, image_filename),
                        'vector': vector
                    })

    return data_dict

dataset_path = "dataset\Columbia Gaze Data Set"
data_dict = process_gaze_dataset(dataset_path)
#print(data_dict['0001_15P'])
'''
Example dict output of data_dict['0056_30P']:

where 0056_30P (subject 56 with headpose 30) is the ID for a series of images

'0056:30P': [{'image_path': 'path\\to\image.jpg', 'vector': array([-0.0, 0.0, 1])}, 
            {'image_path': 'path\\to\image.jpg', 'vector': array([-0.017741, 0.890741948, 1])}, etc.. ]


''' 