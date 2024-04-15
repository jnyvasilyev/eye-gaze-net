import numpy as np
import cv2

def get_head_pose(face_landmarks, image_shape):
    img_h, img_w, img_c = image_shape
    face_2d, face_3d = [], []
    for idx, lm in enumerate(face_landmarks.landmark):
        if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
            x,y = int(lm.x * img_w),int(lm.y * img_h)
            face_2d.append([x,y])
            face_3d.append(([x,y,lm.z]))
    face_2d = np.array(face_2d,dtype=np.float64)
    face_3d = np.array(face_3d,dtype=np.float64)

    #Calibrate camera to adjust the following constants
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length,0,img_h/2],
                            [0,focal_length,img_w/2],
                            [0,0,1]])
    distortion_matrix = np.zeros((4,1),dtype=np.float64)

    #get rotation of face
    success,rotation_vec,translation_vec = cv2.solvePnP(face_3d,face_2d,cam_matrix,distortion_matrix)
    rmat,jac = cv2.Rodrigues(rotation_vec)

    angles,mtxR,mtxQ,Qx,Qy,Qz = cv2.RQDecomp3x3(rmat)

    pitch = angles[0] * 360
    yaw = angles[1] * 360
    roll = angles[2] * 360

    return pitch, yaw
