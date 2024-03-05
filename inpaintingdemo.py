import os
import cv2
import mediapipe as mp
import pyvirtualcam
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from model import ECCNet
from warp import WarpImageWithFlowAndBrightness
from utils.vcam_utils import add_outline, get_eye_patch

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # depends on input device, usually 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

look_vec = np.array([[0.0, 0.0]])
look_vec = np.tile(look_vec[:, :, np.newaxis, np.newaxis], (1, 1, 32, 64))
look_vec = torch.tensor(look_vec).float().to(device)

OUTPUT_DIR = "./output"


def create_virtual_cam():
    with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True
    ) as face_mesh:
        _, frame1 = cap.read()
        with pyvirtualcam.Camera(
            width=frame1.shape[1], height=frame1.shape[0], fps=20
        ) as cam:
            # Initialize ECCNet
            model = ECCNet().to(device)
            ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoints/ckpt_{15}.pt")
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            warp = WarpImageWithFlowAndBrightness(torch.zeros((1, 3, 32, 64)))

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Error reading camera frame")
                    break

                # To improve performance, optionally mark the image as not writeable
                # to pass by reference
                image.flags.writeable = False
                results = face_mesh.process(image)

                # Draw the face mesh annotations on the image
                image.flags.writeable = True
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        face = face_landmarks.landmark
                        # draw(face, image)

                        # Apply ECCNet to image
                        with torch.no_grad():
                            # Get eye image patch
                            og_eye_patch, og_size, cut_coord = get_eye_patch(
                                face, image
                            )
                            eye_patch = torch.tensor(og_eye_patch).float().to(device)
                            eye_patch = eye_patch.permute(2, 0, 1).unsqueeze(
                                0
                            )  # H, W, C -> N, C, H, W

                            # Input into model
                            flow_corr, bright_corr = model(eye_patch, look_vec)
                            eye_corr = warp(eye_patch, flow_corr, bright_corr)
                            eye_corr = eye_corr / 255.0
                            eye_corr = eye_corr.squeeze().permute(1, 2, 0).cpu().numpy()

                            # Paste eye back
                            eye_corr = cv2.resize(eye_corr, (og_size[1], og_size[0]))
                            # print("eyecorr")
                            # print(eye_corr.shape)
                            # print(
                            #     image[
                            #         cut_coord[0] : cut_coord[0] + og_size[0],
                            #         cut_coord[1] : cut_coord[1] + og_size[1],
                            #     ].shape
                            # )
                            image[
                                cut_coord[0] : cut_coord[0] + og_size[0],
                                cut_coord[1] : cut_coord[1] + og_size[1],
                            ] = (
                                eye_corr * 255.0
                            )

                        # image = add_outline(face, image)

                        # mp_drawing.draw_landmarks(
                        #     image=image,
                        #     landmark_list=face_landmarks,
                        #     connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                        #     landmark_drawing_spec=None,
                        #     connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
                        # )

                        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        cam.send(image)
                        cam.sleep_until_next_frame()
                        cv2.imshow("Eye", cv2.flip(eye_corr, 1))

                cv2.imshow("Face", cv2.flip(image, 1))  # selfie flip
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if cv2.getWindowProperty("Face", cv2.WND_PROP_VISIBLE) < 1:
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    create_virtual_cam()
