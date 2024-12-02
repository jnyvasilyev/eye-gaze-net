This repo is for the AI model to redirect eye gaze. To integrate with the web conferencing application, clone the following repo:

https://github.com/yamakov03/conference-webrtc/tree/viewerFocus

Here's our senior design showcase [video](https://youtu.be/LL7fj7i5Vsk?t=68)


model.py - LiteGaze model architecture

Train_Loop.py - Trains the model bidirectionally

warp.py - Uses output vector field from model and warps image

inpaintingdemo.py - Demo for live presentation

utils - Data loading

angle_to_vec.py - Converts gaze direction to vectors

