import cv2
import numpy as np
import mediapipe as mp

# Face mesh indices:
# https://github.com/google/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
right_eye_indices = [
    33,
    246,
    161,
    160,
    159,
    158,
    157,
    173,
    133,
    155,
    154,
    153,
    145,
    144,
    163,
    7,
]
left_eye_indices = [
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
]


def get_coords(point, im):
    x = int(im.shape[1] * point.x)
    y = int(im.shape[0] * point.y)
    return (x, y)


def add_eye(minX, maxX, minY, maxY, eye, im):
    eye = cv2.resize(eye, (maxX - minX, maxY - minY))
    alpha = eye[:, :, 3] / 255.0
    alpha_inv = 1 - alpha
    for c in range(0, 3):
        im[minY:maxY, minX:maxX, c] = (
            alpha * eye[:, :, c] + alpha_inv * im[minY:maxY, minX:maxX, c]
        )


def add_outline(face, im):
    img_height, img_width = im.shape[:2]
    left_eye_landmarks = [face[i] for i in left_eye_indices]
    right_eye_landmarks = [face[i] for i in right_eye_indices]
    left_eye_landmarks = [
        (int(landmark.x * img_width), int(landmark.y * img_height))
        for landmark in left_eye_landmarks
    ]
    right_eye_landmarks = [
        (int(landmark.x * img_width), int(landmark.y * img_height))
        for landmark in right_eye_landmarks
    ]
    left_eye_np = np.array(left_eye_landmarks)
    right_eye_np = np.array(right_eye_landmarks)

    eye_fill_mask = np.zeros_like(im, dtype=np.uint8)
    eye_outline_mask = np.zeros_like(im, dtype=np.uint8)
    cv2.fillPoly(eye_fill_mask, [left_eye_np], (0, 255, 0))
    cv2.fillPoly(eye_fill_mask, [right_eye_np], (0, 255, 0))
    cv2.polylines(eye_outline_mask, [left_eye_np], True, (0, 0, 255), 2)
    cv2.polylines(eye_outline_mask, [right_eye_np], True, (0, 0, 255), 2)

    alpha = 0.4
    fill_mask = eye_fill_mask.astype(bool)
    outline_mask = eye_outline_mask.astype(bool)
    im[fill_mask] = cv2.addWeighted(im, alpha, eye_fill_mask, 1 - alpha, 0)[fill_mask]
    im[outline_mask] = eye_outline_mask[outline_mask]

    return im


def get_eye_patch(face, im, left=True):
    """
    Return a cropped 64x32 image patch around the left eye
    """
    img_height, img_width = im.shape[:2]
    if left:
        eye_landmarks = [face[i] for i in left_eye_indices]
        eye_landmarks = np.array(
            [
                (int(landmark.x * img_width), int(landmark.y * img_height))
                for landmark in eye_landmarks
            ]
        )
    else:
        eye_landmarks = [face[i] for i in right_eye_indices]
        eye_landmarks = np.array(
            [
                (int(landmark.x * img_width), int(landmark.y * img_height))
                for landmark in eye_landmarks
            ]
        )

    # Calculate bounding box
    min_x, min_y = np.min(eye_landmarks, axis=0)
    max_x, max_y = np.max(eye_landmarks, axis=0)

    # Get bbox length
    l = max_x - min_x
    aspect_ratio = 2
    width = 1.5 * l
    height = max_y - min_y

    # Calculate center of bbox
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Correct aspect ratio
    if (width) / (height) > aspect_ratio:
        # too wide. Expand height
        height = width / aspect_ratio
    else:
        # too tall. Expand width
        width = height * aspect_ratio

    # Resize to 64x32
    im_cropped = im[
        int(center_y - (height / 2)) : int(center_y + (height / 2)),
        int(center_x - (width / 2)) : int(center_x + (width / 2)),
    ]
    og_size = [im_cropped.shape[0], im_cropped.shape[1]]
    cut_coord = [int(center_y - (height / 2)), int(center_x - (width / 2))]

    im_cropped = cv2.resize(im_cropped, (64, 32))

    return im_cropped, og_size, cut_coord


def get_dist(pts):
    x1, y1 = pts[0]
    x2, y2 = pts[1]
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def draw(face, im):
    # Face mesh indices:
    # https://github.com/google/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    left_eye_outer = face[33]
    left_eye_inner = face[133]
    left_eye_upper = face[159]
    left_eye_lower = face[145]
    right_eye_outer = face[263]
    right_eye_inner = face[362]
    right_eye_upper = face[386]
    right_eye_lower = face[374]

    lox, _ = get_coords(left_eye_outer, im)
    lix, _ = get_coords(left_eye_inner, im)
    rox, _ = get_coords(right_eye_outer, im)
    rix, _ = get_coords(right_eye_inner, im)
    _, luy = get_coords(left_eye_upper, im)
    _, lly = get_coords(left_eye_lower, im)
    _, ruy = get_coords(right_eye_upper, im)
    _, rly = get_coords(right_eye_lower, im)

    eye = cv2.imread("eye.png", cv2.IMREAD_UNCHANGED)
    add_eye(min(lox, lix), max(lox, lix), min(luy, lly) - 5, max(luy, lly) + 5, eye, im)

    eye = cv2.flip(eye, 1)
    add_eye(min(rox, rix), max(rox, rix), min(ruy, rly) - 5, max(ruy, rly) + 5, eye, im)
