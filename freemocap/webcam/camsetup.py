import cv2
import imutils
import platform
import time

try:
    import mediapipe as mp
except ImportError:
    mp = None

import numpy as np


if mp is not None:
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
else:
    mp_drawing = None
    mp_drawing_styles = None
    mp_holistic = None


def _open_capture(cam_id):
    if platform.system() == 'Windows':
        return cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if platform.system() == 'Darwin':
        return cv2.VideoCapture(cam_id, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(cam_id, cv2.CAP_ANY)


def _configure_capture(cap, cam_id, parameterDictionary, cam_exposure):
    resWidth = parameterDictionary.get("resWidth")
    resHeight = parameterDictionary.get("resHeight")
    framerate = parameterDictionary.get("framerate")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resHeight)
    if framerate is not None:
        cap.set(cv2.CAP_PROP_FPS, framerate)

    # These settings are useful for some Windows/DirectShow webcams, but on macOS
    # AVFoundation does not reliably support them through OpenCV and they can
    # cause empty frames or camera-open instability.
    if platform.system() != 'Darwin':
        cap.set(cv2.CAP_PROP_EXPOSURE, cam_exposure)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    print("__________________________________________")
    print("cv2::videocapture properties for Camera# {}".format(cam_id))
    print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("CAP_PROP_FPS : '{}'".format(cap.get(cv2.CAP_PROP_FPS)))
    print("CAP_PROP_EXPOSURE : '{}'".format(cap.get(cv2.CAP_PROP_EXPOSURE)))
    print("CAP_PROP_POS_MSEC : '{}'".format(cap.get(cv2.CAP_PROP_POS_MSEC)))
    print("CAP_PROP_FRAME_COUNT  : '{}'".format(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print("CAP_PROP_BRIGHTNESS : '{}'".format(cap.get(cv2.CAP_PROP_BRIGHTNESS)))
    print("CAP_PROP_CONTRAST : '{}'".format(cap.get(cv2.CAP_PROP_CONTRAST)))
    print("CAP_PROP_SATURATION : '{}'".format(cap.get(cv2.CAP_PROP_SATURATION)))
    print("CAP_PROP_HUE : '{}'".format(cap.get(cv2.CAP_PROP_HUE)))
    print("CAP_PROP_GAIN  : '{}'".format(cap.get(cv2.CAP_PROP_GAIN)))
    print("CAP_PROP_CONVERT_RGB : '{}'".format(cap.get(cv2.CAP_PROP_CONVERT_RGB)))
    print("__________________________________________")


def _draw_mediapipe_overlay(frame, holistic):
    results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mp_drawing.draw_landmarks(
        frame,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
    )
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )
    mp_drawing.draw_landmarks(
        frame,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
    )
    mp_drawing.draw_landmarks(
        frame,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
    )
    return frame


def RunSetup(cam_inputs, rotation_input, exposure_input, paramDict, mediaPipeOverlay):
    """
    Preview all cameras during setup.

    macOS compatibility note: all OpenCV HighGUI calls are kept on the main
    thread. The old implementation launched one thread per camera and each
    thread called cv2.namedWindow/imshow/waitKey, which can crash on macOS.
    """
    if not cam_inputs:
        raise ValueError("Camera input list (cam_inputs) is empty")

    captures = []
    window_names = []
    holistic = None

    try:
        for cam_input, cam_rotation, cam_exposure in zip(cam_inputs, rotation_input, exposure_input):
            cap = _open_capture(cam_input)
            _configure_capture(cap, cam_input, paramDict, cam_exposure)
            captures.append((cam_input, cap, cam_rotation))

            window_name = "Camera" + str(cam_input) + " Preview - Press ESC to exit Setup"
            cv2.namedWindow(window_name)
            window_names.append(window_name)

        if mediaPipeOverlay and mp_holistic is None:
            print("MediaPipe overlay requested, but mediapipe is not installed. Continuing without overlay.")
            mediaPipeOverlay = False

        if mediaPipeOverlay:
            holistic = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=0,
                enable_segmentation=True,
            )

        timestamps = []
        while True:
            any_success = False
            timestamps.append(time.time())

            for (cam_input, cap, cam_rotation), window_name in zip(captures, window_names):
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                any_success = True

                if mediaPipeOverlay and holistic is not None:
                    try:
                        frame = _draw_mediapipe_overlay(frame, holistic)
                    except Exception as e:
                        print(e)

                if cam_rotation is not None:
                    frame = imutils.rotate_bound(frame, angle=cam_rotation)

                cv2.imshow(window_name, frame)

            if not any_success:
                break

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        if holistic is not None:
            holistic.close()
        for _, cap, _ in captures:
            cap.release()
        cv2.destroyAllWindows()


# def detect_charuco_board(image,  annotate_image=True):
#     """
#     Charuco base pose estimation code intentionally left commented out from the original file.
#     """
