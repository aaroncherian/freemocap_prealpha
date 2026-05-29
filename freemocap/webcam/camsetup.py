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
import threading
import cv2
import imutils
import os
import platform
import time 

import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

class VideoSetup(threading.Thread):
    """
    Capture-only worker for preview. On macOS every OpenCV HighGUI call
    (namedWindow/imshow/waitKey) must run on the main thread, so this thread
    only grabs frames into self.latest_frame; RunSetup() displays them on the
    main thread. Pressing ESC there sets stop_event, which ends this loop.
    """
    def __init__(self, camID, parameterDictionary, rotNum, cam_exposure, stop_event=None):
        threading.Thread.__init__(self)
        self.camID = camID
        self.parameterDictionary = parameterDictionary
        self.rotNum = rotNum
        self.cam_exposure = cam_exposure
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.latest_frame = None
        self.window_name = "Camera" + str(camID) + ' Preview - Press ESC to exit Setup'

    def run(self):
        self.record(self.parameterDictionary, self.rotNum, self.cam_exposure)

    def record(self, parameterDictionary, rotNum, cam_exposure):
        exposure = cam_exposure
        resWidth = parameterDictionary.get("resWidth")
        resHeight = parameterDictionary.get("resHeight")

        if platform.system() == 'Windows':
            cap = cv2.VideoCapture(self.camID, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.camID, cv2.CAP_ANY)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resWidth)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resHeight)
        if platform.system() == 'Windows':
            # AVFoundation rejects FOURCC and uses a different exposure scale;
            # only apply these on Windows.
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        print("__________________________________________")
        print("cv2::videocapture properties for Camera# {}".format(self.camID))
        print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("CAP_PROP_FPS : '{}'".format(cap.get(cv2.CAP_PROP_FPS)))
        print("__________________________________________")

        if not cap.isOpened():
            print("Camera {} failed to open".format(self.camID))
            return

        while not self.stop_event.is_set():
            ret1, frame1 = cap.read()
            if not ret1 or frame1 is None:
                time.sleep(0.01)  # macOS warmup can yield empty reads
                continue
            if rotNum is not None:
                frame1 = imutils.rotate_bound(frame1, angle=rotNum)
            self.latest_frame = frame1
        cap.release()

class MediaPipeVideoSetup(threading.Thread):
    """
    Capture + MediaPipe-overlay worker for preview (no GUI in-thread; see
    VideoSetup). Inference stays here; display happens on the main thread.
    """
    def __init__(self, camID, parameterDictionary, rotNum, cam_exposure, stop_event=None):
        threading.Thread.__init__(self)
        self.camID = camID
        self.parameterDictionary = parameterDictionary
        self.rotNum = rotNum
        self.cam_exposure = cam_exposure
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.latest_frame = None
        self.window_name = "Camera" + str(camID) + ' - Press ESC to exit'

    def run(self):
        self.record(self.parameterDictionary, self.rotNum, self.cam_exposure)

    def record(self, parameterDictionary, rotNum, cam_exposure):
        exposure = cam_exposure
        resWidth = parameterDictionary.get("resWidth")
        resHeight = parameterDictionary.get("resHeight")

        if platform.system() == 'Windows':
            cap = cv2.VideoCapture(self.camID, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.camID, cv2.CAP_ANY)

        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=True) as holistic:

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resWidth)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resHeight)
            if platform.system() == 'Windows':
                cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

            if not cap.isOpened():
                print("Camera {} failed to open".format(self.camID))
                return

            while not self.stop_event.is_set():
                ret1, frame1 = cap.read()
                if not ret1 or frame1 is None:
                    time.sleep(0.01)
                    continue
                try:
                    results = holistic.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
                    mp_drawing.draw_landmarks(
                        frame1, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        frame1, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    mp_drawing.draw_landmarks(
                        frame1, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
                    mp_drawing.draw_landmarks(
                        frame1, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
                except Exception as e:
                    print(e)
                if rotNum is not None:
                    frame1 = imutils.rotate_bound(frame1, angle=rotNum)
                self.latest_frame = frame1
        cap.release()

def RunSetup(cam_inputs, rotation_input, exposure_input, paramDict,mediaPipeOverlay):
    """
    Start capture-only threads (one per camera) and run the preview windows on
    the MAIN thread. macOS requires all OpenCV GUI calls on the main thread.
    Pressing ESC in any preview window stops every camera at once.
    """
    if not cam_inputs:
        raise ValueError("Camera input list (cam_inputs) is empty")

    stop_event = threading.Event()
    ulist = []

    for cam_input, cam_rotation, cam_exposure in zip(cam_inputs, rotation_input, exposure_input):
        if mediaPipeOverlay == True:
            u = MediaPipeVideoSetup(cam_input, paramDict, cam_rotation, cam_exposure, stop_event)
        else:
            u = VideoSetup(cam_input, paramDict, cam_rotation, cam_exposure, stop_event)
        u.start()
        ulist.append(u)

    # ---- Main-thread preview loop (required on macOS) ----
    try:
        while any(t.is_alive() for t in ulist):
            for t in ulist:
                frame = t.latest_frame
                if frame is not None:
                    cv2.imshow(t.window_name, frame)
            if (cv2.waitKey(1) & 0xFF) == 27:  # ESC stops all cameras
                stop_event.set()
                break
    finally:
        stop_event.set()
        for k in ulist:
            k.join()
        cv2.destroyAllWindows()


# def detect_charuco_board(image,  annotate_image=True):
#     """
#     Charuco base pose estimation.
#     more-or-less copied from - https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco
#     /sandbox/ludovic/aruco_calibration_rotation.html
#     """
#     charuco_corners = []
#     charuco_ids = []
    
#     aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_4X4_250)
#     charuco_length = 7
#     charuco_width = 5

#     board = cv2.aruco.CharucoBoard(charuco_length, charuco_width, 1, .8, aruco_dict)
#     global num_charuco_corners
#     num_charuco_corners = (charuco_length-1) * (charuco_width-1)


#     # SUB PIXEL CORNER DETECTION CRITERION
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     aruco_square_corners, aruco_square_ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray,
#         aruco_dict)

#     if len(aruco_square_corners) > 0:
#         # SUB PIXEL DETECTION
#         for this_corner in aruco_square_corners:
#             cv2.cornerSubPix(gray, this_corner,
#                 winSize=(3, 3),
#                 zeroZone=(-1, -1),
#                 criteria=criteria)
#         res2 = cv2.aruco.interpolateCornersCharuco(aruco_square_corners, aruco_square_ids, gray,
#             board)

#         if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3:
#             charuco_corners = res2[1]
#             charuco_ids = res2[2]


#     return charuco_corners, charuco_ids, aruco_square_corners, aruco_square_ids

# def annotate_image_with_charuco_data(image, charuco_corners, charuco_ids)->bool:

#     full_charuco_detected_on_this_frame = False
#     if len(charuco_ids) == num_charuco_corners:
#         full_charuco_detected_on_this_frame = True

#     image_w_markers = cv2.aruco.drawDetectedCornersCharuco(image,
#                                                             np.array(charuco_corners),
#                                                             np.array(charuco_ids),
#                                                             (200,100,200,255)) #I think cv2 uses BGR instead of RGB?


#     text_to_write_on_this_camera = ''
#     current_cam_corner_count_str = str(
#         len(charuco_ids)) + " of " + str(
#         num_charuco_corners) + " ChAruco Corner Points detected | Full Board Detected: " + str(
#         full_charuco_detected_on_this_frame)
#     # TODO - Determine 'shared views' (i.e. frames in which a full board is detected by 2 cameras)
#     # TODO - self.determine_shared_charuco_board_views()
#     # this_cam_shared_views_str = " | Shared Views: " + str(
#     #     each_cameras_shared_board_view_count_total)
#     text_to_write_on_this_camera = current_cam_corner_count_str

#     position = (10, 50)
#     cv2.putText(
#         image_w_markers,  # numpy array on which text is written
#         text_to_write_on_this_camera,  # text
#         position,  # position at which writing has to start
#         cv2.FONT_HERSHEY_SIMPLEX,  # font family
#         .5,  # font size
#         (0, 10, 0, 255),  # font color
#         4)  # font stroke (draw a darker heavier font beneath a lighter/thinner copy for readability)

#     cv2.putText(
#         image_w_markers,  # numpy array on which text is written
#         text_to_write_on_this_camera,  # text
#         position,  # position at which writing has to start
#         cv2.FONT_HERSHEY_SIMPLEX,  # font family (very limited selection, i think there's some interesting CV history here...)
#         .5,  # font size
#         (209, 180, 0, 255),  # font color
#         2 ) # font stroke

#     if full_charuco_detected_on_this_frame:
#         cv2.polylines(image_w_markers, np.int32([charuco_corners]), True, (0, 255, 255), 4)
#         # for these_corners in charuco_points_from_previous_frames:
#         # if len(these_corners)>0:
#         #     cv2.polylines(image_w_markers, np.int32([these_corners]), True, (0,100,255,255/2), 2)

#     return True