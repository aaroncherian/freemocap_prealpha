import threading
import cv2
import time
import pickle
import os
import platform


# OpenCV HighGUI windows are safest on macOS when all namedWindow/imshow/waitKey
# calls happen on the main thread. Camera worker threads write frames here; the
# main thread displays them from runcams.RecordCams().
flag = False
preview_lock = threading.Lock()
preview_frames = {}


def request_stop():
    """Ask all camera recording threads to stop."""
    global flag
    flag = True


def reset_recording_state():
    """Reset shared state before starting a new recording."""
    global flag
    flag = False
    with preview_lock:
        preview_frames.clear()


def get_preview_frames():
    """Return a copy of the latest preview frames for main-thread display."""
    with preview_lock:
        return {
            cam_id: frame.copy()
            for cam_id, frame in preview_frames.items()
            if frame is not None
        }


def _store_preview_frame(cam_id, frame):
    with preview_lock:
        preview_frames[cam_id] = frame.copy()


class CamRecordingThread(threading.Thread):
    def __init__(
        self, session, camID, unix_camID, camInput, videoName, rawVidPath, beginTime, parameterDictionary, exposure_setting
    ):
        threading.Thread.__init__(self)
        self.camID = camID
        self.unix_camID = unix_camID
        self.camInput = camInput
        self.videoName = videoName
        self.rawVidPath = rawVidPath
        self.beginTime = beginTime
        self.parameterDictionary = parameterDictionary
        self.session = session
        self.exposure = exposure_setting
        self.timeStamps = ([], [])

    def run(self):
        print("Starting " + self.camID)
        self.timeStamps = CamRecording(
            self.session,
            self.camID,
            self.unix_camID,
            self.camInput,
            self.videoName,
            self.rawVidPath,
            self.beginTime,
            self.parameterDictionary,
            self.exposure
        )

    def getStamps(self):
        return self.timeStamps


def _open_capture(camInput):
    if platform.system() == 'Windows':
        return cv2.VideoCapture(camInput, cv2.CAP_DSHOW)
    if platform.system() == 'Darwin':
        return cv2.VideoCapture(camInput, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(camInput, cv2.CAP_ANY)


def _configure_capture(cam, parameterDictionary, exposure):
    resWidth = parameterDictionary.get("resWidth")
    resHeight = parameterDictionary.get("resHeight")
    framerate = parameterDictionary.get("framerate")

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, resWidth)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resHeight)
    if framerate is not None:
        cam.set(cv2.CAP_PROP_FPS, framerate)

    # These settings are useful for some Windows/DirectShow webcams, but on macOS
    # AVFoundation does not reliably support them through OpenCV and they can
    # cause empty frames or camera-open instability.
    if platform.system() != 'Darwin':
        cam.set(cv2.CAP_PROP_EXPOSURE, exposure)
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    actual_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cam.get(cv2.CAP_PROP_FPS)
    print(f"Camera actual mode: {actual_width}x{actual_height} @ {actual_fps:.1f} fps")
    return resWidth, resHeight, framerate


def _dump_timestamps(session, camID, unix_camID, timeStamps, timeStamps_unix):
    with open(session.rawVidPath / camID, "wb") as f:
        pickle.dump(timeStamps, f)
    with open(session.rawVidPath / unix_camID, "wb") as g:
        pickle.dump(timeStamps_unix, g)


# the recording function that each threaded camera object runs
def CamRecording(
    session, camID, unix_camID, camInput, videoName, rawVidPath, beginTime, parameterDictionary, exposure
):
    """
    Runs the recording process for one camera thread. Saves a video to the RawVideos folder
    and per-frame timestamps to pickle files.

    Important macOS compatibility note: this worker thread does not call cv2.namedWindow,
    cv2.imshow, cv2.waitKey, or cv2.destroyWindow. The main thread owns all preview windows.
    """
    cam = _open_capture(camInput)

    resWidth, resHeight, framerate = _configure_capture(cam, parameterDictionary, exposure)
    codec = parameterDictionary.get("codec")
    fourcc = cv2.VideoWriter_fourcc(*codec)
    saveRawVidPath = str(rawVidPath / videoName)

    out = cv2.VideoWriter(saveRawVidPath, fourcc, framerate, (resWidth, resHeight))
    if not out.isOpened():
        print(f"WARNING: VideoWriter failed to open for {saveRawVidPath}")

    timeStamps = []
    timeStamps_unix = [beginTime]

    try:
        success = cam.isOpened()
        if not success:
            print(f"Camera {camID} failed to open at input {camInput}")

        while success:
            if flag:
                break

            success, frame = cam.read()
            if not success or frame is None:
                break

            _store_preview_frame(camID, frame)

            frame_sized = cv2.resize(frame, (resWidth, resHeight))
            out.write(frame_sized)
            timeStamps.append(time.time() - beginTime)
            timeStamps_unix.append(time.time())

    finally:
        _dump_timestamps(session, camID, unix_camID, timeStamps, timeStamps_unix)
        cam.release()
        out.release()

    return timeStamps, timeStamps_unix


# this is how we sync our time frames, based on our recorded timestamps
