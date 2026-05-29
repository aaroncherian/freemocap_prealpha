import threading
import cv2
import time
import pickle
import os
import platform

class CamRecordingThread(threading.Thread):
    def __init__(
        self, session, camID, unix_camID, camInput, videoName, rawVidPath, beginTime, parameterDictionary, exposure_setting, stop_event=None
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
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.latest_frame = None
        self.window_name = "RECORDING - " + str(camID) + ' - Press ESC to exit'
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
            self.exposure,
            self,
        )

    def getStamps(self):
        return self.timeStamps


# the recording function that each threaded camera object runs (NO GUI here;
# macOS requires all OpenCV window calls on the main thread - see RecordCams)
def CamRecording(
    session, camID, unix_camID, camInput, videoName, rawVidPath, beginTime, parameterDictionary, exposure, thread_handle=None
):
    """
    Capture + write loop for one camera, run in a worker thread. Stores each
    frame on thread_handle.latest_frame for the main thread to preview, and
    stops when thread_handle.stop_event is set (ESC in any preview window).
    Saves a video to RawVideos and pickles the per-frame timestamps.
    """
    stop_event = thread_handle.stop_event if thread_handle is not None else None

    if platform.system() == 'Windows':
        cam = cv2.VideoCapture(camInput, cv2.CAP_DSHOW)
    else:
        cam = cv2.VideoCapture(camInput, cv2.CAP_ANY)

    resWidth = parameterDictionary.get("resWidth")
    resHeight = parameterDictionary.get("resHeight")
    framerate = parameterDictionary.get("framerate")
    codec = parameterDictionary.get("codec")

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, resWidth)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resHeight)
    if platform.system() == 'Windows':
        # AVFoundation rejects FOURCC and uses a different exposure scale;
        # only apply these on Windows.
        cam.set(cv2.CAP_PROP_EXPOSURE, exposure)
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    fourcc = cv2.VideoWriter_fourcc(*codec)

    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("width:", width, "height:", height)
    saveRawVidPath = str(rawVidPath / videoName)

    timeStamps = []
    timeStamps_unix = []
    timeStamps_unix.append(beginTime)  # first unix timestamp is always the begin time

    def _dump_timestamps():
        with open(session.rawVidPath/camID, "wb") as f:
            pickle.dump(timeStamps, f)
        with open(session.rawVidPath/unix_camID, "wb") as g:
            pickle.dump(timeStamps_unix, g)

    if not cam.isOpened():
        print("Could not open camera at input " + str(camInput))
        _dump_timestamps()
        return timeStamps, timeStamps_unix

    # The VideoWriter is created lazily from the first real frame so its size
    # matches what the camera actually delivers (on macOS the camera may ignore
    # the requested resolution; a mismatched writer silently produces an
    # unreadable video).
    out = None

    while stop_event is None or not stop_event.is_set():
        success, frame = cam.read()
        if not success or frame is None:
            # macOS warmup / transient empty read - keep going until stopped
            continue

        if out is None:
            frame_h, frame_w = frame.shape[0], frame.shape[1]
            out = cv2.VideoWriter(saveRawVidPath, fourcc, framerate, (frame_w, frame_h))

        out.write(frame)
        timeStamps.append(time.time() - beginTime)
        timeStamps_unix.append(time.time())
        if thread_handle is not None:
            thread_handle.latest_frame = frame

    _dump_timestamps()
    cam.release()
    if out is not None:
        out.release()
    return timeStamps, timeStamps_unix


# this is how we sync our time frames, based on our recorded timestamps