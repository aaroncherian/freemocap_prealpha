from tqdm import tqdm
import cv2
import os
import platform
import time


# How long to let USB cameras finish enumerating before we start probing.
# On macOS several identical UVC cameras come online over a ~1s window after
# the process starts; probing too early silently drops the slow ones.
_WARMUP_BEFORE_PROBE_S = 1.0

# Each index gets several read attempts before we conclude no camera is there,
# so a camera that is merely slow to wake is not dropped.
_READ_ATTEMPTS = 8
_READ_RETRY_DELAY_S = 0.15

# We only need to look at a small number of ports. range(20) mostly just adds
# slow misses on macOS (each empty index still pays an open/timeout cost).
_MAX_PORTS_TO_CHECK = 10


def TestDevice(source):
    """
    Check whether a usable camera exists at the given index.

    Retries the first read several times before giving up, because on macOS a
    freshly-enumerated camera can take a moment before it delivers frames. A
    single failed read (the old behaviour) made detection race-y: the same
    camera would appear on one run and vanish on the next purely on timing.
    Returns the index if a frame is captured, otherwise None.
    """
    if platform.system() == 'Windows':
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_ANY)

    try:
        if not cap.isOpened():
            return None

        for _ in range(_READ_ATTEMPTS):
            success, image = cap.read()
            if success and image is not None:
                return source
            time.sleep(_READ_RETRY_DELAY_S)
        return None
    finally:
        # NOTE: no cv2.destroyAllWindows() here. This function is called from
        # inside the Tk GUI (recordGUI.__init__); mixing OpenCV HighGUI teardown
        # with the Tk event loop can trigger the "Tcl_FindHashEntry on deleted
        # table" crash. No windows are opened here, so there is nothing to
        # destroy anyway.
        cap.release()


def CreateAvailableCamList():
    """
    Build the list of camera indices that have a usable camera attached.

    Gives the USB cameras a brief moment to finish coming online before
    probing, then checks each port with retries (see TestDevice). This is what
    makes the detected count stable run-to-run instead of "sometimes all the
    cameras, sometimes not".
    """
    # Let cameras finish enumerating before the first probe.
    time.sleep(_WARMUP_BEFORE_PROBE_S)

    openCamList = []
    for x in tqdm(range(_MAX_PORTS_TO_CHECK)):
        openCamera = TestDevice(x)
        if openCamera is not None:
            openCamList.append(openCamera)
        # Small gap so a just-released device (AVFoundation tears sessions down
        # asynchronously) does not make the next probe miss.
        time.sleep(0.1)

    return openCamList