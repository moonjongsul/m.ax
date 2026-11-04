"""Real-time top-down tray view.

usage: python3 run_live.py [camera_index_or_video]   (default: camera 0)
keys:  q = quit
"""
import sys
import cv2
from tray_tracker import TrayTracker, warp_tray

REF_IMG = 'ref.png'
# tray top-rim corners in ref.png (x, y): TL, TR, BR, BL
REF_CORNERS = [[392, 119], [562, 133], [603, 307], [382, 321]]

src = sys.argv[1] if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
tracker = TrayTracker(cv2.imread(REF_IMG), REF_CORNERS)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame.shape[:2] != (480, 640):
        frame = cv2.resize(frame, (640, 480))  # reference corners are in 640x480 pixels

    corners, tracked = tracker.update(frame)
    out = warp_tray(frame, corners)

    dbg = frame.copy()
    cv2.polylines(dbg, [corners.astype('int32')], True, (0, 255, 0) if tracked else (0, 0, 255), 2)
    cv2.imshow('camera', dbg)
    cv2.imshow('tray', out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
