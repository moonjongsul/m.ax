import cv2, numpy as np, sys

src_path, out_path = sys.argv[1], sys.argv[2]
img = cv2.imread(src_path)

# tray top-rim corners (x, y): TL, TR, BR, BL in the original image
pts = np.float32([[392, 119], [562, 133], [603, 307], [382, 321]])

OUT_W, OUT_H = 640, 480  # final size after CCW 90° rotation

# warp to (OUT_H x OUT_W) first, so rotating CCW 90° gives OUT_W x OUT_H
w, h = OUT_H, OUT_W
dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
H = cv2.getPerspectiveTransform(pts, dst)
top = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_CUBIC)
out = cv2.rotate(top, cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.imwrite(out_path, out)
print('size', out.shape[1], 'x', out.shape[0])
