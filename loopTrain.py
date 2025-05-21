import os
import cv2
import numpy as np
import time

# --- Load Ground Truth Data once ---
gt = []
with open("pandasDataset/groundtruth.txt", 'r') as f:
    for line in f:
        vals = line.strip().split(',')
        gt.append([int(float(v)) for v in vals])

# --- Experiment configurations ---
experiment_configs = [
    {'maxCorners': 5,  'qualityLevel': 0.1, 'winSize': (10, 10)},
    {'maxCorners': 10, 'qualityLevel': 0.1, 'winSize': (10, 10)},
    {'maxCorners': 5,  'qualityLevel': 0.3, 'winSize': (10, 10)},
    {'maxCorners': 5,  'qualityLevel': 0.1, 'winSize': (30, 30)},
]

vid_path = 'output.mp4'  # your input video

for idx, config in enumerate(experiment_configs, start=1):
    print(f"\n=== Experiment {idx}: {config} ===")

    # 1) Shi-Tomasi & LK parameters
    ST_PARAMS = dict(
        maxCorners=config['maxCorners'],
        qualityLevel=config['qualityLevel'],
        minDistance=7,
        blockSize=7
    )
    LK_PARAMS = dict(
        winSize=config['winSize'],
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # 2) Open video fresh
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {vid_path}")

    # 3) Read first frame & initialize ROI + features
    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read first frame")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    x, y, w, h = map(int, gt[0][:4])

    # Detect features in the whole frame…
    all_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **ST_PARAMS)
    if all_pts is None:
        raise RuntimeError("No features detected at all!")

    # …then filter those inside the ROI
    sel_pts = [pt for pt in all_pts if x <= pt[0][0] <= x + w and y <= pt[0][1] <= y + h]
    if not sel_pts:
        raise RuntimeError(f"No features in initial bbox with params {config}")

    prevPts = np.array(sel_pts, dtype=np.float32)

    # 4) Prepare video writer
    fps     = cap.get(cv2.CAP_PROP_FPS)
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    outpath = f'output_experiment_{idx}.mp4'
    out     = cv2.VideoWriter(outpath, fourcc, fps, (width, height))

    # 5) Reset mask + counters
    mask = np.zeros_like(prev_frame)
    frame_count = 0
    start_time  = time.time()

    # --- Tracking loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prevPts is not None:
            nextPts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, frame_gray, prevPts, None, **LK_PARAMS
            )

            if nextPts is not None:
                good_new  = nextPts[status == 1]
                good_prev = prevPts[status == 1]

                # draw points
                for new, _ in zip(good_new, good_prev):
                    x_new, y_new = map(int, new.ravel())
                    frame = cv2.circle(frame, (x_new, y_new), 5, (0, 255, 0), -1)

                visualized = cv2.add(frame, mask)
                elapsed = time.time() - start_time
                fps_disp = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(visualized, f"FPS: {fps_disp:.2f}",
                            (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2, cv2.LINE_AA)

                out.write(visualized)
                prevPts = good_new.reshape(-1, 1, 2)
            else:
                out.write(frame)
                prevPts = None
        else:
            out.write(frame)

        prev_gray = frame_gray.copy()
        frame_count += 1

    # --- Cleanup for this experiment ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print(f"→ Saved {outpath}: {frame_count} frames, "
          f"{frame_count/total_time:.2f} avg FPS")
