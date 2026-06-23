import sys

try:
    import cv2
except Exception:
    print('OpenCV is not installed or not available in this environment.')
    print('Install it with: pip install opencv-python')
    raise SystemExit(1)


def annotate_frame(frame):
    cv2.putText(
        frame,
        "Object Detection Running",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    return frame


def save_frame(frame, path='detection_output.jpg'):
    if cv2.imwrite(path, frame):
        print(f'Saved frame to {path}')
    else:
        print('Failed to save frame to', path)


def run_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Unable to open camera. Please make sure your webcam is connected and available.')
        return 1

    use_gui = '--no-gui' not in sys.argv
    if use_gui:
        try:
            cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
        except Exception:
            use_gui = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Unable to read from camera. Exiting.')
            break

        frame = annotate_frame(frame)

        if use_gui:
            cv2.imshow('Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            save_frame(frame)
            break

    cap.release()
    if use_gui:
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(run_detection())
