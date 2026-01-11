"""
Camera Diagnostic Tool for OpenVUE
Run this to identify why your camera isn't working.
"""
import cv2
import sys
import time

def check_opencv_info():
    print("=" * 60)
    print("OPENCV INFORMATION")
    print("=" * 60)
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Build info available backends:")

    # Check available backends
    backends = [
        (cv2.CAP_DSHOW, "DirectShow (Windows)"),
        (cv2.CAP_MSMF, "Microsoft Media Foundation"),
        (cv2.CAP_ANY, "Auto-detect"),
        (cv2.CAP_V4L2, "Video4Linux2 (Linux)"),
    ]

    for backend_id, name in backends:
        print(f"  - {name}: backend ID {backend_id}")
    print()

def test_camera_with_backend(index, backend, backend_name, timeout=5):
    """Test a specific camera index with a specific backend."""
    print(f"  Testing camera {index} with {backend_name}...")

    try:
        cap = cv2.VideoCapture(index, backend)
    except Exception as e:
        print(f"    ERROR: Failed to create capture: {e}")
        return False

    if not cap.isOpened():
        print(f"    FAILED: Could not open camera")
        cap.release()
        return False

    print(f"    Camera opened, waiting for frames...")

    # Try to read frames
    start_time = time.time()
    success_count = 0
    black_frame_count = 0

    while time.time() - start_time < timeout:
        ret, frame = cap.read()
        if ret and frame is not None:
            # Check if frame is black/empty
            mean_val = frame.mean()
            if mean_val > 5:  # Not a black frame
                success_count += 1
                if success_count >= 3:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    fourcc_raw = int(cap.get(cv2.CAP_PROP_FOURCC))
                    fourcc_str = "".join([chr((fourcc_raw >> 8 * i) & 0xFF) for i in range(4)])
                    print(f"    SUCCESS! Resolution: {width}x{height}, FPS: {fps}, Format: {fourcc_str}")
                    cap.release()
                    return True
            else:
                black_frame_count += 1
        time.sleep(0.1)

    if black_frame_count > 0:
        print(f"    FAILED: Got {black_frame_count} black/empty frames")
    else:
        print(f"    FAILED: No valid frames received in {timeout}s")

    cap.release()
    return False

def test_all_cameras():
    print("=" * 60)
    print("CAMERA DETECTION TEST")
    print("=" * 60)

    backends_to_test = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Auto-detect"),
    ]

    working_configs = []

    for cam_index in range(6):
        print(f"\nCamera index {cam_index}:")
        for backend, name in backends_to_test:
            if test_camera_with_backend(cam_index, backend, name):
                working_configs.append((cam_index, backend, name))

    return working_configs

def test_camera_formats(index, backend):
    """Test different pixel formats."""
    print(f"\nTesting pixel formats for camera {index}...")

    formats = [
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
        ('YUY2', cv2.VideoWriter_fourcc(*'YUY2')),
        ('NV12', cv2.VideoWriter_fourcc(*'NV12')),
        ('RGB3', cv2.VideoWriter_fourcc(*'RGB3')),
    ]

    for name, fourcc in formats:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            time.sleep(0.5)
            ret, frame = cap.read()
            if ret and frame is not None and frame.mean() > 5:
                print(f"  {name}: SUCCESS")
            else:
                print(f"  {name}: FAILED")
            cap.release()
        else:
            print(f"  {name}: Could not open camera")

def show_test_feed(index, backend, duration=5):
    """Show a test video feed."""
    print(f"\nShowing test feed for {duration} seconds...")
    print("Press 'q' to quit early.\n")

    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        print("Failed to open camera for test feed")
        return

    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Camera Test Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def main():
    print("\n" + "=" * 60)
    print("OPENVUE CAMERA DIAGNOSTIC TOOL")
    print("=" * 60 + "\n")

    check_opencv_info()

    working_configs = test_all_cameras()

    print("\n" + "=" * 60)
    print("DIAGNOSTIC RESULTS")
    print("=" * 60)

    if working_configs:
        print(f"\nFound {len(working_configs)} working camera configuration(s):")
        for idx, (cam_idx, backend, name) in enumerate(working_configs):
            print(f"  {idx + 1}. Camera {cam_idx} with {name} backend")

        # Test formats on the first working camera
        best_cam, best_backend, best_name = working_configs[0]
        test_camera_formats(best_cam, best_backend)

        print("\n" + "-" * 60)
        print("RECOMMENDATION:")
        print(f"  Use camera index {best_cam} with {best_name} backend")

        if best_backend != cv2.CAP_DSHOW:
            print(f"\n  The code currently uses DirectShow (CAP_DSHOW).")
            print(f"  Try changing to {best_name} (CAP_{'MSMF' if best_backend == cv2.CAP_MSMF else 'ANY'})")

        # Offer to show test feed
        response = input("\nShow test feed? (y/n): ").strip().lower()
        if response == 'y':
            show_test_feed(best_cam, best_backend)
    else:
        print("\nNO WORKING CAMERAS FOUND!")
        print("\nPossible causes:")
        print("  1. Camera is being used by another application (Teams, Zoom, etc.)")
        print("  2. Windows privacy settings are blocking camera access")
        print("  3. Camera drivers need to be updated")
        print("  4. Camera hardware issue")
        print("\nTroubleshooting steps:")
        print("  1. Close all other applications that might use the camera")
        print("  2. Go to Settings > Privacy > Camera and ensure access is enabled")
        print("  3. Update camera drivers via Device Manager")
        print("  4. Try the OBS Virtual Camera workaround:")
        print("     - Install OBS Studio from https://obsproject.com")
        print("     - Add your camera as a video source")
        print("     - Click 'Start Virtual Camera'")
        print("     - Run this diagnostic again")

if __name__ == "__main__":
    main()
