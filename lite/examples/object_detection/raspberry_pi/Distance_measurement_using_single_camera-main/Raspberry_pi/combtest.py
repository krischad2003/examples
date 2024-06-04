# limitations under the License.

"""Main script to run the object detection routine."""
import argparse
import sys
import time
import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
import AiPhile

# Variables
KNOWN_DISTANCE = 76.2  # centimeter
KNOWN_WIDTH = 14.3  # centimeter
WHITE = (255, 255, 255)

# Focal length calculation function
def focal_length(measured_distance, real_width, width_in_rf_image):
    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value

# Distance estimation function
def distance_finder(focal_length, real_width, width_in_frame):
    distance = (real_width * focal_length) / width_in_frame
    return distance

# Load reference image and calculate focal length
ref_image = cv2.imread("../Ref_image.png")

def object_width_in_image(image, model, num_threads, enable_edgetpu):
    # Initialize the object detection model
    base_options = core.BaseOptions(
        file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
    detection_options = processor.DetectionOptions(
        max_results=1, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = vision.TensorImage.create_from_array(rgb_image)
    
    detection_result = detector.detect(input_tensor)
    if detection_result.detections:
        bbox = detection_result.detections[0].bounding_box
        object_width = bbox.width
    else:
        raise ValueError("No object detected in reference image.")
    
    return object_width

# Initialize object detector for reference image processing
model_path = "efficientdet_lite0.tflite"
ref_image_object_width = object_width_in_image(ref_image, model_path, num_threads=4, enable_edgetpu=False)
focal_length_found = focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, ref_image_object_width)
print(f"Focal Length Found: {focal_length_found}")

# Run object detection with distance estimation
def run(model: str, camera_id: int, width: int, height: int, num_threads: int, enable_edgetpu: bool) -> None:
    counter, fps = 0, 0
    start_time = time.time()

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    base_options = core.BaseOptions(
        file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
    detection_options = processor.DetectionOptions(
        max_results=3, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')

        counter += 1
        image = cv2.flip(image, 1)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        detection_result = detector.detect(input_tensor)
        image = utils.visualize(image, detection_result)

        for detection in detection_result.detections:
            bbox = detection.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

            if w != 0:  # To avoid division by zero
                Distance = distance_finder(focal_length_found, KNOWN_WIDTH, w)
                cv2.rectangle(image, (x, y), (x + w, y + h), WHITE, 2)
                AiPhile.textBGoutline(image, f"Distance = {round(Distance, 2)} CM", (x, y - 10), scaling=0.5, text_color=AiPhile.YELLOW)

        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        fps_text = 'FPS = {:.1f}'.format(fps)
        text_location = (left_margin, row_size)
        cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)

        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('object_detector', image)

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='Path of the object detection model.', required=False, default='efficientdet_lite0.tflite')
    parser.add_argument('--cameraId', help='Id of camera.', required=False, type=int, default=0)
    parser.add_argument('--frameWidth', help='Width of frame to capture from camera.', required=False, type=int, default=640)
    parser.add_argument('--frameHeight', help='Height of frame to capture from camera.', required=False, type=int, default=480)
    parser.add_argument('--numThreads', help='Number of CPU threads to run the model.', required=False, type=int, default=4)
    parser.add_argument('--enableEdgeTPU', help='Whether to run the model on EdgeTPU.', action='store_true', required=False, default=False)
    args = parser.parse_args()

    run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight, int(args.numThreads), bool(args.enableEdgeTPU))

if __name__ == '__main__':
    main()
