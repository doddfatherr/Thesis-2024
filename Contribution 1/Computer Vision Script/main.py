import cv2
import numpy as np
from scipy.signal import find_peaks

# Select Video from files
vid = "2024-05-08 08-11-46.mov"
cap = cv2.VideoCapture(vid)

# Get video details
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the video codec and create a VideoWriter object
output_video = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Use 'XVID' codec for AVI format
out = cv2.VideoWriter(output_video,fourcc,30.0, (width, height))

grip_aperture_list = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # ------------------------------------
    # Detect Black Dots

    #Grayscale image
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get black areas as white
    _, binary = cv2.threshold(gray_image, 20, 255, cv2.THRESH_BINARY_INV)
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on size
    min_area = 900  # Adjust this to your specific size requirements
    max_area = 15000
    min_height = 30  # Minimum height to be considered as a valid contour
    min_width = 30
    max_height = 50
    max_width = 50
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area and cv2.contourArea(cnt) < max_area and cv2.boundingRect(cnt)[3] > min_height and cv2.boundingRect(cnt)[3]< max_height and cv2.boundingRect(cnt)[2] < max_width and cv2.boundingRect(cnt)[2] > min_width]
    dot_coordinate_list = []
    black_threshold = 255

    #center = (width // 2, height // 2)
    #exclusion_width, exclusion_height = 200, 150  # Adjust these values as needed
    #exclusion_zone = (center[0] - exclusion_width, center[1], exclusion_width, exclusion_height)

    # Draw contours
    #if len(large_contours) == 2:
    for contour in large_contours:
        # Get the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        #Filter to see if it is in the middle of the frame
        length_x = frame.shape[1]
        length_y = frame.shape[0]
        if x > length_x*0.25 and x< length_x*0.70:
            #if x > length_x*0.55 or x < length_x*0.45 or y > length_y*0.60 or y < length_y*0.45:
            #if x > length_x*0.40:
            roi = binary[y:y+h, x:x+w]
            if cv2.mean(roi)[0] < black_threshold:  # color filter
                #if not(x > exclusion_zone[0] and x + w < exclusion_zone[0] + exclusion_zone[2] and y > exclusion_zone[1] and y + h < exclusion_zone[1] + exclusion_zone[3]):
                # Draw a rectangle around the contour
                dot_coordinate_list.append((x, y))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    grip_aperture_pixels = 0
    if len(dot_coordinate_list)==2:
            x1, y1 = dot_coordinate_list[0]
            x2, y2 = dot_coordinate_list[1]
            grip_aperture_pixels = np.sqrt((x2-x1)**2 + (y2-y1)**2)

    #---------------------------------------------------------------------------

    #Detect Ruler values

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to get a binary image
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Re-find contours on the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Set Black thresholding value
    black_threshold = 255
    color_filtered_contours = []

    sizey= frame.shape[0]
    sizex = frame.shape[1]

    min_area = 10        # minimum area
    max_area = 5000     # maximum area
    min_height = 5     # minimum height
    min_width =3       # minimum width
    max_height = 20     # maximmum height
    max_width = 10      # maximum width
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area and cv2.contourArea(cnt) < max_area and cv2.boundingRect(cnt)[3] > min_height and cv2.boundingRect(cnt)[3]< max_height and cv2.boundingRect(cnt)[2] < max_width and cv2.boundingRect(cnt)[2] > min_width]
    coordinates_list = []
    for c in large_contours:
        # Bounding box of the contour
        x, y, w, h = cv2.boundingRect(c)
        # Region of interest
        roi = thresh[y:y+h, x:x+w]
        if cv2.mean(roi)[0] < black_threshold:  # color filter
            if x < 0.1 * sizex and y > 0.95 * sizey:  # bottom left quadrant filter
                coordinates_list.append((x, y))
                color_filtered_contours.append(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    pixels_per_inch = 0
    x1 = x2 = y1 = y2 =0
    if len(coordinates_list)==2:
        x1, y1 = coordinates_list[0]
        x2, y2 = coordinates_list[1]
        pixels_per_inch = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        #pixels_per_inch = np.sqrt((x2-x1)**2 + (y2-y1)**2)*1.778
#------------------------------------------------------------------------------------------------------
#Final Distance calculation

    #Calculation
    grip_aperture_inches = 0
    if pixels_per_inch != 0:
        grip_aperture_inches = (grip_aperture_pixels /pixels_per_inch)
        grip_aperture_list.append(grip_aperture_inches)

    #Display Distance
    cv2.putText( frame, f"{round(grip_aperture_inches, 3)} in", (int(x2),int(y2)), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255))
    cv2.imshow('frame', frame)
    out.write(frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()

peaks, _ = find_peaks(grip_aperture_list, prominence = 0.4, distance = 50)
print(peaks)
peak_loc = []
for idx in peaks:
    loc = grip_aperture_list[idx]
    peak_loc.append(loc)
print(peak_loc)
