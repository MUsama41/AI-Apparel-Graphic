import cv2
import numpy as np
import torch.nn.functional as nnf
from PIL import Image
import math
from scipy.ndimage import rotate
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch.nn as nn
from struct import unpack
import traceback

# Load the pre-trained model and processor
processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer-b2-fashion")
model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer-b2-fashion")

label_mapping = {
    2: 'top, t-shirt, sweatshirt'
}

def get_bounding_box_of_large_region(image, min_area=7000, kernel_size=(35, 35)):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    kernel = np.ones(kernel_size, np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
    if filtered_contours:
        largest_contour = max(filtered_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
        output_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        output_image_path = 'ref2_seg.png'
        cv2.imwrite(output_image_path, output_image)
        return x, y, w, h, output_image_path
    else:
        return None

def get_model_predictions(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()
    return logits

def upsample_logits(logits, size):
    return nnf.interpolate(logits, size=size, mode="bilinear", align_corners=False)

def get_predicted_segmentation(upsampled_logits):
    return upsampled_logits.argmax(dim=1)[0].numpy()

def label_to_color(label):
    color_img = np.zeros((*label.shape, 3), dtype=np.uint8)
    for class_index in label_mapping.keys():
        color_img[label == class_index] = [255, 20, 147]
    return color_img

def extract_roi(pred_seg, label):
    return (pred_seg == label).astype(np.uint8)

def get_shirt_bounding_box(image, processor, model, min_area=5000, kernel_size_open=(10, 10), kernel_size_close=(10, 10)):
    logits = get_model_predictions(image, processor, model)
    upsampled_logits = upsample_logits(logits, size=image.size[::-1])
    pred_seg = get_predicted_segmentation(upsampled_logits)
    roi = extract_roi(pred_seg, label=2)
    kernel_open = np.ones(kernel_size_open, np.uint8)
    kernel_close = np.ones(kernel_size_close, np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel_open)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel_close)
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
    if filtered_contours:
        largest_contour = max(filtered_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        image_np = np.array(image)
        return x, y, w, h, image_np, pred_seg, roi
    else:
        return None, None, None, None, np.array(image), pred_seg, roi

def calculate_centroid(coords):
    """Calculate the centroid of the coordinates."""
    centroid = np.mean(coords, axis=0)
    return centroid

def adjust_coordinates_inward(coords, centroid, factor):
    """Adjust coordinates inward towards the centroid by the given factor."""
    direction = centroid - coords
    adjusted_coords = coords + factor * direction
    adjusted_coords = np.clip(adjusted_coords, 0, None)
    return adjusted_coords.astype(int)

def create_adjusted_mask(image_shape, coords):
    """Create a new mask from the adjusted coordinates."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [coords], 255)
    return mask


def calculate_angle_between_peaks(image_path, x, y, w, h):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not loaded correctly. Please check the file path.")
    bounding_box = image[y:y+h, x:x+w]
    half_height = h // 2
    half_width = w // 2
    upper_left_quadrant = bounding_box[:half_height, :half_width]
    upper_right_quadrant = bounding_box[:half_height, half_width:]
    def find_first_peak_point(quadrant, offset_x, offset_y):
        for row in range(quadrant.shape[0]):
            for col in range(quadrant.shape[1]):
                if quadrant[row, col] == 255:
                    if row == 0 or quadrant[row - 1, col] == 0:
                        return (offset_x + col, offset_y + row)
        return None
    peak_point_upper_left = find_first_peak_point(upper_left_quadrant, x, y)
    peak_point_upper_right = find_first_peak_point(upper_right_quadrant, x + half_width, y)
    if peak_point_upper_left and peak_point_upper_right:
        dx = peak_point_upper_right[0] - peak_point_upper_left[0]
        dy = peak_point_upper_right[1] - peak_point_upper_left[1]
        angle = math.degrees(math.atan2(dy, dx))
    else:
        angle = None
    return angle

def get_blend_ratio(shirt_intensity):
    if shirt_intensity < 0.25:
        return 0.51
    elif shirt_intensity < 0.5:
        return 0.5
    elif shirt_intensity < 0.75:
        return 0.2
    else:
        return shirt_intensity

def adjust_intensity(color, intensity, intensity_factor):
    intensity = intensity + intensity_factor
    return color * intensity

def overlay_logo(image, logo, x, y, w, h, rotation_angle, threshold_val, intensity_factor, x_offset=0, y_offset=0):
    logo_width = int(w * 0.55)
    logo_height = int(h * 0.26)
    logo = logo.resize((logo_width, logo_height), Image.ANTIALIAS)
    logo_np = np.array(logo)
    rotated_logo = rotate(logo_np, angle=rotation_angle, axes=(1, 0), reshape=True)
    center_x = x + w // 2 + x_offset
    center_y = y + h // 2 + y_offset
    logo_x1 = center_x - rotated_logo.shape[1] // 2
    logo_y1 = y + int(h * 0.20) + y_offset
    shirt_bbox = image[y:y+h, x:x+w]
    shirt_bbox_normalized = shirt_bbox / 255.0
    for i in range(rotated_logo.shape[0]):
        for j in range(rotated_logo.shape[1]):
            if rotated_logo[i, j, 3] > 0:
                shirt_pixel = shirt_bbox_normalized[logo_y1 + i - y, logo_x1 + j - x]
                shirt_pixel_intensity = np.mean(shirt_pixel)
                if shirt_pixel_intensity < threshold_val:
                    blend_ratio = get_blend_ratio(shirt_pixel_intensity)
                    logo_pixel = rotated_logo[i, j, :3] / 255.0
                    adjusted_logo_pixel = adjust_intensity(logo_pixel, shirt_pixel_intensity, intensity_factor)
                    alpha = rotated_logo[i, j, 3] / 255.0
                    blended_pixel = (1 - alpha) * shirt_pixel + alpha * adjusted_logo_pixel
                    blended_pixel = np.clip(blended_pixel * 255, 0, 255).astype(np.uint8)
                    image[logo_y1 + i, logo_x1 + j] = blended_pixel
                else:
                    logo_pixel = rotated_logo[i, j, :3] / 255.0
                    alpha = rotated_logo[i, j, 3] / 255.0
                    blended_pixel = (1 - alpha) * shirt_pixel + alpha * logo_pixel
                    blended_pixel = np.clip(blended_pixel * 255, 0, 255).astype(np.uint8)
                    image[logo_y1 + i, logo_x1 + j] = blended_pixel
    return image

# Function to read ACV file and return RGB curves
def read_curve(acv_file):
    curve = []
    number_of_points_in_curve, = unpack("!h", acv_file.read(2))
    for j in range(number_of_points_in_curve):
        y, x = unpack("!hh", acv_file.read(4))
        curve.append((x, y))
    return curve

def adjust_curve(curve, factor):
    adjusted_curve = []
    for x, y in curve:
        y = int(y * factor)
        y = min(max(y, 0), 255)
        adjusted_curve.append((x, y))
    return adjusted_curve

def adjust_tones(curve, shadow_factor, midtone_factor, highlight_factor):
    adjusted_curve = []
    for x, y in curve:
        if x < 85:
            factor = shadow_factor
        elif x < 170:
            factor = midtone_factor
        else:
            factor = highlight_factor
        y = int(y * factor)
        y = min(max(y, 0), 255)
        adjusted_curve.append((x, y))
    return adjusted_curve

def return_polynomial_coefficients(curve_list):
    xdata = [x[0] for x in curve_list]
    ydata = [x[1] for x in curve_list]
    np.set_printoptions(precision=6)
    np.set_printoptions(suppress=True)
    p = np.polyfit(xdata, ydata, len(curve_list)-1)
    return p

def apply_curve(image_channel, curve):
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = np.clip(np.polyval(curve, i), 0, 255)
    return image_channel.point(list(lut))

def process_image(acv_file_path, image, red_factor, green_factor, blue_factor, shadow_factor, midtone_factor, highlight_factor):
    try:
        with open(acv_file_path, "rb") as acv_file:
            _, nr_curves = unpack("!hh", acv_file.read(4))
            curves = []
            for i in range(nr_curves):
                curves.append(read_curve(acv_file))
            compositeCurve = curves[0]
            redCurve = curves[1]
            greenCurve = curves[2]
            blueCurve = curves[3]

            adjustedRedCurve = adjust_curve(redCurve, red_factor)
            adjustedRedCurve = adjust_tones(adjustedRedCurve, shadow_factor, midtone_factor, highlight_factor)
            pRed = return_polynomial_coefficients(adjustedRedCurve)

            adjustedGreenCurve = adjust_curve(greenCurve, green_factor)
            adjustedGreenCurve = adjust_tones(adjustedGreenCurve, shadow_factor, midtone_factor, highlight_factor)
            pGreen = return_polynomial_coefficients(adjustedGreenCurve)

            adjustedBlueCurve = adjust_curve(blueCurve, blue_factor)
            adjustedBlueCurve = adjust_tones(adjustedBlueCurve, shadow_factor, midtone_factor, highlight_factor)
            pBlue = return_polynomial_coefficients(adjustedBlueCurve)

        if image.mode == 'RGBA':
            r, g, b, a = image.split()
        else:
            r, g, b = image.split()
            a = None

        r = apply_curve(r, pRed)
        g = apply_curve(g, pGreen)
        b = apply_curve(b, pBlue)

        if a is not None:
            result_image = Image.merge("RGBA", (r, g, b, a))
        else:
            result_image = Image.merge("RGB", (r, g, b))

        return result_image

    except Exception as e:
        print('ERROR, UNEXPECTED EXCEPTION')
        print(str(e))
        traceback.print_exc()
        return None

def get_combined_image(image, acv_file_path, red_factor, green_factor, blue_factor, shadow_factor, midtone_factor, highlight_factor):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    sleeves_label = 32
    neckline_label = 34
    shirt_label = 2

    mask = (pred_seg == sleeves_label) | (pred_seg == neckline_label) | (pred_seg == shirt_label)

    # Create a colored mask
    original_np = np.array(image)
    mask_np = mask.numpy()
    colored_mask = np.zeros_like(original_np)
    colored_mask[mask_np > 0] = original_np[mask_np > 0]

    # Find contours of the mask
    contours, _ = cv2.findContours(mask_np.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_coords = np.vstack(contours).squeeze()

    # Calculate the centroid of the edge coordinates
    centroid = calculate_centroid(edge_coords)

    # Adjust the coordinates inward by a factor (e.g., 0.05 for 5% inward)
    adjustment_factor = 0.03

    adjusted_coords = adjust_coordinates_inward(edge_coords, centroid, adjustment_factor)

    # Create the adjusted mask
    adjusted_mask = create_adjusted_mask(original_np.shape, adjusted_coords)

    # Extract the corresponding region from the original image using the new mask
    reduced_shirt_image = np.zeros_like(original_np)
    reduced_shirt_image[adjusted_mask > 0] = original_np[adjusted_mask > 0]

    # Convert reduced_shirt_image to PIL Image
    reduced_shirt_image_pil = Image.fromarray(reduced_shirt_image.astype(np.uint8))

    # Apply the RGB curves to the reduced shirt image
    reduced_shirt_image_with_curves = process_image(acv_file_path, reduced_shirt_image_pil, red_factor, green_factor, blue_factor, shadow_factor, midtone_factor, highlight_factor)

    # Combine the modified segment back with the original image
    segmented_np = np.array(reduced_shirt_image_with_curves)
    combined_image_np = np.where(np.expand_dims(adjusted_mask, axis=-1), segmented_np, original_np)

    combined_image = Image.fromarray(combined_image_np.astype(np.uint8))
    return combined_image



def process_shirt_files(shirt_files,target_rgb):
    combined_images = []
    output_path = 'result_image.jpg'
    standard_image_path = '200.jpg'
    for shirt_image in shirt_files:
        #combined_image = get_combined_image(shirt_image, target_rgb)
        result_image = change_shirt_color(shirt_image, standard_image_path, target_rgb, output_path)

        combined_images.append(combined_image)
    return combined_images
