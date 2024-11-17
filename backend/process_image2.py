
import cv2
import numpy as np
import imutils
import keras

from skimage.segmentation import clear_border

img = cv2.imread('test.jpg')
model = keras.models.load_model("keras_mnist_model.h5")

import cv2
import numpy as np

def locate_puzzle(img):
    """
    Locate and extract a puzzle grid from an input image.
    
    Parameters:
    img (numpy.ndarray): Input image in which the puzzle is to be located.
    
    Returns:
    numpy.ndarray: The final transformed image containing the extracted puzzle grid.
    """
    
    # Step 1: Display the raw input image
    winname = "raw image"
    cv2.namedWindow(winname)
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 100, 100)

    # Step 2: Apply Gaussian blur to smooth the image and reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    winname = "blurred"
    cv2.namedWindow(winname)
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 100, 150)
    
    # Step 3: Convert the blurred image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((gray.shape), np.uint8)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    winname = "gray"
    cv2.namedWindow(winname)
    cv2.imshow(winname, gray)
    cv2.moveWindow(winname, 100, 200)

    # Step 4: Enhance the contrast using morphological operations
    close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel1)
    div = np.float32(gray) / close
    res = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
    res2 = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    winname = "res2"
    cv2.namedWindow(winname)
    cv2.imshow(winname, res2)
    cv2.moveWindow(winname, 100, 250)
    
    # Step 5: Thresholding and contour detection to locate the puzzle area
    thresh = cv2.adaptiveThreshold(res, 255, 0, 1, 19, 2)
    contour, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 6: Find the largest contour which likely corresponds to the puzzle area
    max_area = 0
    best_cnt = None
    for cnt in contour:
        area = cv2.contourArea(cnt)
        if area > 1000 and area > max_area:
            max_area = area
            best_cnt = cnt

    # Draw the best contour onto a mask
    cv2.drawContours(mask, [best_cnt], 0, 255, -1)
    cv2.drawContours(mask, [best_cnt], 0, 0, 2)
    res = cv2.bitwise_and(res, mask)

    winname = "puzzle only"
    cv2.namedWindow(winname)
    cv2.imshow(winname, res)
    cv2.moveWindow(winname, 100, 300)
    cv2.imwrite("puzzleonly.jpg", res)
    
    # Step 7: Detect vertical lines
    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
    dx = cv2.Sobel(res, cv2.CV_16S, 1, 0)
    dx = cv2.convertScaleAbs(dx)
    cv2.normalize(dx, dx, 0, 255, cv2.NORM_MINMAX)
    _, close = cv2.threshold(dx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernelx, iterations=1)

    # Remove small noise contours
    contour, hier = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x, y, w, h = cv2.boundingRect(cnt)
        if h / w > 10:
            cv2.drawContours(close, [cnt], 0, 255, -1)
        else:
            cv2.drawContours(close, [cnt], 0, 0, -1)
    closex = cv2.morphologyEx(close, cv2.MORPH_CLOSE, None, iterations=2)

    # Step 8: Detect horizontal lines
    kernely = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    dy = cv2.Sobel(res, cv2.CV_16S, 0, 2)
    dy = cv2.convertScaleAbs(dy)
    cv2.normalize(dy, dy, 0, 255, cv2.NORM_MINMAX)
    _, close = cv2.threshold(dy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernely)
    contour, hier = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x, y, w, h = cv2.boundingRect(cnt)
        if w / h > 10:
            cv2.drawContours(close, [cnt], 0, 255, -1)
        else:
            cv2.drawContours(close, [cnt], 0, 0, -1)
    closey = cv2.morphologyEx(close, cv2.MORPH_DILATE, None, iterations=2)

    # Step 9: Find intersections of vertical and horizontal lines
    res = cv2.bitwise_and(closex, closey)
    winname = "intersections"
    cv2.namedWindow(winname)
    cv2.imshow(winname, res)
    cv2.moveWindow(winname, 100, 450)

    # Step 10: Extract centroids of detected intersections
    contour, hier = cv2.findContours(res, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contour:
        mom = cv2.moments(cnt)
        if mom['m00'] != 0:
            x, y = int(mom['m10'] / mom['m00']), int(mom['m01'] / mom['m00'])
            centroids.append((x, y))
    
    # Sort centroids into a 10x10 grid
    centroids = np.array(centroids, dtype=np.float32).reshape((100, 2))
    c2 = centroids[np.argsort(centroids[:, 1])]
    b = np.vstack([c2[i * 10:(i + 1) * 10][np.argsort(c2[i * 10:(i + 1) * 10, 0])] for i in range(10)])
    bm = b.reshape((10,10,2))
    
    # Make a copy for labeling in order
    labeled_in_order = res2.copy()

    textcolor = (0, 0, 255)  # Blue for text
    pointcolor = (255, 0, 0)  # Blue for points

    # Label the centroids with numbers in the output image
    for index, pt in enumerate(b):
        cv2.putText(labeled_in_order, str(index), tuple(map(int, pt)), cv2.FONT_HERSHEY_DUPLEX, 0.75, textcolor)
        cv2.circle(labeled_in_order, tuple(map(int, pt)), 5, pointcolor)

    # Display and save the labeled image
    winname = "labeled in order"
    cv2.namedWindow(winname)
    cv2.imshow(winname, labeled_in_order)
    cv2.moveWindow(winname, 100, 500)
    cv2.imwrite("labelled.jpg", labeled_in_order)  # Save the labeled image
    
    # Step 11: Generate the final output by transforming the puzzle grid
    output = np.zeros((450, 450, 3), np.uint8)
    for i, j in enumerate(b):
        ri, ci = int(i / 10), i % 10
        if ci != 9 and ri != 9:
            src = bm[ri:ri + 2, ci:ci + 2].reshape((4, 2))
            dst = np.array([[ci * 50, ri * 50], [(ci + 1) * 50 - 1, ri * 50], [ci * 50, (ri + 1) * 50 - 1], [(ci + 1) * 50 - 1, (ri + 1) * 50 - 1]], np.float32)
            retval = cv2.getPerspectiveTransform(src, dst)
            warp = cv2.warpPerspective(res2, retval, (450, 450))
            output[ri * 50:(ri + 1) * 50 - 1, ci * 50:(ci + 1) * 50 - 1] = warp[ri * 50:(ri + 1) * 50 - 1, ci * 50:(ci + 1) * 50 - 1].copy()

    # Step 12: Display and save the final output
    winname = "final"
    cv2.namedWindow(winname)
    cv2.imshow(winname, output)
    cv2.moveWindow(winname, 600, 100)
    cv2.waitKey(0)
    cv2.imwrite("output.jpg", output)
    
    return output


def extract_digit(cell_image, debug=False):
    # Convert the image to grayscale (single channel) before applying thresholding
    gray_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to the grayscale cell and clear any connected borders
    threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    if threshold_image is None:
        raise ValueError("Thresholding failed, resulting in NoneType.")
    
    threshold_image = clear_border(threshold_image)
    
    # Debugging: Visualize the thresholded cell if `debug` is True
    if debug:
        cv2.imshow("Cell Threshold", threshold_image)
        cv2.waitKey(0)

    # Find contours in the thresholded cell image
    contours = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # If no contours are found, return None (empty cell)
    if len(contours) == 0:
        return None

    # Find the largest contour and create a mask for it
    largest_contour = max(contours, key=cv2.contourArea)
    contour_mask = np.zeros(threshold_image.shape, dtype='uint8')
    cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)

    # Calculate the percentage of the mask that is filled
    height, width = threshold_image.shape
    filled_area_percent = cv2.countNonZero(contour_mask) / float(width * height)

    # Ignore the contour if it's less than 3% of the cell area (likely noise)
    if filled_area_percent < 0.03:
        return None

    # Apply the mask to the thresholded cell to isolate the digit
    extracted_digit = cv2.bitwise_and(threshold_image, threshold_image, mask=contour_mask)

    # Debugging: Visualize the extracted digit if `debug` is True
    if debug:
        cv2.imshow("Extracted Digit", extracted_digit)
        cv2.waitKey(0)

    # Return the extracted digit
    return extracted_digit


output = locate_puzzle(img)
board = np.zeros((9, 9), dtype='int')

for i in range(9):
    for j in range(9):
        startX = i * 50
        startY = j * 50
        endX = (i+1) * 50
        endY = (j+1) * 50
        cell = output[startX:endX, startY:endY]
        digit = extract_digit(cell)
        if digit is not None:
            pre_nn_digit = cv2.resize(digit, (28, 28))
            cv2.imshow("Digit", pre_nn_digit)
            pre_nn_digit = pre_nn_digit.astype("float") / 255.0
            pre_nn_digit = keras.preprocessing.image.img_to_array(pre_nn_digit)
            pre_nn_digit = np.expand_dims(pre_nn_digit, axis=0)

            predictedDigit = model.predict(pre_nn_digit).argmax(axis=1)[0]
            board[i, j] = predictedDigit
print(board)
