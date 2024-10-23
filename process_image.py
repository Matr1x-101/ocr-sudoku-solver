import cv2
import keras
import imutils
import numpy as np
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border

model = keras.models.load_model("keras_mnist_model.h5")

def locate_puzzle(image, debug=True):
    # Convert to grayscale and blur the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    
    # Apply adaptive thresholding and invert the image
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    thresh = cv2.bitwise_not(thresh)
    
    # Debug: Display thresholded image if `debug` is True
    if debug:
        cv2.imshow("Puzzle Threshold", thresh)
        cv2.waitKey(0)

    # Find contours and sort them by area in descending order
    contours = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Initialize the puzzle outline and search for a contour with four points
    puzzleOutline = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            puzzleOutline = approx
            break
    
    # Handle case where no puzzle outline is found
    if puzzleOutline is None:
        raise Exception("Could not find Sudoku puzzle outline. "
                        "Try debugging your thresholding and contour steps.")
    
    # Debugging: Display the detected puzzle outline if `debug` is True
    if debug:
        output = image.copy()
        cv2.drawContours(output, [puzzleOutline], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", output)
        cv2.waitKey(0)
    
    # Apply perspective transform to get a top-down view of the puzzle
    puzzle = four_point_transform(image, puzzleOutline.reshape(4, 2))
    warpedResult = four_point_transform(gray, puzzleOutline.reshape(4, 2))
    
    # Debugging: Display the transformed puzzle if `debug` is True
    if debug:
        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)

    # Return the puzzle in both RGB and grayscale
    return (puzzle, warpedResult)

def extract_digit(cell_image, debug=False):
    # Apply thresholding to the cell and clear any connected borders
    threshold_image = cv2.threshold(cell_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
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

    #  If no contours are found, return None (empty cell)
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

image = cv2.imread("sudoku.jpg")
image = imutils.resize(image, width=600)

# Call the function
puzzle, warpedResult = locate_puzzle(image)

# Initialise the board with numpy
board = np.zeros((9, 9), dtype='int')

mX = warpedResult.shape[1] // 9
mY = warpedResult.shape[0] // 9
print(len(warpedResult))
coords = []

for i in range(9):
    rowArray = []
    for j in range(9):
        startX = i * mX
        startY = j * mY
        endX = (i + 1) * mX
        endY = (j + 1) * mY
        print(startY, endY)
        cell = warpedResult[startY:endY, startX:endX]
        print(cell)
        digit = extract_digit(cell)
        if digit is not None:
            pre_nn_digit = cv2.resize(digit, (28, 28))
            cv2.imshow("Digit", pre_nn_digit)
            pre_nn_digit = pre_nn_digit.astype("float") / 255.0
            pre_nn_digit = keras.preprocessing.image.img_to_array(pre_nn_digit)
            pre_nn_digit = np.expand_dims(pre_nn_digit, axis=0)

            predictedDigit = model.predict(pre_nn_digit).argmax(axis=1)[0]
            board[j, i] = predictedDigit
            rowArray.append(None)
        else:
            rowArray.append((startX, startY, endX, endY))
    coords.append(rowArray)

print(board.tolist())