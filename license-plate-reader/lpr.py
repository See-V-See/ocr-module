import cv2
import imutils
import numpy
import pytesseract

from skimage.segmentation import clear_border
from lpr_debbuger import LPRDebugger


# The class that contains the methods for reading license plate text from images.
# As parameters to the constructor it accepts values that correspond to the aspect ratio of rectangular plates.
class LicensePlateReader:

    def __init__(self, min_aspect_ratio=4, max_aspect_ratio=5, debug_mode_on=True):
        
        self.min_aspect_ratio = min_aspect_ratio  # The minimum aspect ratio used to detect and filter license plates.
        self.max_aspect_ratio = max_aspect_ratio  # The maximum aspect ratio used to detect and filter license plates.
        self.debugger = LPRDebugger(debug_mode_on)

    # The method that contains the image processing pipeline.
    # Also, it finds the contour candidates from the image.
    def find_license_plate_candidate_regions(self, gray_image, contours_count=5):
        
        rectangle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        black_hat_image = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, rectangle_kernel)
        self.debugger.debug_imshow("Black Hat", black_hat_image)
        
        square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light_regions = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, square_kernel)
        light_regions = cv2.threshold(light_regions, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debugger.debug_imshow("Light Regions", light_regions)
        
        gradient_x_image = cv2.Sobel(black_hat_image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradient_x_image = numpy.absolute(gradient_x_image)
        (min_value, max_value) = (numpy.min(gradient_x_image), numpy.max(gradient_x_image))
        gradient_x_image = 255 * ((gradient_x_image - min_value) / (max_value - min_value))
        gradient_x_image = gradient_x_image.astype("uint8")
        self.debugger.debug_imshow("Scharr Filter on X axis", gradient_x_image)
        
        gradient_x_image = cv2.GaussianBlur(gradient_x_image, (5, 5), 0)
        gradient_x_image = cv2.morphologyEx(gradient_x_image, cv2.MORPH_CLOSE, rectangle_kernel)
        threshold_gradient_x_image = cv2.threshold(gradient_x_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debugger.debug_imshow("Threshold on the Gradient X image", threshold_gradient_x_image)

        # TODO substitute with Open
        threshold_gradient_x_image = cv2.erode(threshold_gradient_x_image, None, iterations=2)
        threshold_gradient_x_image = cv2.dilate(threshold_gradient_x_image, None, iterations=2)
        self.debugger.debug_imshow("Threshold GradientX Image after Erode & Dilate", threshold_gradient_x_image)

        threshold_gradient_x_image = cv2.bitwise_and(
            threshold_gradient_x_image,
            threshold_gradient_x_image,
            mask=light_regions)

        # TODO substitute with Close
        threshold_gradient_x_image = cv2.dilate(threshold_gradient_x_image, None, iterations=2)
        threshold_gradient_x_image = cv2.erode(threshold_gradient_x_image, None, iterations=2)
        self.debugger.debug_imshow("Final Image", threshold_gradient_x_image, wait_key=True)

        contours = cv2.findContours(threshold_gradient_x_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:contours_count]

        return contours

    def locate_license_plate(self, gray_image, candidates, clear_image_border=False):

        license_plate_contour = None
        region_of_interest = None

        for candidate in candidates:
            (x, y, w, h) = cv2.boundingRect(candidate)
            aspect_ratio = float(w) / float(h)

            if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                license_plate_contour = candidate
                license_plate_snip = gray_image[y:y + h, x:x + w]
                self.debugger.debug_imshow("License Plate Snip", license_plate_snip)

                region_of_interest = cv2.threshold(
                    license_plate_snip,
                    0,
                    255,
                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
                )[1]

                if clear_image_border:
                    region_of_interest = clear_border(region_of_interest)

                self.debugger.debug_imshow("Region Of Interest", region_of_interest, wait_key=True)
                break
        
        return region_of_interest, license_plate_contour

    # TODO Refactor this.
    @staticmethod
    def get_tesseract_options(psm=7):
        alpha_numeric_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alpha_numeric_chars)
        options += " --psm {}".format(psm)

        return options

    def find_and_extract_text(self, image, psm=7, clear_image_border=False):
        
        license_plate_text = None

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = self.find_license_plate_candidate_regions(gray_image)

        (region_of_interest, license_plate_contour) = self.locate_license_plate(
            gray_image,
            candidates,
            clear_image_border=clear_image_border
        )

        if region_of_interest is not None:
            options = self.get_tesseract_options(psm=psm)
            license_plate_text = pytesseract.image_to_string(region_of_interest, config=options)
            self.debugger.debug_imshow("License Plate Region Of Interest", region_of_interest)

        return license_plate_text, license_plate_contour
