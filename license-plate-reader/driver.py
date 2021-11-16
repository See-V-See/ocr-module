import argparse
import cv2
import imutils

from imutils import paths
from lpr import LicensePlateReader


def cleanup_text(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()


arguments = argparse.ArgumentParser()

arguments.add_argument("-i",
                       "--input",
                       required=True,
                       help="Path to the input directory.")

arguments.add_argument("-c",
                       "--clear-border",
                       type=int,
                       default=-1,
                       help="Clear border pixels before OCR.")

arguments.add_argument("-p",
                       "--psm",
                       type=int,
                       default=7,
                       help="The PSM mode for the OCR.")

arguments.add_argument("-d",
                       "--debug",
                       type=int,
                       default=-1,
                       help="Debug mode on(1) or off(-1).")

print("The args have been set.")

args = vars(arguments.parse_args())

lpr_instance = LicensePlateReader(debug_mode_on=args["debug"] > 0)
image_paths = sorted(list(paths.list_images(args["input"])))

print("The input path: " + str(image_paths))

for path in image_paths:
    image = cv2.imread(path)
    image = imutils.resize(image, width=600)

    print("The path: " + path)

    (license_plate_text, license_plate_contour) = lpr_instance.find_and_extract_text(
        image,
        psm=args["psm"],
        clear_image_border=args["clear_border"] > 0
    )

    print(license_plate_text)

    if license_plate_text is not None and license_plate_contour is not None:
        box = cv2.boxPoints(cv2.minAreaRect(license_plate_contour))
        box = box.astype("int")
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

        (x, y, w, h) = cv2.boundingRect(license_plate_contour)
        cv2.putText(image, cleanup_text(license_plate_text), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        print("[INFO] {}".format(license_plate_text))
        cv2.imshow("Output Image", image)
        cv2.waitKey(0)   

# Run with this: python driver.py --input "PATH_TO_IMAGES"
