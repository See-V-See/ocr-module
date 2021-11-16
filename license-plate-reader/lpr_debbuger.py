import cv2


# The class that contains debug functionality.
class LPRDebugger:

    def __init__(self, debug_mode_on):
        self.debug_mode_on = debug_mode_on  # A flag value used to display intermediate results

    # The method that implements a debug version for the imshow() method.
    # It is used to debug the image processing pipeline.   
    def debug_imshow(self, title, image, wait_key=True):
        
        if self.debug_mode_on:
            cv2.imshow(title, image)
            
            if wait_key:
                cv2.waitKey(0)
