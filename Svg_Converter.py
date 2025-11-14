import cv2
from functools import lru_cache
import numpy as np
from svgpathtools import Path, Line, CubicBezier, QuadraticBezier, wsvg
from math import sin, pi
from typing import List

class SvgConverter:

    def __init__(self ,imagePath : str , outPath :str ):
        self.imagePath = imagePath
        self.outPath = outPath
        self.image = cv2.imread(self.imagePath)



    def frange(self,start, stop, increment=1.0):
        current = start
        while current < stop:
            yield current
            current += increment

    def addWhiteBorders(self, image, border_size=10):

        bordered = cv2.copyMakeBorder(
            image,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
    @classmethod
    def resize_image(self, image : np.ndarray, height: int):
        # Compute the aspect ratio
        aspect_ratio = float(image.shape[1]) / float(image.shape[0])

        # Calculate the new width based on the target height and original aspect ratio
        width = int(height * aspect_ratio)

        # Resize the image
        resized_image = cv2.resize(image, (width, height))
        return resized_image

    @lru_cache
    def get_range_val(self,start, end, increment, idx):
        return list(self.frange(start, end, increment))[::-1][idx]

    def SvgToSin(self, height : int, pixel_width : float ,resolution = 0.7  , max_amplitude : float =2.0 , max_frequency=3  , white :int = 230, White_Removal : bool = False):

        image = cv2.imread(self.imagePath)
        if White_Removal:
            self.image = self.addWhiteBorders(image , 15)
        outpath_white = "TestingImages/ProcessedWhite.svg"
        image = self.resize_image(image , height)  # adjust height
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        WHITE_THRESHOLD = white
        all_lines = []
        white_paths = []

        for row in range(image.shape[0]):
            line_start_height = (row * pixel_width) + (pixel_width / 2)
            current_x = 0
            current_sin_amplitude = 0
            current_sin_frequency = 20
            current_sin_phase = 0

            sin_line = Path()
            white_line = Path()
            in_white = False

            start_point = complex(current_x, line_start_height)

            for col in range(image.shape[1]):
                pixel = image[row, col]

                target_amp = self.get_range_val(0, max_amplitude, max_amplitude / 255, pixel)
                target_freq = self.get_range_val(0, max_frequency, max_frequency / 255, pixel)

                for _ in self.frange(0, pixel_width, resolution):

                    current_sin_amplitude += (target_amp - current_sin_amplitude) * resolution
                    current_sin_frequency += (target_freq - current_sin_frequency) * resolution
                    current_sin_phase += current_sin_frequency * resolution

                    current_y = (current_sin_amplitude * sin(current_sin_phase)) + line_start_height
                    end_point = complex(current_x, current_y)
                    line = Line(start_point, end_point)

                    if pixel >= WHITE_THRESHOLD:

                        if not in_white:

                            if len(sin_line) > 0:
                                all_lines.append(sin_line)
                                sin_line = Path()
                            in_white = True
                        white_line.append(line)
                        sin_line.append(line)
                    else:

                        if in_white:

                            if len(white_line) > 0:
                                white_paths.append(white_line)
                                white_line = Path()
                            in_white = False
                        sin_line.append(line)

                    current_x += resolution
                    start_point = end_point

            if len(sin_line) > 0:
                all_lines.append(sin_line)
            if len(white_line) > 0:
                white_paths.append(white_line)

        wsvg(paths=all_lines, filename=self.outPath)
        wsvg(paths=white_paths, filename=outpath_white)
        svg_white = f"{outpath_white}"
        svg = f"{self.outPath}"

        return svg, svg_white
