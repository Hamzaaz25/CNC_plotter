import cv2
from functools import lru_cache
import numpy as np
from svgpathtools import Path, Line, CubicBezier, QuadraticBezier, wsvg
from math import sin, pi

class SvgConverter:
    def __init__(self , height : int, pixel_width : int ,resolution = 0.7  , max_amplitude=2 , max_frequency=3 ):
        self.height = height
        self.pixel_width = pixel_width
        self.resolution = resolution
        self.max_amplitude = max_amplitude
        self.max_frequency = max_frequency

    def frange(self,start, stop, increment=1.0):
        current = start
        while current < stop:
            yield current
            current += increment

    def resize_image(self,image: np.ndarray, height: int):
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

    def SvgToSin (self, imagepath : str) :
        image = cv2.imread(imagepath)
        image = self.resize_image(image, self.height)  # adjust height
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # make it grayscale

        all_lines: List[Path] = []
        for row in range(image.shape[0]):
            print(row, end=" ")
            sin_line = Path()
            current_x = 0
            current_sin_amplitude = 0
            current_sin_frequency = 20
            current_sin_phase = 0
            line_start_height = (row * self.pixel_width) + (self.pixel_width / 2)
            start_point = complex(current_x, line_start_height)

            for col in range(image.shape[1]):
                pixel = image[row, col]

                # WHITE_THRESHOLD = 230
                # if pixel >= WHITE_THRESHOLD:
                #     # end the current line path
                #     if len(sin_line) > 0:
                #         all_lines.append(sin_line)
                #         sin_line = Path()
                #     current_x += pixel_width
                #     start_point = complex(current_x, line_start_height)
                #     continue

                # 255 is max value of grayscale pixel
                target_sin_amplitude = self.get_range_val(0,self.max_amplitude,
                                                     self.max_amplitude / 255,
                                                     pixel)
                target_sin_frequency = self.get_range_val(0, self.max_frequency,
                                                     self.max_frequency / 255,
                                                     pixel)

                for _ in self.frange(0, self.pixel_width, self.resolution):
                    sin_amplitude_diff = target_sin_amplitude - current_sin_amplitude
                    current_sin_amplitude += sin_amplitude_diff * self.resolution

                    sin_frequency_diff = target_sin_frequency - current_sin_frequency
                    current_sin_frequency += sin_frequency_diff * self.resolution

                    # keep track of phase
                    # y = amp * sin((frequency * x) + phase)
                    # phase_shift = phase/frequency -> phase is args.resolution
                    # phase = frequency * phase_shift
                    phase_diff = current_sin_frequency * self.resolution
                    current_sin_phase += phase_diff

                    current_y = (current_sin_amplitude * sin(current_sin_phase)) + line_start_height
                    end_point = complex(current_x, current_y)
                    line = Line(start_point, end_point)
                    sin_line.append(line)

                    current_x += self.resolution
                    start_point = end_point

            all_lines.append(sin_line)
        return all_lines
