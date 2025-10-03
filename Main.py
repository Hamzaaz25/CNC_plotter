import cv2
import numpy as np
from typing import List
from svgpathtools import Path, Line, CubicBezier, QuadraticBezier, wsvg
from math import sin, pi
from functools import lru_cache
import subprocess


def frange(start, stop, increment=1.0):
    current = start
    while current < stop:
        yield current
        current += increment


def resize_image(image: np.ndarray, height: int):
    # Compute the aspect ratio
    aspect_ratio = float(image.shape[1]) / float(image.shape[0])

    # Calculate the new width based on the target height and original aspect ratio
    width = int(height * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (width, height))
    return resized_image


@lru_cache
def get_range_val(start, end, increment, idx):
    return list(frange(start, end, increment))[::-1][idx]



cap = cv2.VideoCapture(0)
cap.set(3, 590)
cap.set(4, 840)
cap.set(10, 100)
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
while(True):
    success, img = cap.read()
    img = cv2.flip(img, 1)
    cv2.imshow('Out', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break

cv2.imwrite("C:/Users/pc/PycharmProjects/Cnc_Plotter/TestingImages/image.png",img , [cv2.IMWRITE_PNG_COMPRESSION, 9])

imagepath = "C:/Users/pc/PycharmProjects/Cnc_Plotter/TestingImages/image.png"
outpath = "C:/Users/pc/PycharmProjects/Cnc_Plotter/TestingImages/Processed.svg"
height =int(90)
pixel_width =int(4)
resolution = 0.7
max_amplitude = 3
max_frequency = 3



image = cv2.imread(imagepath)
image = resize_image(image, height)  # adjust height
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # make it grayscale

all_lines: List[Path] = []
for row in range(image.shape[0]):
    print(row,end=" ")
    sin_line = Path()
    current_x = 0
    current_sin_amplitude = 0
    current_sin_frequency = 20
    current_sin_phase = 0
    line_start_height = (row * pixel_width) + (pixel_width / 2)
    start_point = complex(current_x, line_start_height)

    for col in range(image.shape[1]):
        pixel = image[row, col]

        # 255 is max value of grayscale pixel
        target_sin_amplitude = get_range_val(0, max_amplitude,
                                                 max_amplitude / 255,
                                                 pixel)
        target_sin_frequency = get_range_val(0, max_frequency,
                                                 max_frequency / 255,
                                                 pixel)

        for _ in frange(0, pixel_width, resolution):
            sin_amplitude_diff = target_sin_amplitude - current_sin_amplitude
            current_sin_amplitude += sin_amplitude_diff * resolution

            sin_frequency_diff = target_sin_frequency - current_sin_frequency
            current_sin_frequency += sin_frequency_diff * resolution

                # keep track of phase
                # y = amp * sin((frequency * x) + phase)
                # phase_shift = phase/frequency -> phase is args.resolution
                # phase = frequency * phase_shift
            phase_diff = current_sin_frequency * resolution
            current_sin_phase += phase_diff

            current_y = (current_sin_amplitude * sin(current_sin_phase)) + line_start_height
            end_point = complex(current_x, current_y)
            line = Line(start_point, end_point)
            sin_line.append(line)

            current_x += resolution
            start_point = end_point

    all_lines.append(sin_line)


wsvg(paths=all_lines, filename=outpath)
print(f"SVG saved as {outpath}")
subprocess.run(f"vpype read {outpath} linemerge linesort gwrite -p step_motor \"C:/Users/pc/PycharmProjects/Cnc_Plotter/TestingImages/meow.gc\"")
print(f"Gcode saved")