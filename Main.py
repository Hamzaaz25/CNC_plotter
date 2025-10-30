import cv2
import numpy as np
from typing import List
from svgpathtools import Path, Line, CubicBezier, QuadraticBezier, wsvg
from math import sin, pi
from functools import lru_cache
import subprocess
import time




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

def center_gcode(lines, bed_width, bed_height):
    x_vals, y_vals = [], []

    # المرحلة 1: جمع الإحداثيات X و Y من أسطر الحركة فقط
    for line in lines:
        if line.startswith(('G1', 'G0')):
            parts = line.split()
            for p in parts:
                if p.startswith('X'):
                    x_vals.append(float(p[1:]))
                elif p.startswith('Y'):
                    y_vals.append(float(p[1:]))

    if not x_vals or not y_vals:
        return lines

    # حساب الأبعاد الحالية والرسم داخل مساحة الورقة
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)

    draw_width = max_x - min_x
    draw_height = max_y - min_y

    offset_x = (bed_width - draw_width) / 2 - min_x
    offset_y = (bed_height - draw_height) / 2 - min_y

    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('G1', 'G0')):
            parts = []
            for p in stripped.split():
                if p.startswith('X'):
                    parts.append(f"X{float(p[1:]) + offset_x:.3f}")
                elif p.startswith('Y'):
                    parts.append(f"Y{float(p[1:]) + offset_y:.3f}")
                else:
                    parts.append(p)
            new_line = ' '.join(parts)
        else:
            new_line = stripped
        # ✳️ تأكد من وجود newline بعد كل سطر
        new_lines.append(new_line + '\n')

    return new_lines


# Take a picture
cap = cv2.VideoCapture(0)
cap.set(3, 590)
cap.set(4, 840)
cap.set(10, 100)

while(True):
    success, img = cap.read()
    img = cv2.flip(img, 1)
    cv2.imshow('Out', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break

imgg = cv2.imread("C:/Users/pc/Desktop/CNC_plotter_photo/Docc.jpg")
imagepath = "C:/Users/pc/Desktop/CNC_plotter_photo/Docc.jpg"
cv2.imwrite(imagepath, imgg)
outpath = "TestingImages/Processed.svg"

# Svg Sin Photo Parameters
height =int(120)
pixel_width =int(4)
resolution = 0.7
max_amplitude = 3
max_frequency = 3

#Adjust the image params
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

all_lines = [p for p in all_lines if len(p) > 0]
wsvg(paths=all_lines, filename=outpath)
print(f"SVG saved as {outpath}")
svg = f"C:/Users/pc/PycharmProjects/Cnc_Plotter/{outpath}"
# Convert to gcode line
subprocess.run(f"vpype --config \"C:/Users/pc/PycharmProjects/Cnc_Plotter/TestingImages/myconfig.toml\" read {svg} translate 10mm 10mm scale 0.3mm 0.3mm linemerge linesort gwrite -p marlin_servo \"C:/Users/pc/PycharmProjects/Cnc_Plotter/TestingImages/meow.gc\"")
print(f"Gcode saved")
with open("C:/Users/pc/PycharmProjects/Cnc_Plotter/TestingImages/meow.gc") as f:
    lines = f.readlines()

centered = center_gcode(lines, bed_width=297, bed_height=210)

with open("TestingImages/Centered.gc", "w") as f:
    f.writelines(centered)
time.sleep(5)

from grbl_uploader import GRBLUploader

if __name__ == "__main__":
    uploader = GRBLUploader(port="COM17")
    uploader.connect()
    uploader.start_stream("TestingImages/Centered.gc")

    # time.sleep(5)
    # uploader.Pause()
    # print("Paused...")
    # time.sleep(3)
    # uploader.resume()
    # print("Resumed...")

    # uploader.stop()  # Optional emergency stop
    uploader.thread.join()  # Wait until upload finishes

