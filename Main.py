import cv2
import numpy as np
from typing import List
from svgpathtools import Path, Line, CubicBezier, QuadraticBezier, wsvg
from math import sin, pi
from functools import lru_cache
import subprocess
import time
import serial
from threading import Event


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
time.sleep(3)
imagepath = "TestingImages/image.png"
cv2.imwrite(imagepath, img ,[cv2.IMWRITE_PNG_COMPRESSION, 9])

outpath = "TestingImages/Processed.svg"
height =int(90)
pixel_width =int(4)
resolution = 0.7
max_amplitude = 4
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
svg = f"C:/Users/pc/PycharmProjects/Cnc_Plotter/{outpath}"
subprocess.run(f"vpype --config \"C:/Users/pc/PycharmProjects/Cnc_Plotter/TestingImages/myconfig.toml\" read {svg}  scale 0.3mm 0.3mm linemerge linesort gwrite -p marlin_servo \"C:/Users/pc/PycharmProjects/Cnc_Plotter/TestingImages/meow.gc\"")
print(f"Gcode saved")

time.sleep(10)
"""
This is a simple script that attempts to connect to the GRBL controller at 
> /dev/tty.usbserial-A906L14X
It then reads the grbl_test.gcode and sends it to the controller

The script waits for the completion of the sent line of gcode before moving onto the next line

tested on
> MacOs Monterey arm64
> Python 3.9.5 | packaged by conda-forge | (default, Jun 19 2021, 00:24:55) 
[Clang 11.1.0 ] on darwin
> Vscode 1.62.3
> Openbuilds BlackBox GRBL controller
> GRBL 1.1
"""
#
#
#
# BAUD_RATE = 115200
#
#
# def remove_comment(string):
#     if (string.find(';') == -1):
#         return string
#     else:
#         return string[:string.index(';')]
#
#
# def remove_eol_chars(string):
#     # removed \n or trailing spaces
#     return string.strip()
#
#
# def send_wake_up(ser):
#     # Wake up
#     # Hit enter a few times to wake the Printrbot
#     ser.write(str.encode("\r\n\r\n"))
#     time.sleep(2)   # Wait for Printrbot to initialize
#     ser.flushInput()  # Flush startup text in serial input
#
#
# def wait_for_movement_completion(ser, cleaned_line):
#     Event().wait(1)
#
#     if cleaned_line != '$X' or '$$':
#         idle_counter = 0
#
#         while True:
#             # Event().wait(0.01)
#             ser.reset_input_buffer()
#             command = str.encode('?' + '\n')
#             ser.write(command)
#             grbl_out = ser.readline()
#             grbl_response = grbl_out.strip().decode('utf-8')
#
#             if grbl_response != 'ok':
#                 if grbl_response.find('Idle') > 0:
#                     idle_counter += 1
#
#             if idle_counter > 10:
#                 break
#     return
#
#
# def stream_gcode(GRBL_port_path, gcode_path):
#     # with context opens file/connection and closes it if function(with) scope is left
#     with open(gcode_path, "r") as file, serial.Serial(GRBL_port_path, BAUD_RATE) as ser:
#         send_wake_up(ser)
#         for line in file:
#             # cleaning up gcode from file
#             cleaned_line = remove_eol_chars(remove_comment(line))
#             if cleaned_line:  # checks if string is empty
#                 print("Sending gcode: " + str(cleaned_line))
#                 # converts string to byte encoded string and append newline
#                 command = str.encode(line + '\n')
#                 ser.write(command)  # Send g-code
#
#                 wait_for_movement_completion(ser, cleaned_line)
#
#                 grbl_out = ser.readline()  # Wait for response with carriage return
#                 print(" : ", grbl_out.strip().decode('utf-8'))
#
#         print('End of gcode')
#
#
# if __name__ == "__main__":
#     # GRBL_port_path = '/dev/tty.usbserial-A906L14X'
#     GRBL_port_path = 'COM13'
#     gcode_path = "C:/Users/pc/PycharmProjects/Cnc_Plotter/TestingImages/meow.gc"
#
#     print("USB Port: ", GRBL_port_path)
#     print("Gcode file: ", gcode_path)
#     stream_gcode(GRBL_port_path, gcode_path)
#     print('EOF')
#
#
#


































