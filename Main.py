import cv2
import numpy as np
from typing import List
from svgpathtools import Path, Line, CubicBezier, QuadraticBezier, wsvg
from math import sin, pi
from functools import lru_cache
import subprocess
import time
import Svg_Converter

def take_picture(imagepath : str):
    cap = cv2.VideoCapture(0)
    cap.set(3,480)
    cap.set(4,620)
    cap.set(10,100)
    while True:
        ret, frame = cap.read()
        img = cv2.flip(frame , 1)
        cv2.imshow("Camera", img)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite(imagepath , frame)
            cap.release()
            cv2.destroyAllWindows()
            break



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



imgg = cv2.imread("C:/Users/pc/Desktop/CNC_plotter_photo/Docc.jpg")
imagepath = "C:/Users/pc/Desktop/CNC_plotter_photo/Docc.jpg"
cv2.imwrite(imagepath, imgg)
outpath = "TestingImages/Processed.svg"


SvgObject = Svg_Converter.SvgConverter(10 , 4)
all_lines = SvgObject.SvgToSin(imagepath)
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

