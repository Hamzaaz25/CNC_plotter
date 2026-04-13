import cv2
import time
import pygame
import sys
import Cmyk_Converter
import Gcode_Converter
from Button import Button
from tkinter import Tk, filedialog
import Svg_Converter
from grbl_uploader import GRBLUploader


# ---------------------------------------------------------------------------
# Image input helpers
# ---------------------------------------------------------------------------

def upload_image():
    root = Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg")]
    )
    if filepath:
        print(f"Selected image path {filepath}")
        img = cv2.imread(filepath)
        cv2.imwrite("./TestingImages/image.png", img)
    else:
        print("No image selected")
        raise Exception("No image selected")


def take_picture():
    cap = cv2.VideoCapture(0)
    cap.set(3, 480)
    cap.set(4, 620)
    cap.set(10, 100)
    while True:
        ret, frame = cap.read()
        img = cv2.flip(frame, 1)
        cv2.imshow("Camera", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("./TestingImages/image.png", img)
            cap.release()
            cv2.destroyAllWindows()
            break


# ---------------------------------------------------------------------------
# Pygame setup
# ---------------------------------------------------------------------------

pygame.init()

SCREEN = pygame.display.set_mode((1280, 720))
pygame.display.set_caption("Menu")

BG_MAIN = pygame.image.load("assets/MainBack.png")
BG_DARK = pygame.image.load("assets/Background.png")

# Pre-load reusable button images once
IMG_PLAY_RECT = pygame.image.load("assets/Play Rect.png")
IMG_OPTIONS_RECT = pygame.image.load("assets/Options Rect.png")
IMG_QUIT_RECT = pygame.image.load("assets/Quit Rect.png")
IMG_CAMERA = pygame.image.load("assets/Camera.png")
IMG_UPLOAD = pygame.image.load("assets/Upload.png")

IMAGE_PATH = "TestingImages/image.png"
OUTPUT_SVG = "./TestingImages/Processed.svg"
GCODE_PATH = "./TestingImages/Gcode.gc"
GCODE_WHITE_PATH = "./TestingImages/White.gc"


def get_font(size):
    return pygame.font.Font("assets/font.ttf", size)


# ---------------------------------------------------------------------------
# Squiggle processing helper
# ---------------------------------------------------------------------------

def process_squiggle(height, white_removal=True, max_amplitude=2.0):
    """Run the SVG-to-Gcode squiggle pipeline for a given line count (height)."""
    svg_converter = Svg_Converter.SvgConverter(IMAGE_PATH, OUTPUT_SVG)
    svg, svg_white = svg_converter.SvgToSin(height, 4, max_amplitude=max_amplitude,
                                             White_Removal=white_removal)
    time.sleep(0.5)
    gcon = Gcode_Converter.GcodeConverter(
        SvgPath=svg, GPath=GCODE_PATH, scale=0.3,
        bed_width=297, bed_height=297
    )
    gcon.firstConvert(SvgWhite=svg_white, GpathWhite=GCODE_WHITE_PATH)
    time.sleep(1.5)
    gcon.secondConvert(GpathWhite=GCODE_WHITE_PATH)


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------

def main_menu():
    while True:
        SCREEN.blit(BG_MAIN, (0, 0))

        mouse_pos = pygame.mouse.get_pos()

        menu_text = get_font(100).render("CNC PLotter", True, "#900000")
        menu_rect = menu_text.get_rect(center=(640, 100))

        play_btn = Button(image=IMG_PLAY_RECT, pos=(640, 250),
                          text_input="DRAW", font=get_font(75),
                          base_color="White", hovering_color="#900000")
        options_btn = Button(image=IMG_OPTIONS_RECT, pos=(640, 400),
                             text_input="OPTIONS", font=get_font(75),
                             base_color="White", hovering_color="#900000")
        quit_btn = Button(image=IMG_QUIT_RECT, pos=(640, 550),
                          text_input="QUIT", font=get_font(75),
                          base_color="White", hovering_color="#900000")

        SCREEN.blit(menu_text, menu_rect)

        for button in [play_btn, options_btn, quit_btn]:
            button.changeColor(mouse_pos)
            button.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if play_btn.checkForInput(mouse_pos):
                    draw()
                elif options_btn.checkForInput(mouse_pos):
                    options()
                elif quit_btn.checkForInput(mouse_pos):
                    pygame.quit()
                    sys.exit()

        pygame.display.update()


def options():
    while True:
        mouse_pos = pygame.mouse.get_pos()

        SCREEN.fill("white")

        options_text = get_font(45).render("Thank You for visiting OPTIONS screen.", True, "Black")
        options_rect = options_text.get_rect(center=(640, 260))
        SCREEN.blit(options_text, options_rect)

        back_btn = Button(image=None, pos=(640, 460),
                          text_input="BACK", font=get_font(75),
                          base_color="Black", hovering_color="Green")
        back_btn.changeColor(mouse_pos)
        back_btn.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if back_btn.checkForInput(mouse_pos):
                    main_menu()

        pygame.display.update()


def draw():
    while True:
        mouse_pos = pygame.mouse.get_pos()

        SCREEN.fill("black")

        title = get_font(80).render("Choose Your Way ", True, "#900000")
        title_rect = title.get_rect(center=(640, 100))
        SCREEN.blit(title, title_rect)

        capture_btn = Button(image=IMG_CAMERA, pos=(640, 300),
                             text_input="CAPTURE  ", font=get_font(55),
                             base_color="White", hovering_color="Red")
        upload_btn = Button(image=IMG_UPLOAD, pos=(640, 460),
                            text_input=" UPLOAD FROM FILES   ", font=get_font(45),
                            base_color="White", hovering_color="Red")
        back_btn = Button(image=None, pos=(640, 570),
                          text_input="BACK", font=get_font(75),
                          base_color="White", hovering_color="Green")

        for btn in [capture_btn, upload_btn, back_btn]:
            btn.changeColor(mouse_pos)
            btn.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if capture_btn.checkForInput(mouse_pos):
                    take_picture()
                    print("Captured")
                    style()
                elif back_btn.checkForInput(mouse_pos):
                    main_menu()
            if event.type == pygame.MOUSEBUTTONUP:
                if upload_btn.checkForInput(mouse_pos):
                    upload_image()
                    print("UPLOAD FROM DEVICE")
                    style()

        pygame.display.update()


def style():
    while True:
        mouse_pos = pygame.mouse.get_pos()
        SCREEN.blit(BG_DARK, (0, 0))

        title = get_font(80).render("Choose Your Style", True, "#900000")
        title_rect = title.get_rect(center=(640, 100))
        SCREEN.blit(title, title_rect)

        squiggle_btn = Button(image=IMG_PLAY_RECT, pos=(410, 370),
                              text_input="Squiggle ", font=get_font(55),
                              base_color="Black", hovering_color="#900000")
        tsp_btn = Button(image=IMG_PLAY_RECT, pos=(850, 370),
                         text_input=" TSP(Beta)", font=get_font(55),
                         base_color="Black", hovering_color="#900000")
        back_btn = Button(image=None, pos=(640, 570),
                          text_input="BACK", font=get_font(75),
                          base_color="White", hovering_color="Green")

        for btn in [squiggle_btn, tsp_btn, back_btn]:
            btn.changeColor(mouse_pos)
            btn.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if tsp_btn.checkForInput(mouse_pos):
                    tsp = Svg_Converter.SvgConverter(imagePath=IMAGE_PATH, outPath=OUTPUT_SVG)
                    tsp.voronoi_tsp_pipeline(
                        image_path=IMAGE_PATH,
                        output_dir="TestingImages",
                        num_points=10000,
                        edge_fraction=0.6,
                        relax_iters=12,
                        relax_alpha=0.8,
                        make_single_path=True
                    )
                    print("tsp")
                    gcon = Gcode_Converter.GcodeConverter(
                        SvgPath=OUTPUT_SVG, GPath="TestingImages/Centered.gc",
                        scale=0.2, bed_width=280, bed_height=200
                    )
                    gcon.gcodeConvert(svg=OUTPUT_SVG, gcode="TestingImages/Centered.gc")
                    gcon.center_gcode("TestingImages/Centered.gc")
                    run_gcode()
                elif squiggle_btn.checkForInput(mouse_pos):
                    print("Squiggle")
                    squiggle_a4()
                    run_gcode()
                elif back_btn.checkForInput(mouse_pos):
                    draw()

        pygame.display.update()


def squiggle_a4():
    while True:
        mouse_pos = pygame.mouse.get_pos()
        SCREEN.blit(BG_DARK, (0, 0))

        title = get_font(80).render("Choose Your Color Style ", True, "#900000")
        title_rect = title.get_rect(center=(640, 100))
        SCREEN.blit(title, title_rect)

        cmyk_btn = Button(image=IMG_PLAY_RECT, pos=(450, 370),
                          text_input="CMYK", font=get_font(60),
                          base_color="Black", hovering_color="#900000")
        grayscale_btn = Button(image=IMG_PLAY_RECT, pos=(830, 370),
                               text_input="GrayScale", font=get_font(60),
                               base_color="Black", hovering_color="#900000")
        back_btn = Button(image=None, pos=(640, 570),
                          text_input="BACK", font=get_font(75),
                          base_color="White", hovering_color="Green")

        for btn in [cmyk_btn, grayscale_btn, back_btn]:
            btn.changeColor(mouse_pos)
            btn.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if cmyk_btn.checkForInput(mouse_pos):
                    colored()
                elif grayscale_btn.checkForInput(mouse_pos):
                    not_colored_squiggle()
                elif back_btn.checkForInput(mouse_pos):
                    style()

        pygame.display.update()


def colored():
    cmyk_gcode = [
        "TestingImages/CMYK_Parts/cyan.gc",
        "TestingImages/CMYK_Parts/magenta.gc",
        "TestingImages/CMYK_Parts/yellow.gc",
        "TestingImages/CMYK_Parts/black.gc",
    ]
    cmyk_svg = [
        "TestingImages/CMYK_Parts/cyan.svg",
        "TestingImages/CMYK_Parts/magenta.svg",
        "TestingImages/CMYK_Parts/yellow.svg",
        "TestingImages/CMYK_Parts/black.svg",
    ]
    gcode_whites = [
        "TestingImages/CMYK_Parts/White/w1.gc",
        "TestingImages/CMYK_Parts/White/w2.gc",
        "TestingImages/CMYK_Parts/White/w3.gc",
        "TestingImages/CMYK_Parts/White/w4.gc",
    ]
    cmyk_names = ["cyan", "magenta", "yellow", "black"]

    while True:
        mouse_pos = pygame.mouse.get_pos()
        SCREEN.blit(BG_DARK, (0, 0))

        title = get_font(80).render("Processing...", True, "#900000")
        title_rect = title.get_rect(center=(640, 100))
        SCREEN.blit(title, title_rect)

        color_btn = Button(image=IMG_PLAY_RECT, pos=(640, 300),
                           text_input="Press Me", font=get_font(60),
                           base_color="Black", hovering_color="#900000")
        color_btn.changeColor(mouse_pos)
        color_btn.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if color_btn.checkForInput(mouse_pos):
                    cmyk = Cmyk_Converter.Cmyk("./TestingImages/image.png", 800)
                    cmyk_layers = cmyk.cmyk_paths()
                    for layer, gcode, name, svg, white in zip(
                        cmyk_layers, cmyk_gcode, cmyk_names, cmyk_svg, gcode_whites
                    ):
                        svg_obj = Svg_Converter.SvgConverter(layer, svg)
                        svg_out, svg_white = svg_obj.SvgToSin(90, 4, White_Removal=True)
                        gcode_conv = Gcode_Converter.GcodeConverter(
                            SvgPath=svg_out, GPath=gcode, scale=0.3
                        )
                        gcode_conv.firstConvert(SvgWhite=svg_white, GpathWhite=white)
                        time.sleep(0.2)
                        gcode_conv.secondConvertMulti(GpathWhite=white, gcode_paths=cmyk_gcode)
                        time.sleep(0.2)

                    try:
                        run_gcode(cmyk_gcode)
                    except Exception as e:
                        print(f"Error: {e}")
                        sys.exit(0)

        pygame.display.update()


def not_colored_squiggle():
    line_options = [
        (80,  True,  2.0),
        (100, True,  2.0),
        (120, False, 2.0),
        (140, False, 1.5),
    ]

    while True:
        mouse_pos = pygame.mouse.get_pos()
        SCREEN.blit(BG_DARK, (0, 0))

        title = get_font(80).render("How Many Lines ?", True, "#900000")
        title_rect = title.get_rect(center=(640, 100))
        SCREEN.blit(title, title_rect)

        buttons = [
            Button(image=IMG_PLAY_RECT, pos=(450, 300), text_input="80",
                   font=get_font(70), base_color="Black", hovering_color="#900000"),
            Button(image=IMG_PLAY_RECT, pos=(830, 300), text_input="100",
                   font=get_font(70), base_color="Black", hovering_color="#900000"),
            Button(image=IMG_PLAY_RECT, pos=(450, 460), text_input="120",
                   font=get_font(70), base_color="Black", hovering_color="#900000"),
            Button(image=IMG_PLAY_RECT, pos=(830, 460), text_input="140",
                   font=get_font(70), base_color="Black", hovering_color="#900000"),
        ]
        back_btn = Button(image=None, pos=(640, 570),
                          text_input="BACK", font=get_font(75),
                          base_color="White", hovering_color="Green")

        for btn in buttons + [back_btn]:
            btn.changeColor(mouse_pos)
            btn.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                for btn, (height, white_removal, max_amp) in zip(buttons, line_options):
                    if btn.checkForInput(mouse_pos):
                        process_squiggle(height, white_removal, max_amp)
                        try:
                            run_gcode()
                        except Exception as e:
                            print(e)
                        print(f"{height}-line squiggle done")
                        break
                else:
                    if back_btn.checkForInput(mouse_pos):
                        style()

        pygame.display.update()


def run_gcode(filepaths=None):
    """
    Unified G-code run screen.

    If *filepaths* is None, streams the single default file (Centered.gc).
    If *filepaths* is a list, streams all layers sequentially (CMYK mode).
    """
    started = False
    paused = False
    uploader = GRBLUploader(port="COM17")
    uploader.connect()

    while True:
        mouse_pos = pygame.mouse.get_pos()
        SCREEN.blit(BG_DARK, (0, 0))

        title = get_font(80).render("Running G-Code", True, "#900000")
        title_rect = title.get_rect(center=(640, 100))
        SCREEN.blit(title, title_rect)

        start_btn = Button(image=IMG_PLAY_RECT, pos=(640, 240),
                           text_input="START", font=get_font(55),
                           base_color="Black", hovering_color="#900000")
        resume_btn = Button(image=IMG_PLAY_RECT, pos=(640, 350),
                            text_input="RESUME", font=get_font(55),
                            base_color="Black", hovering_color="#900000")
        pause_btn = Button(image=IMG_PLAY_RECT, pos=(640, 460),
                           text_input="PAUSE", font=get_font(55),
                           base_color="Black", hovering_color="#900000")

        for btn in [start_btn, resume_btn, pause_btn]:
            btn.changeColor(mouse_pos)
            btn.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if pause_btn.checkForInput(mouse_pos) and not paused:
                    paused = True
                    uploader.Pause()
                    print("Paused")
                    time.sleep(1)
                elif resume_btn.checkForInput(mouse_pos) and paused:
                    paused = False
                    uploader.resume()
                    print("Resumed")
                    time.sleep(1)
                elif start_btn.checkForInput(mouse_pos) and not started:
                    print("Started")
                    started = True
                    if filepaths is not None:
                        uploader.streamLayers(filepaths)
                    else:
                        uploader.start_stream("TestingImages/Centered.gc")

        pygame.display.update()


main_menu()
