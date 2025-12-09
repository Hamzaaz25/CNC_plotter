import cv2
import subprocess
import time
import pygame, sys

import Cmyk_Converter
import Gcode_Converter
from Button import Button
from pygame import event
from tkinter import Tk, filedialog
import Svg_Converter
import threading
from grbl_uploader import GRBLUploader



def upload_image():
    root = Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png , *jpg , *jpeg")])
    if filepath:
        print(f"Selected image path {filepath}")
        img = cv2.imread(filepath)
        img = cv2.imwrite("./TestingImages/image.png", img)

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
        # cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("./TestingImages/image.png", img)
            cap.release()
            cv2.destroyAllWindows()
            break


pygame.init()

SCREEN = pygame.display.set_mode((1280, 720))
pygame.display.set_caption("Menu")

Bg = pygame.image.load("assets/MainBack.png")
BG = pygame.image.load("assets/Background.png")


def get_font(size):  # Returns Press-Start-2P in the desired size
    return pygame.font.Font("assets/font.ttf", size)


def draw():

        while True:
            DRAW_MOUSE_POS = pygame.mouse.get_pos()

            SCREEN.fill("black")

            PLAY_TEXT = get_font(80).render("Choose Your Way ", True, "#900000")
            PLAY_RECT = PLAY_TEXT.get_rect(center=(640, 100))
            SCREEN.blit(PLAY_TEXT, PLAY_RECT)

            Draw_Capture = Button(image=pygame.image.load("assets/Camera.png"), pos=(640, 300), text_input="CAPTURE  ",
                                  font=get_font(55), base_color="White", hovering_color="Red")
            Draw_UPLOAD = Button(image=pygame.image.load("assets/Upload.png"), pos=(640, 460),
                                 text_input=" UPLOAD FROM FILES   ", font=get_font(45), base_color="White",
                                 hovering_color="Red")
            DRAW_BACK = Button(image=None, pos=(640, 570),
                               text_input="BACK", font=get_font(75), base_color="White", hovering_color="Green")

            DRAW_BACK.changeColor(DRAW_MOUSE_POS)
            DRAW_BACK.update(SCREEN)
            Draw_Capture.changeColor(DRAW_MOUSE_POS)
            Draw_Capture.update(SCREEN)
            Draw_UPLOAD.changeColor(DRAW_MOUSE_POS)
            Draw_UPLOAD.update(SCREEN)

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if Draw_Capture.checkForInput(DRAW_MOUSE_POS):
                        take_picture()
                        print("Captured")
                        style()
                if event.type == pygame.MOUSEBUTTONUP:
                    if Draw_UPLOAD.checkForInput(DRAW_MOUSE_POS):
                        imagePath = upload_image()

                        print("UPLOAD FROM DEVICE")
                        style()

                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if DRAW_BACK.checkForInput(DRAW_MOUSE_POS):
                        main_menu()

            pygame.display.update()


def options():
    while True:
        OPTIONS_MOUSE_POS = pygame.mouse.get_pos()

        SCREEN.fill("white")

        OPTIONS_TEXT = get_font(45).render("Thank You for visiting OPTIONS screen.", True, "Black")
        OPTIONS_RECT = OPTIONS_TEXT.get_rect(center=(640, 260))
        SCREEN.blit(OPTIONS_TEXT, OPTIONS_RECT)

        OPTIONS_BACK = Button(image=None, pos=(640, 460),
                              text_input="BACK", font=get_font(75), base_color="Black", hovering_color="Green")

        OPTIONS_BACK.changeColor(OPTIONS_MOUSE_POS)
        OPTIONS_BACK.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if OPTIONS_BACK.checkForInput(OPTIONS_MOUSE_POS):
                    main_menu()

        pygame.display.update()


def main_menu():
    while True:
        SCREEN.blit(Bg, (0, 0))

        MENU_MOUSE_POS = pygame.mouse.get_pos()

        MENU_TEXT = get_font(100).render("CNC PLotter", True, "#900000")
        MENU_RECT = MENU_TEXT.get_rect(center=(640, 100))

        PLAY_BUTTON = Button(image=pygame.image.load("assets/Play Rect.png"), pos=(640, 250),
                             text_input="DRAW", font=get_font(75), base_color="White", hovering_color="#900000")
        OPTIONS_BUTTON = Button(image=pygame.image.load("assets/Options Rect.png"), pos=(640, 400),
                                text_input="OPTIONS", font=get_font(75), base_color="White", hovering_color="#900000")
        QUIT_BUTTON = Button(image=pygame.image.load("assets/Quit Rect.png"), pos=(640, 550),
                             text_input="QUIT", font=get_font(75), base_color="White", hovering_color="#900000")

        SCREEN.blit(MENU_TEXT, MENU_RECT)

        for button in [PLAY_BUTTON, OPTIONS_BUTTON, QUIT_BUTTON]:
            button.changeColor(MENU_MOUSE_POS)
            button.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if PLAY_BUTTON.checkForInput(MENU_MOUSE_POS):
                    draw()
                if OPTIONS_BUTTON.checkForInput(MENU_MOUSE_POS):
                    options()
                if QUIT_BUTTON.checkForInput(MENU_MOUSE_POS):
                    pygame.quit()
                    sys.exit()

        pygame.display.update()


def style():
    imagePath = "TestingImages/image.png"
    outputPath = "./TestingImages/Processed.svg"
    while True:
        Style_MOUSE_POS = pygame.mouse.get_pos()
        SCREEN.blit(BG, (0, 0))
        Style_Text = get_font(80).render("Choose Your Style", True, "#900000")
        Style_RECT = Style_Text.get_rect(center=(640, 100))
        SCREEN.blit(Style_Text, Style_RECT)
        Squiggle_Button = Button(image=pygame.image.load("assets/Play Rect.png"), pos=(410, 420),
                                 text_input="Squiggle ", font=get_font(55), base_color="Black",
                                 hovering_color="#900000")
        Scribble_Button = Button(image=pygame.image.load("assets/Play Rect.png"), pos=(850, 420),
                                 text_input="Scribble ", font=get_font(55), base_color="Black",
                                 hovering_color="#900000")
        Tsp_Button =Button(image=pygame.image.load("assets/Play Rect.png"), pos=(850, 260),text_input=" TSP(Beta)"
                           , font=get_font(55),base_color="Black",hovering_color="#900000")
        Vector_Button  = Button(image=pygame.image.load("assets/Play Rect.png"), pos=(410, 260),text_input="Vector"
                           , font=get_font(55),base_color="Black",hovering_color="#900000")
        Style_BACK = Button(image=None, pos=(640, 570),
                            text_input="BACK", font=get_font(75), base_color="White", hovering_color="Green")
        Vector_Button.changeColor(Style_MOUSE_POS)
        Vector_Button.update(SCREEN)
        Tsp_Button.changeColor(Style_MOUSE_POS)
        Tsp_Button.update(SCREEN)
        Style_BACK.changeColor(Style_MOUSE_POS)
        Style_BACK.update(SCREEN)
        Squiggle_Button.changeColor(Style_MOUSE_POS)
        Squiggle_Button.update(SCREEN)
        Scribble_Button.changeColor(Style_MOUSE_POS)
        Scribble_Button.update(SCREEN)
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if Vector_Button.checkForInput(Style_MOUSE_POS):
                    print("vector")
            if event.type == pygame.MOUSEBUTTONDOWN:
                if Tsp_Button.checkForInput(Style_MOUSE_POS):
                    Tsp = Svg_Converter.SvgConverter(imagePath=imagePath, outPath=outputPath)
                    Tsp.voronoi_tsp_pipeline(
                        image_path=imagePath,
                        output_dir="TestingImages",
                        num_points=10000,
                        edge_fraction=0.6,
                        relax_iters=12,
                        relax_alpha=0.8,
                        make_single_path=True
                    )
                    print("tsp")
                    Gcon = Gcode_Converter.GcodeConverter(SvgPath= outputPath,GPath="TestingImages/Centered.gc" , scale=0.2 ,bed_width=280 , bed_height=200 )
                    Gcon.gcodeConvert(svg=outputPath, gcode="TestingImages/Centered.gc")
                    Gcon.center_gcode("TestingImages/Centered.gc")
                    Run()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if Squiggle_Button.checkForInput(Style_MOUSE_POS):
                    print("Squiggle")
                    SquiggleA4()
                    Run()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if Scribble_Button.checkForInput(Style_MOUSE_POS):
                    print("Scribble")
                    Run()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if Style_BACK.checkForInput(Style_MOUSE_POS):
                    draw()
        pygame.display.update()


def SquiggleA4():

    while True:
        Squiggle_Mos_Pos = pygame.mouse.get_pos()
        SCREEN.blit(BG, (0, 0))
        Squiggle_Text = get_font(80).render("Choose Your Color Style ", True, "#900000")
        Squiggle_RECT = Squiggle_Text.get_rect(center=(640, 100))
        SCREEN.blit(Squiggle_Text, Squiggle_RECT)
        first_Button = Button(image=pygame.image.load("assets/Play Rect.png"), pos=(450, 370), text_input="CMYK",
                              font=get_font(60), base_color="Black", hovering_color="#900000")
        Second_Button = Button(image=pygame.image.load("assets/Play Rect.png"), pos=(830, 370), text_input="GrayScale",
                               font=get_font(60), base_color="Black", hovering_color="#900000")

        Style_BACK = Button(image=None, pos=(640, 570),
                            text_input="BACK", font=get_font(75), base_color="White", hovering_color="Green")

        Style_BACK.changeColor(Squiggle_Mos_Pos)
        Style_BACK.update(SCREEN)
        first_Button.changeColor(Squiggle_Mos_Pos)
        first_Button.update(SCREEN)
        Second_Button.changeColor(Squiggle_Mos_Pos)
        Second_Button.update(SCREEN)



        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if first_Button.checkForInput(Squiggle_Mos_Pos):
                    Colored()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if Second_Button.checkForInput(Squiggle_Mos_Pos):
                    NotcoloredSquiggle()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if Style_BACK.checkForInput(Squiggle_Mos_Pos):
                    style()
        pygame.display.update()

def Colored ():
    cmyk_gcode = ["TestingImages/CMYK_Parts/cyan.gc",
                  "TestingImages/CMYK_Parts/magenta.gc",
                  "TestingImages/CMYK_Parts/yellow.gc",
                  "TestingImages/CMYK_Parts/black.gc" ]
    cmyk_svg = ["TestingImages/CMYK_Parts/cyan.svg",
                "TestingImages/CMYK_Parts/magenta.svg",
                "TestingImages/CMYK_Parts/yellow.svg",
                "TestingImages/CMYK_Parts/black.svg"]
    gcode_whites = ["TestingImages/CMYK_Parts/White/w1.gc",
                "TestingImages/CMYK_Parts/White/w2.gc",
                "TestingImages/CMYK_Parts/White/w3.gc",
                "TestingImages/CMYK_Parts/White/w4.gc"]
    cmyk_names = ["cyan", "magenta", "yellow", "black"]
    while True:
            Colored_Mouse = pygame.mouse.get_pos()
            SCREEN.blit(BG, (0, 0))
            Colored_Text = get_font(80).render("Processing...", True, "#900000")
            Colored_RECT = Colored_Text.get_rect(center=(640, 100))
            SCREEN.blit(Colored_Text, Colored_RECT)
            Color_Button =Button(image=pygame.image.load("assets/Play Rect.png"), pos=(640, 300), text_input="Press Me",
                              font=get_font(60), base_color="Black", hovering_color="#900000")
            Color_Button.changeColor(Colored_Mouse)
            Color_Button.update(SCREEN)
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if Color_Button.checkForInput(Colored_Mouse):
                       cmyk = Cmyk_Converter.Cmyk("./TestingImages/image.png" , 800)
                       cmyk_layers = cmyk.cmyk_paths()
                       for layer, gcode, names , svg ,white in zip(cmyk_layers, cmyk_gcode, cmyk_names , cmyk_svg , gcode_whites):
                           SvgObject = Svg_Converter.SvgConverter(layer,svg)

                           svg, svgwhite = SvgObject.SvgToSin(90, 4, White_Removal=True)
                           Gcode = Gcode_Converter.GcodeConverter(SvgPath=svg , GPath=gcode , scale = 0.3)
                           Gcode.firstConvert(SvgWhite=svgwhite, GpathWhite=white)
                           time.sleep(0.2)
                           Gcode.secondConvert(GpathWhite=white)
                           time.sleep(0.2)



                       try:
                        Run2()
                       except Exception as e:
                           print("Error")
                           sys.exit(0)
            pygame.display.update()

def NotcoloredSquiggle():
    while True:
        Squiggle_Mos_Poss = pygame.mouse.get_pos()
        SCREEN.blit(BG, (0, 0))
        Squiggle_Text = get_font(80).render("How Many Lines ?", True, "#900000")
        Squiggle_RECT = Squiggle_Text.get_rect(center=(640, 100))
        SCREEN.blit(Squiggle_Text, Squiggle_RECT)
        first_Button = Button(image=pygame.image.load("assets/Play Rect.png"), pos=(450, 300), text_input="80",
                              font=get_font(70), base_color="Black", hovering_color="#900000")
        Second_Button = Button(image=pygame.image.load("assets/Play Rect.png"), pos=(830, 300), text_input="100",
                               font=get_font(70), base_color="Black", hovering_color="#900000")
        Third_Button = Button(image=pygame.image.load("assets/Play Rect.png"), pos=(450, 460), text_input="120",
                              font=get_font(70), base_color="Black",
                              hovering_color="#900000")
        Fourth_Button = Button(image=pygame.image.load("assets/Play Rect.png"), pos=(830, 460), text_input="140",
                               font=get_font(70), base_color="Black",
                               hovering_color="#900000")
        Style_BACK = Button(image=None, pos=(640, 570),
                            text_input="BACK", font=get_font(75), base_color="White", hovering_color="Green")

        Style_BACK.changeColor(Squiggle_Mos_Poss)
        Style_BACK.update(SCREEN)
        first_Button.changeColor(Squiggle_Mos_Poss)
        first_Button.update(SCREEN)
        Second_Button.changeColor(Squiggle_Mos_Poss)
        Second_Button.update(SCREEN)
        Third_Button.changeColor(Squiggle_Mos_Poss)
        Third_Button.update(SCREEN)
        Fourth_Button.changeColor(Squiggle_Mos_Poss)
        Fourth_Button.update(SCREEN)
        imagePath = "TestingImages/image.png"
        outPath = "./TestingImages/Processed.svg"
        GcodePath = "./TestingImages/Gcode.gc"
        GcodePathWhite = "./TestingImages/White.gc"

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if first_Button.checkForInput(Squiggle_Mos_Poss):
                    Svg80 = Svg_Converter.SvgConverter(imagePath, outPath)
                    svg, svgwhite = Svg80.SvgToSin(80, 4, White_Removal=True)
                    time.sleep(0.5)
                    print("hello")
                    Gcon = Gcode_Converter.GcodeConverter(SvgPath=svg, GPath=GcodePath, scale=0.3)
                    Gcon.firstConvert(SvgWhite=svgwhite, GpathWhite=GcodePathWhite)
                    time.sleep(1.5)
                    Gcon.secondConvert(GpathWhite=GcodePathWhite)
                    # Run()
                    print("First Button")
            if event.type == pygame.MOUSEBUTTONDOWN:
                if Second_Button.checkForInput(Squiggle_Mos_Poss):
                    Svg100 = Svg_Converter.SvgConverter(imagePath, outPath)
                    svg, svgwhite = Svg100.SvgToSin(100, 4, White_Removal=True)
                    time.sleep(0.5)
                    Gcon = Gcode_Converter.GcodeConverter(SvgPath=svg, GPath=GcodePath, scale=0.3, bed_width=297,
                                                          bed_height=297)
                    Gcon.firstConvert(SvgWhite=svgwhite, GpathWhite=GcodePathWhite)
                    time.sleep(1.5)
                    Gcon.secondConvert(GpathWhite=GcodePathWhite)
                    Run()
                    print("Second Button")
            if event.type == pygame.MOUSEBUTTONDOWN:
                if Third_Button.checkForInput(Squiggle_Mos_Poss):
                    Svg120 = Svg_Converter.SvgConverter(imagePath, outPath)
                    svg, svgwhite = Svg120.SvgToSin(120, 4, White_Removal=False)
                    time.sleep(0.5)
                    Gcon = Gcode_Converter.GcodeConverter(SvgPath=svg, GPath=GcodePath,
                                                          scale=0.3, bed_width=297,
                                                          bed_height=297)
                    Gcon.firstConvert(SvgWhite=svgwhite, GpathWhite=GcodePathWhite)
                    time.sleep(1.5)
                    Gcon.secondConvert(GpathWhite=GcodePathWhite)
                    time.sleep(0.5)
                    Run()
                    print("Third Button")
            if event.type == pygame.MOUSEBUTTONDOWN:
                if Fourth_Button.checkForInput(Squiggle_Mos_Poss):
                    Svg140 = Svg_Converter.SvgConverter(imagePath, outPath)
                    svg, svgwhite = Svg140.SvgToSin(140, 4, max_amplitude=1.5, White_Removal=False)
                    time.sleep(0.5)
                    Gcon = Gcode_Converter.GcodeConverter(SvgPath=svg, GPath=GcodePath,
                                                          scale=0.3, bed_width=297,
                                                          bed_height=297)
                    Gcon.firstConvert(SvgWhite=svgwhite, GpathWhite=GcodePathWhite)
                    time.sleep(1.5)
                    Gcon.secondConvert(GpathWhite=GcodePathWhite)
                    time.sleep(0.5)
                    try:
                        Run()
                    except Exception as e:
                        print(e)
                    print("Fourth Button")
            if event.type == pygame.MOUSEBUTTONDOWN:
                if Style_BACK.checkForInput(Squiggle_Mos_Poss):
                    style()
        pygame.display.update()

def Run():
    Started = False
    Paused = False
    uploader = GRBLUploader(port="COM17")
    uploader.connect()
    while True:
        Gcode_Mouse_Pos = pygame.mouse.get_pos()
        SCREEN.blit(BG, (0, 0))
        Gcode_Text = get_font(80).render("Running G-Code", True, "#900000")
        Gcode_RECT = Gcode_Text.get_rect(center=(640, 100))
        SCREEN.blit(Gcode_Text, Gcode_RECT)
        Start_Button = Button(image=pygame.image.load("assets/Play Rect.png"), pos=(640, 240), text_input="START",
                              font=get_font(55), base_color="Black", hovering_color="#900000")
        Pause_Button = Button(image=pygame.image.load("assets/Play Rect.png"), pos=(640, 460), text_input="PAUSE",
                              font=get_font(55), base_color="Black", hovering_color="#900000")
        Resume_Button = Button(image=pygame.image.load("assets/Play Rect.png"), pos=(640, 350), text_input="RESUME",
                               font=get_font(55), base_color="Black", hovering_color="#900000")
        Pause_Button.changeColor(Gcode_Mouse_Pos)
        Pause_Button.update(SCREEN)
        Resume_Button.changeColor(Gcode_Mouse_Pos)
        Resume_Button.update(SCREEN)
        Start_Button.changeColor(Gcode_Mouse_Pos)
        Start_Button.update(SCREEN)


        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if Pause_Button.checkForInput(Gcode_Mouse_Pos):
                    if not Paused:
                      Paused = True
                      uploader.Pause()
                      print("Paused")
                      time.sleep(1)


            if event.type == pygame.MOUSEBUTTONDOWN:
                if Resume_Button.checkForInput(Gcode_Mouse_Pos):
                    if Paused:
                        Paused = False
                        uploader.resume()
                        print("Resumed")
                        time.sleep(1)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if Start_Button.checkForInput(Gcode_Mouse_Pos):

                    if not Started:
                        print("Started")
                        Started = True
                        uploader.start_stream("TestingImages/Centered.gc")

        pygame.display.update()
def Run2():
    cmyk_gcode = ["TestingImages/CMYK_Parts/cyan.gc",
                  "TestingImages/CMYK_Parts/magenta.gc",
                  "TestingImages/CMYK_Parts/yellow.gc",
                  "TestingImages/CMYK_Parts/black.gc"]
    Started = False
    Paused = False
    uploader = GRBLUploader(port="COM17")
    uploader.connect()
    while True:
        Gcode_Mouse_Pos = pygame.mouse.get_pos()
        SCREEN.blit(BG, (0, 0))
        Gcode_Text = get_font(80).render("Running G-Code", True, "#900000")
        Gcode_RECT = Gcode_Text.get_rect(center=(640, 100))
        SCREEN.blit(Gcode_Text, Gcode_RECT)
        Start_Button = Button(image=pygame.image.load("assets/Play Rect.png"), pos=(640, 240), text_input="START",
                              font=get_font(55), base_color="Black", hovering_color="#900000")
        Pause_Button = Button(image=pygame.image.load("assets/Play Rect.png"), pos=(640, 460), text_input="PAUSE",
                              font=get_font(55), base_color="Black", hovering_color="#900000")
        Resume_Button = Button(image=pygame.image.load("assets/Play Rect.png"), pos=(640, 350), text_input="RESUME",
                               font=get_font(55), base_color="Black", hovering_color="#900000")
        Pause_Button.changeColor(Gcode_Mouse_Pos)
        Pause_Button.update(SCREEN)
        Resume_Button.changeColor(Gcode_Mouse_Pos)
        Resume_Button.update(SCREEN)
        Start_Button.changeColor(Gcode_Mouse_Pos)
        Start_Button.update(SCREEN)


        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if Pause_Button.checkForInput(Gcode_Mouse_Pos):
                    if not Paused:
                      Paused = True
                      uploader.Pause()
                      print("Paused")
                      time.sleep(1)


            if event.type == pygame.MOUSEBUTTONDOWN:
                if Resume_Button.checkForInput(Gcode_Mouse_Pos):
                    if Paused:
                        Paused = False
                        uploader.resume()
                        print("Resumed")
                        time.sleep(1)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if Start_Button.checkForInput(Gcode_Mouse_Pos):

                    if not Started:
                        print("Started")
                        Started = True
                        uploader.streamLayers(cmyk_gcode)


        pygame.display.update()


main_menu()




