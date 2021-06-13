# -*- coding: utf-8 -*-
"""
Dinamic Convolution
By: Renuá Meireles
"""

import cv2
import numpy as np

import mediapipe as mp
import numpy as np
from numpy.core.fromnumeric import shape
import hand_tracking as htm
from grid import Grid
import kernel_set
from util import overlay_image_alpha

ICON_SIZE = 32
PADDING = 10
GAP = 5
Grid.gap = GAP
Grid.padding = PADDING

HEIGHT = 720
WIDTH = 1280

DRAW_COLOR = (0,255,0)
THICKNESS = 2



def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    detector = htm.handDetector(maxHands=1)

    # *** ICONS ***
    settings_icon = cv2.imread(r'src\icons\ajustes.png')
    settings_icon_x, settings_icon_y = PADDING+(GAP+ICON_SIZE*2)*9, PADDING
    sets_icon = cv2.imread(r'src\icons\sets.png')
    sets_icon_x, sets_icon_y = PADDING+(GAP+ICON_SIZE*2)*10, PADDING
    exit_icon = cv2.imread(r'src\icons\exit.png')

    option = None

    # *** Kernels ***
    set_of_kernels = [
        Grid(2, col, 3, 3, ICON_SIZE, kernel) for col, kernel in enumerate(kernel_set.all_kernels, start=1)
    ]
    select_kernel_grid = Grid(1, 1, 1, len(kernel_set.all_kernels), ICON_SIZE, np.arange(1,7))
    custom_kernel_grid = Grid(1, 3, 3, 3, ICON_SIZE, kernel_matrix= kernel_set.identity)
    
    kernel = custom_kernel_grid
    custom_kernel = True

    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)

        # Find Hand Landmarks
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        img = cv2.filter2D(img, -1, kernel.getKernel())
        
        if len(lmList) != 0:
            # tip of index and little fingers
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[20][1:]

            # Check which fingers are up
            fingers, hand = detector.fingersUp()

            if fingers[1:] == [1,0,0,0]:
                cv2.circle(img, (x1,y1), 5, (0, 255, 0), cv2.FILLED)
                if y1 <= PADDING+ICON_SIZE:
                    if settings_icon_x <= x1 <= settings_icon_x+ICON_SIZE:
                        option = 1
                        kernel = custom_kernel_grid
                        custom_kernel = True
                    if sets_icon_x <= x1 <= sets_icon_x+ICON_SIZE:
                        option = 2

                # Fechando as configurações
                if option == 1 and custom_kernel_grid.coordinateInsideExitIcon(x1, y1):
                    option = None
                elif option == 2 and select_kernel_grid.coordinateInsideExitIcon(x1, y1):
                    option = None

            # Edição incremento do Kernel personalizado        
            if option == 1 and hand == 0 and fingers[1:] == [1,0,0,0]:
                cv2.circle(img, (x1,y1), 5, (255, 255, 255), cv2.FILLED)
                for node in custom_kernel_grid.getNodes():
                    if node.coordinateInside(x1, y1):
                        cv2.circle(img, (x1, y1), 10, (255, 50, 255), cv2.FILLED)
                        node.value -= .01
                        kernel = custom_kernel_grid
                        custom_kernel = True
                        break
            
            # Edição decremento do Kernel personalizado        
            if option == 1 and hand == 0 and fingers[1:] == [0,0,0,1]:
                cv2.circle(img, (x2,y2), 5, (255, 255, 255), cv2.FILLED)
                for node in custom_kernel_grid.getNodes():
                    if node.coordinateInside(x2, y2):
                        cv2.circle(img, (x2, y2), 10, (255, 50, 255), cv2.FILLED)
                        node.value += .01
                        kernel = custom_kernel_grid
                        custom_kernel = True
                        break
            
            # Seleção de um Kernel do conjunto de kernels
            if option == 2:
                for index, node in enumerate(select_kernel_grid.getNodes()):
                    if node.coordinateInside(x1, y1):
                        kernel = set_of_kernels[index]
                        select_kernel_grid.legend = kernel_set.names[index]
                        set_of_kernels[index]
                        custom_kernel = False
                        break

            
        # *** Adicionando elementos na tela ***
        if option == 2 and custom_kernel == False:
            kernel.drawGrid(img)
            select_kernel_grid.drawLegend(img)

        cv2.circle(img, (settings_icon_x+ICON_SIZE//2, settings_icon_y+ICON_SIZE//2), 20, (237, 237, 237), cv2.FILLED)
        overlay_image_alpha(img, settings_icon, settings_icon_x, settings_icon_y)
        cv2.circle(img, (sets_icon_x+ICON_SIZE//2, sets_icon_y+ICON_SIZE//2), 20, (237, 237, 237), cv2.FILLED)
        overlay_image_alpha(img, sets_icon, sets_icon_x, sets_icon_y)
        if option == 1:
            _x, _y = custom_kernel_grid.exit_icon_pos
            cv2.circle(img, (_x+ICON_SIZE//2, _y+ICON_SIZE//2), 15, (90, 100, 210), cv2.FILLED)
            overlay_image_alpha(img, exit_icon, *custom_kernel_grid.exit_icon_pos)
            cv2.line(img, (settings_icon_x+GAP, settings_icon_y+ICON_SIZE+GAP*2), (settings_icon_x+ICON_SIZE-GAP, settings_icon_y+ICON_SIZE+GAP*2), DRAW_COLOR, THICKNESS)
            custom_kernel_grid.drawGrid(img)

        elif option == 2:
            _x, _y = select_kernel_grid.exit_icon_pos
            cv2.circle(img, (_x+ICON_SIZE//2, _y+ICON_SIZE//2), 15, (90, 100, 210), cv2.FILLED)
            overlay_image_alpha(img, exit_icon, *select_kernel_grid.exit_icon_pos)
            cv2.line(img, (sets_icon_x+GAP, sets_icon_y+ICON_SIZE+GAP*2), (sets_icon_x+ICON_SIZE-GAP, sets_icon_y+ICON_SIZE+GAP*2), DRAW_COLOR, THICKNESS)
            select_kernel_grid.drawGrid(img)

        cv2.imshow("Video", img)

        key = cv2.waitKey(1)
        if key == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()