import cv2
import numpy as np
from util import overlay_image_alpha


ICON = cv2.imread(r'src\icons\square.png')

class Square(object):
    icon = ICON[:]
    def __init__(self, x0, y0, size, padding, gap):
        self.pos = x0, y0
        self.end = x0+size*2, y0+size*2
        self.number_pos = x0+size//4, y0+size+gap
        self.img = Square.icon
        self.value = 0.0
    
    def coordinateInside(self, x, y):
        x0, y0 = self.pos
        xf, yf = self.end
        return x0 < x < xf and y0 < y < yf
    


class Grid(object):
    padding = None
    gap = None
    def __init__(self, start_lin, start_col, lins, cols, icon_size, kernel_matrix) -> None:
        self.legend = ''
        self.startlin = start_lin
        self.startcol = start_col
        self.endline = start_lin+lins
        self.endcol = start_col+cols
        self.square_size = Grid.gap+icon_size*2

        self.exit_icon_pos = Grid.padding+self.square_size*(self.endcol), Grid.padding+self.square_size*self.startlin-icon_size
        self.exit_icon_end = Grid.padding+self.square_size*(self.endcol)+icon_size, Grid.padding+self.square_size*self.startlin+icon_size

        self.legend_pos = Grid.padding+self.square_size*(self.startcol) + Grid.gap, Grid.padding+self.square_size*self.startlin-icon_size

        self.nodes = [[Square(Grid.padding+self.square_size*col, Grid.padding+self.square_size*lin, icon_size, Grid.padding, Grid.gap)
            for col in range(start_col, self.endcol)]
            for lin in range(start_lin, self.endline)]
        for kernel_value, node in zip(kernel_matrix.flatten(), self.getNodes()):
            node.value = kernel_value
    
    def getNodes(self):
        return [node for node_line in self.nodes for node in node_line]
    
    def getKernel(self):
        return np.array([[node.value for node in node_line] for node_line in self.nodes])

    def coordinateInsideExitIcon(self, x, y):
        x0, y0 = self.exit_icon_pos
        xf, yf = self.exit_icon_end
        return x0 < x < xf and y0 < y < yf

    def drawGrid(self, img):
        pd = Grid.padding
        xpos = {
            1: lambda x: x+pd*2,
            2: lambda x: x+(pd*3)//2,
            3: lambda x: x+pd//2
        }
        for node in self.getNodes():
            val_str = str(round(node.value,2))
            len_val = len(val_str.replace('.',''))
            x,y = node.number_pos
            x = x+pd//4 if len_val > 3 else xpos[len_val](x)
            overlay_image_alpha(img, node.img, *node.pos)
            cv2.putText(img, val_str,(x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,200,0), 2)

    def drawLegend(self, img):
        x,y = self.legend_pos
        cv2.putText(img, self.legend, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,200,0), 2)