import numpy as np

sharpen = np.array([
    [-1., -1., -1.],
    [-1.,  9., -1.],
    [-1., -1., -1.]
])

sepia = np.array([
    [0.272, 0.534, 0.131],
    [0.349, 0.686, 0.168],
    [0.393, 0.769, 0.189]
])

gaussianBlur = np.array([
    [1., 2., 1.],
    [2., 4., 2.],
    [1., 2., 1.]
])/16

emboss = np.array([
    [0.,-1.,-1.],
    [1., 0.,-1.],
    [1., 1., 0.]
])

identity = np.array([
    [0., 0., 0.],
    [0., 1., 0.],
    [0., 0., 0.]
])

edge_detection = np.array([
    [-1.,-1., -1.],
    [-1., 8., -1.],
    [-1.,-1., -1.]
])

all_kernels = [identity, sharpen, gaussianBlur, sepia, emboss, edge_detection]
names = ['Identity', 'Sharpen', 'GaussianBlur', 'Sepia', 'Emboss', 'Edge Detection']