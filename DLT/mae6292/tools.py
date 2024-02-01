import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.linalg as linalg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class WireFrameCamera:
    def __init__(self, width=1, height=0.8, f=0.5):
        self.width = width
        self.height = height
        self.f = f

        self.vertices=np.empty((3,5))
        self.vertices[:,0]=[0, 0, 0]
        self.vertices[:,1]=[-width/2, height/2, f]
        self.vertices[:,2]=[width/2, height/2, f]
        self.vertices[:,3]=[width/2, -height/2, f]
        self.vertices[:,4]=[-width/2, -height/2, f]

        self.edges=[(0,1), (0,2), (0,3), (0,4), (1,2), (2,3), (3,4), (4,1)]

    def draw(self, R, T):
        vertices_W = np.empty((3,5))
        vertices_W = R.T@(self.vertices - T@np.ones((1,5)))

        fig = plt.figure(dpi=150)
        # ax = fig.gca(projection='3d') depreciated 
        ax = fig.add_subplot(projection='3d')

        for edge in self.edges:
            plt.plot(vertices_W[0,list(edge)],vertices_W[1,list(edge)],vertices_W[2,list(edge)],'b',linewidth=0.5)

        return fig, ax

class ChessBoard:
    faces=[(0,1,2,3)]
    square_vertices = np.zeros((3, 4))
    square_vertices[:,0] = [0,0,0]
    square_vertices[:,1] = [1,0,0]
    square_vertices[:,2] = [1,1,0]
    square_vertices[:,3] = [0,1,0]

    def __init__(self, CHECKER=(8,6), origin=(0.0)):
        self.origin = origin
        self.CHECKER = CHECKER

    def draw(self, ax):
        for i in range(self.CHECKER[0]+1):
            for j in range(self.CHECKER[1]+1):
                square_vertices_shifted = self.square_vertices + np.array([[i+self.origin[0],j+self.origin[1],0]]).T*np.ones((1,4))

                patch = [[tuple(square_vertices_shifted[:,self.faces[0][i]]) for i in range(len(self.faces[0]))]]
                patch_collection = Poly3DCollection(patch, alpha =0.5)
                if (i+j) % 2 == 1:
                    patch_collection.set_color('w')
                else:
                    patch_collection.set_color('k')

                    ax.add_collection3d(patch_collection)

def calibrateCamera_Tsai(p, P):
    # p : homogeneous coordinates of pixel in the image frame
    # P : homogeneous coordinates of points in the world frame
    assert p.shape[0] == 3, "p : homogeneous coordinates of pixel in the image frame of 3 by n"
    assert P.shape[0] == 4, "P : homogeneous coordinates of points in the world frame"
    assert p.shape[1] == P.shape[1], "number of columns of p shold match with P"

    n = p.shape[1]
    p_uv = p[0:2,:]/p[2,:]

    Q = np.empty((0, 12))
    for i in range(n):
        Qi_0 = np.array([ [1,0, -p_uv[0,i]], [0, 1, -p_uv[1,i]]] )
        Qi = np.kron(Qi_0, P[:,i].T)
        Q = np.append(Q, Qi, axis=0)

    # 1. Find M_tilde using SVD

    U, S, VT = linalg.svd(Q)
    M_tilde =VT.T[:,-1].reshape(3,4)
    # print(M_tilde /M_cv) # M is determined up to scale

    # 2. RQ factorization to find K_tilde and R

    K_tilde, R = linalg.rq(M_tilde [:,0:3])

    # 3. Resolve the ambiguity of RQ factorization
    D = np.diag( np.sign(np.diag(K_tilde)) )
    K_tilde  = K_tilde@D
    R = D@R

    # 4. Find T
    T = linalg.solve(K_tilde, M_tilde[:,-1]).reshape(3,1)

    # 5. Recover scale

    s = 1/K_tilde[2,2]
    K = s*K_tilde
    M = s*M_tilde

    # 6. Resolve sign ambiguity
    if linalg.det(R) < 0:
        R = -R
        T = -T
        M = -M

    return K, R, T, M
