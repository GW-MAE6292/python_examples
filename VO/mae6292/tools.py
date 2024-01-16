import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.linalg as linalg
import scipy.signal
import scipy.spatial.distance
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class WireFrameCamera:
    def __init__(self, width=1, height=0.8, f=0.5):
        self.width = width
        self.height = height
        self.f = f

        self.vertices = np.empty((3,5))
        self.vertices[:,0] = [0, 0, 0]
        self.vertices[:,1] = [-width/2, height/2, f]
        self.vertices[:,2] = [width/2, height/2, f]
        self.vertices[:,3] = [width/2, -height/2, f]
        self.vertices[:,4] = [-width/2, -height/2, f]

        self.edges=[(0,1), (0,2), (0,3), (0,4), (1,2), (2,3), (3,4), (4,1)]

    def draw(self, R, T):
        vertices_W = np.empty((3,5))
        vertices_W = R.T@(self.vertices - T@np.ones((1,5)))

        fig = plt.figure(dpi=150)
        ax = fig.gca(projection='3d')

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
    # DLT using Tsai's method
    # INPUT
    # p : homogeneous coordinates of pixel in the image frame
    # P : homogeneous coordinates of points in the world frame
    # OUTPUT
    # K, R, T, M
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
    M_tilde = VT.T[:,-1].reshape(3,4)
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


def convolve2d(img, filter, padding_type = 'zero'):
# compute convolution  img * filter
# the filtered image is float valued, and it is not necessarily cv2 gray scale image of 'uint8'
    assert (filter.shape[0]-1) % 2 == 0, 'filter size should be odd'
    assert padding_type in ['zero', 'replicate', 'reflect'], 'padding type parameter should be either zero, clamp, or mirror'
    n_rows, n_cols = img.shape
    img_filtered = np.zeros((n_rows, n_cols))
    W = int((filter.shape[0]-1)/2)

    if padding_type == 'zero':
        img_padded = cv2.copyMakeBorder(img, W, W, W, W, cv2.BORDER_CONSTANT, 0)
    elif padding_type == 'replicate': # aaaaaa|abcdefgh|hhhhhhh
        img_padded = cv2.copyMakeBorder(img, W, W, W, W, cv2.BORDER_REPLICATE)
    elif padding_type == 'reflect': # gfedcb|abcdefgh|gfedcba
        img_padded = cv2.copyMakeBorder(img, W, W, W, W, cv2.BORDER_REFLECT_101)

    for i in range(n_rows):
        for j in range(n_cols):
            img_clip = img_padded[i:i+2*W+1, j:j+2*W+1] # extract the image clip centered at (i,j)
            img_clip.astype('float64') # convert to float to avoid error of int*float
            img_filtered[i,j]=(img_clip*np.flip(filter)).sum()

    return img_filtered

def filter2d(filter_type, W, sigma):
    assert filter_type in ['gaussian', 'log', 'dgau_x', 'dgau_y'], 'filter type parameter should be either gaussian, log, dgau_x, dgau_y'
    filter = np.zeros((2*W+1, 2*W+1))
    x = np.arange(-W,W+1).reshape(2*W+1,1)
    if filter_type == 'gaussian':
        filter = np.exp(-x**2/2/sigma**2)
        filter = filter*filter.T
        filter = filter/filter.sum()
    elif filter_type == 'log': # laplacian of gaussian
        filter = np.exp(-x**2/2/sigma**2)
        filter = filter*filter.T
        filter = filter/filter.sum()
        for i in range(-W,W+1):
            for j in range(-W, W+1):
                filter[i+W,j+W] = filter[i+W,j+W]*(i**2+j**2-2*sigma**2)/sigma**4
        filter = filter - filter.sum()/((2*W+1)**2) # make the filter sum zero
    elif filter_type == 'dgau_x': # derivative of gaussian w.r.t x
        filter = np.exp(-x**2/2/sigma**2)
        filter = filter*filter.T
        filter = filter/filter.sum()
        filter = -filter*(np.ones((2*W+1,1))@x.T)/sigma**2
        filter = filter - filter.sum()/((2*W+1)**2) # make the filter sum zero
    elif filter_type == 'dgau_y': # derivative of gaussian w.r.t y
        filter = np.exp(-x**2/2/sigma**2)
        filter = filter*filter.T
        filter = filter/filter.sum()
        filter = -filter*(x@np.ones((1,2*W+1)))/sigma**2
        filter = filter - filter.sum()/((2*W+1)**2) # make the filter sum zero=
    return filter

def harris_corner(img, W, kappa):
    # Compute harris corner score
    # INPUT
    #   img: gray scale image
    #   W: harris patch size (patch width= 2*W+1)
    #   kappa: kappa in the harris score
    # OUTPUT
    #   harris_score: array of the same size as the input image
    S_1 = np.array([1, 2, 1], dtype='float32')
    S_2 = np.array([-1, 0, 1], dtype='float32')

    I_x = scipy.signal.sepfir2d(img, S_2, S_1)
    I_y = scipy.signal.sepfir2d(img, S_1, S_2)

    I_x2 = I_x**2
    I_y2 = I_y**2
    I_xy = I_x*I_y

    filter_harris_sum = np.ones(2*W+1, dtype='float32')
    M_x2 = scipy.signal.sepfir2d(I_x2, filter_harris_sum, filter_harris_sum)
    M_y2 = scipy.signal.sepfir2d(I_y2, filter_harris_sum, filter_harris_sum)
    M_xy = scipy.signal.sepfir2d(I_xy, filter_harris_sum, filter_harris_sum)

    harris_score = (M_x2*M_y2 - M_xy**2) - kappa*(M_x2+M_y2)**2
    harris_score[0:W+1,:]=0
    harris_score[:,0:W+1]=0
    harris_score[:,-(W+1):]=0
    harris_score[-(W+1):,:]=0

    harris_score[harris_score<0]=0

    return harris_score


def select_keypoints(score, N_keypoint, W_nms):
    # Select keypoints from Harris score
    # INPUT
    #   score: array of Harris score
    #   N_keypoints: number of keypoints to be detected
    #   W_nms: patch size for non maximum supression
    # OUTPUT
    #   keypoints: list of (u,v) for keypoints

    n_rows, n_cols = score.shape
    keypoints = []
    for i in range(N_keypoint):
        u_max, v_max = np.unravel_index(score.argmax(), score.shape)
        keypoints.append((u_max,v_max))
        score[ max(u_max-W_nms,0): min(u_max+W_nms+1,n_rows), max(v_max-W_nms,0): min(v_max+W_nms+1,n_cols)]=0

    return keypoints


def describe_keypoints(img, keypoints, W):
    # Extract descriptors as patches centered at keypoints 
    # INPUT
    #   img: image
    #   keypoints: list of (u,v) for keypoints
    #   W: descriptor patch size
    # OUTPUT
    # descriptor: array of len(keypoints) by (2*W+1)**2 
    descriptors = np.zeros( ((2*W+1)**2, len(keypoints) ))
    img_padded = cv2.copyMakeBorder(img, W, W, W, W, cv2.BORDER_CONSTANT, 0)

    for i, keypoint in enumerate(keypoints):
        u, v = keypoint
        descriptors[:,i]= img_padded[u:u+2*W+1, v:v+2*W+1].reshape((2*W+1)**2,).astype('uint8')

    return descriptors


def match_descriptors(descriptors_new, descriptors_old, lambda_match):
    # Match descriptors
    # INPUT
    # descriptors_new (query): array of N_descriptor_new by (2*W+1)**2 
    # descriptors_old (database): array of N_descriptor_old by (2*W+1)**2 
    # lambda_match: match if distance >= lambda_match * min_nonzero_distance
    # OUTPUT
    # unique_match: 
    # if unique_match >= 0:
    #   descriptors_new[:,i] is closest to descriptors_old[:,match[i]]
    # else
    #   descriptors_new[:,i] is not matched to any
    # distance: distance 
    assert descriptors_new.shape[0] == descriptors_old.shape[0], 'descriptor size is different'
    cdist = scipy.spatial.distance.cdist(descriptors_new.T, descriptors_old.T, 'euclidean')

    # match[i] = index of descriptors that is closest to descriptors1[:,i], or
    # descriptors1[:,i] is closest to descriptors[:,match[i]]
    match = np.argmin(cdist, axis=1)
    # the corresopnding min distance between descriptors1[:,i] and descriptors[:,match[i]]
    distance = cdist[np.arange(cdist.shape[0]),match]

    # minimum nonzero distance
    min_nonzero_distance = np.min(distance[np.nonzero(distance)])
    # if the distance is greater than the threshhold, then remove the match
    # unmatched descriptor is defined by negative index
    match[distance >= lambda_match * min_nonzero_distance] = -1

    # remove any repeated match
    _, unique_indices = np.unique(match, return_index=True)
    unique_match=np.ones(match.shape,dtype='int64')*-1
    unique_match[unique_indices] = match[unique_indices]

    return unique_match, distance

def compute_disparity(img_left, img_right, W_patch, d_min, d_max):

    N_rows, N_cols = img_left.shape
    disparity = np.zeros((N_rows,N_cols))

    for i_row in range(W_patch, N_rows-W_patch):
        for i_col in range(W_patch+d_max, N_cols-W_patch):


            patch_left = img_left[i_row-W_patch:i_row+W_patch+1, i_col-W_patch:i_col+W_patch+1]
            strip_right = img_right[i_row-W_patch:i_row+W_patch+1, i_col-W_patch-d_max:i_col+W_patch-d_min+1]

            patch_left = np.reshape( patch_left, ((2*W_patch+1)**2,1), order='F')
            patches_right = np.zeros( ((2*W_patch+1)**2,  d_max-d_min+1) )

            for i in range(2*W_patch+1):
                patches_right[i*(2*W_patch+1):(i+1)*(2*W_patch+1), :] = strip_right[:, i:(i+d_max-d_min+1)]

            SSD = scipy.spatial.distance.cdist(patch_left.T, patches_right.T, 'sqeuclidean').flatten()
            i_match = np.argmin(SSD)
            SSD_min = SSD[i_match]


            if SSD[SSD <= SSD_min*1.5].shape[0] < 3 and i_match != 0 and i_match != SSD.shape[0]-1:
                i_match_3 = [i_match-1, i_match, i_match+1]
                p = np.polyfit( i_match_3, SSD[i_match_3], 2)
                i_match = -p[1]/2/p[0]
                disparity[i_row, i_col] = d_max-i_match

    return disparity

def hat(s):
    S=np.zeros((3,3))
    S[1,2]=-s[0]
    S[2,1]=s[0]
    S[0,2]=s[1]
    S[2,0]=-s[1]
    S[0,1]=-s[2]
    S[1,0]=s[2]

    return S

def disparity2PC(disparity, K, baseline,img_left):
    T = np.array([-baseline, 0, 0])
    M1 = np.zeros((3,4))
    M1[0:3,0:3] = K
    M2 = np.zeros((3,4))
    M2[0:3,0:3] = K
    M2[:,3] = K@T

    N_points = disparity[disparity>0].shape[0]
    P = np.zeros((3,N_points))

    i=0
    for v1, u1 in zip(*np.where(disparity>0)):
        d = disparity[v1,u1]
        u2 = u1 - d
        v2 = v1
        Q = np.zeros((6,4))
        Q[0:3, :] = hat([u1, v1, 1])@M1
        Q[3:6, :] = hat([u2, v2, 1])@M2

        Pi_h = scipy.linalg.null_space(Q).flatten()


        Pi = Pi_h[0:3]/Pi_h[3]
        Pi = Pi*np.sign(Pi[2])

        P[:,i]=Pi
        i=i+1

    return P, img_left[disparity>0].reshape((1,-1))


def triangulation(p1, p2, M1, M2):
    # Input
    # p1, p2: 3 by n pixel homogeneous coordinates
    # M1, M2, 3 by 4 perspective projection matrix
    # Output
    # P: 3 by n (nonhomogeneous) world coordinate
    assert(p1.shape[0]==3 and p2.shape[0]==3 and p1.shape[1]==p2.shape[1] )

    n=p1.shape[1]
    P = np.zeros((3,n))
    for i in range(n):
        Q = np.concatenate(( hat(p1[:,i])@M1, hat(p2[:,i])@M2), axis=0)
        U, S, VT = scipy.linalg.svd(Q)
        Pi = VT[-1,:]
        P[:,i] = Pi[0:3]/Pi[3]

    return P

def triangulation_robust(p1, p2, M1, M2, tol_mu=1e-3, tol_rep = 1):
    # Input
    # p1, p2: 3 by n pixel homogeneous coordinates
    # M1, M2: 3 by 4 perspective projection matrix
    # tol_mu: tolerance for the ratio of the smallest singular value to the next smallest
    # tol_rep: tolerance for the reprojection error
    # Output
    # P: 3 by n (nonhomogeneous) world coordinate
    assert(p1.shape[0]==3 and p2.shape[0]==3 and p1.shape[1]==p2.shape[1] )

    n=p1.shape[1]
    p_W = np.zeros((3,n))
        
    mu = np.zeros((4,n))
    for i in range(n):
        Q = np.concatenate(( hat(p1[:,i])@M1, hat(p2[:,i])@M2), axis=0)
        U, S, VT = scipy.linalg.svd(Q)
        mu[:,i] = S
        Pi = VT[-1,:]
        p_W[:,i] = Pi[0:3]/Pi[3]

    mu_ratio = mu[3,:]/mu[2,:]
    err1 = np.sqrt(np.sum((deh( M1 @ hom(p_W)) - deh(p1))**2, axis=0)) 
    err2 = np.sqrt(np.sum((deh( M2 @ hom(p_W)) - deh(p2))**2, axis=0))
    err = np.max((err1,err2), axis=0)

    index_inliers = np.where( ((mu_ratio < tol_mu) & (err < tol_rep)) )[0]

    # it turned out that the angle is not a good indicator for negative depths
    # M1 = K1 @ np.concatenate( (R1, T1), axis=1 )
    # M2 = K2 @ np.concatenate( (R2, T2), axis=1 )   
    # CP1 = R1.T @ np.linalg.inv(K1) @ p1
    # CP2 = R2.T @ np.linalg.inv(K2) @ p2
    # CP1_n = CP1/np.sqrt(np.sum(CP1**2, axis=0))
    # CP2_n = CP2/np.sqrt(np.sum(CP2**2, axis=0))
    # angle = np.arccos(np.sum(CP1_n*CP2_n, axis=0))*180/np.pi

    return p_W, index_inliers, mu_ratio, err

def triangulation_robust_depth(p1, p2, K1, R1, T1, K2, R2, T2, tol_mu=1e-3, tol_rep = 1, tol_depth = (1, 50)):
    # Input
    # p1, p2: 3 by n pixel homogeneous coordinates
    # M1, M2: 3 by 4 perspective projection matrix
    # tol_mu: tolerance for the ratio of the smallest singular value to the next smallest
    # tol_rep: tolerance for the reprojection error
    # tol_depth: tol_depth[0]=min_depth, tol_depth[1]=max_depth
    # Output
    # P: 3 by n (nonhomogeneous) world coordinate
    assert(p1.shape[0]==3 and p2.shape[0]==3 and p1.shape[1]==p2.shape[1] )

    M1 = K1 @ np.concatenate((R1, T1), axis = 1 )
    M2 = K2 @ np.concatenate((R2, T2), axis = 1 )

    n=p1.shape[1]
    p_W = np.zeros((3,n))
        
    mu = np.zeros((4,n))
    for i in range(n):
        Q = np.concatenate(( hat(p1[:,i])@M1, hat(p2[:,i])@M2), axis=0)
        U, S, VT = scipy.linalg.svd(Q)
        mu[:,i] = S
        Pi = VT[-1,:]
        p_W[:,i] = Pi[0:3]/Pi[3]

    mu_ratio = mu[3,:]/mu[2,:]
    err1 = np.sqrt(np.sum((deh( M1 @ hom(p_W)) - deh(p1))**2, axis=0)) 
    err2 = np.sqrt(np.sum((deh( M2 @ hom(p_W)) - deh(p2))**2, axis=0))
    err = np.max((err1,err2), axis=0)
    
    P_2W = R2 @ p_W +T2 # vector from C2 to p_W resolved in C2 frame
    depth = P_2W[2,:] # depth of p_W in C2 frame
    
    index_inliers = np.where( ( (mu_ratio < tol_mu) & (err < tol_rep) & (tol_depth[0] < depth) & (depth < tol_depth[1]) ) )[0]

    return p_W, index_inliers, mu_ratio, err, depth


def essentialmatrix2RT(E, p1, p2, K1, K2):
    # E: 3 by 3 essential essential matrix
    # p1, p2: 3 by n pixel homogeneous coordinates
    # K1, K2: 3 by 3 intrinsic parameters
    U, S, VT = scipy.linalg.svd(E)

    R = np.zeros((3,3,4))
    T = np.zeros((3,4))
    W = np.array( [[0,1,0], [-1,0,0], [0,0,1]])

    R[:,:,0], T[:,0] = U@W@VT, U[:,2]
    R[:,:,1], T[:,1] = U@W@VT, -U[:,2]
    R[:,:,2], T[:,2] = U@W.T@VT, U[:,2]
    R[:,:,3], T[:,3] = U@W.T@VT, -U[:,2]

    for i in range(4):
        if scipy.linalg.det(R[:,:,i]) < 0:
            R[:,:,i] = -R[:,:,i]

    M1 = K1@np.concatenate( (np.identity(3), np.zeros((3,1))), axis=1 )

    N_negative_depth = np.zeros(4, dtype='int64')
    for i in range(4):
        M2 = K2@np.concatenate( (R[:,:,i], T[:,i].reshape(-1,1)), axis=1 )
        P = triangulation( p1, p2, M1, M2)
        N_negative_depth[i] = len(np.where(P[2,:]<0)[0])

    if np.where(N_negative_depth == N_negative_depth.min())[0].shape[0] > 1:
        print("Warning: mae6292.essentialmatrix2RT: non-unique solution ")

    i_opt = np.argmin(N_negative_depth)

    R=R[:,:,i_opt]
    T=T[:,i_opt].reshape(-1,1)

    return R, T

def eightpoint_algorithm(p1, p2, K1, K2):
    # p1, p2: 3 by n pixel homogeneous coordinates
    # K1, K2: 3 by 3 intrinsic parameters
    n = p1.shape[1]
    p_bar_1 = scipy.linalg.inv(K1)@p1
    p_bar_2 = scipy.linalg.inv(K2)@p2

    Q = np.zeros((n,9))
    for i in range(n):
        Q[i,:] = np.kron(p_bar_1[:,i], p_bar_2[:,i]).T

    U, S, VT = scipy.linalg.svd(Q)

    E = VT[-1,:]
    E = E.reshape(3,3).T

    R, T  = essentialmatrix2RT(E, p1, p2, K1, K2)

    return E, R, T

def estimate_pose_DLT(p, P, K):
    # DLT using Tsai's method assuming that K is known
    # INPUT
    # p : homogeneous coordinates of pixel in the image frame
    # P : homogeneous coordinates of points in the world frame
    # OUTPUT
    # R, T, R_T=[R|T]
    assert p.shape[0] == 3, "p : homogeneous coordinates of pixel in the image frame of 3 by n"
    assert P.shape[0] == 4, "P : homogeneous coordinates of points in the world frame"
    assert p.shape[1] == P.shape[1], "number of columns of p shold match with P"

    n = p.shape[1]
    p = scipy.linalg.inv(K)@p
    p_uv = p[0:2,:]/p[2,:]

    Q = np.empty((0, 12))
    for i in range(n):
        Qi_0 = np.array([ [1,0, -p_uv[0,i]], [0, 1, -p_uv[1,i]]] )
        Qi = np.kron(Qi_0, P[:,i].T)
        Q = np.append(Q, Qi, axis=0)

    # 1. Find M_tilde using SVD

    U, S, VT = linalg.svd(Q)
    M_tilde = VT.T[:,-1].reshape(3,4)
    
    if linalg.det(M_tilde[:,0:3]) < 0:
        M_tilde = -M_tilde
    
    U, S, VT = linalg.svd(M_tilde[:,0:3])
    R = U@VT

    s = np.sqrt(np.sum(R**2)) / np.sqrt(np.sum(M_tilde[:,0:3]**2))

    T = s*M_tilde[:,3].reshape(3,1)

    R_T = np.concatenate((R,T), axis =1 )
    return R, T, R_T


def estimate_pose_RANSAC_DLT(p_matched, p_W_matched, K, N_iter=1000, tol_inlier=10, display_iter=True):
    N_inliers_max = 0
    N_data = p_matched.shape[1]
    for i_iter in range(N_iter):

        # sample 6 points from the matched keypoints and estimate pose using them
        i_sample = np.random.choice(N_data, 6, replace=False)
        R, T, R_T = estimate_pose_DLT(p_matched[:, i_sample], p_W_matched[:,i_sample], K)

        # compute the reprojection error
        p_matched_est = K @ R_T @ p_W_matched
        error = np.sqrt(np.sum((p_matched_est[0:2,:]/p_matched_est[2,:]-p_matched[0:2,:]/p_matched[2,:])**2, axis=0))
        
        # identify inliers
        i_inliers = np.where(error < tol_inlier)[0]

        # if the number of inliers is the best, save the list of inliers
        if len(i_inliers) > N_inliers_max:
            i_inliers_best = i_inliers
            N_inliers_max = len(i_inliers)
            R_best = R
            T_best = T
            if display_iter:
                output = 'i_iter=%d, N_inliers=%d, w=%.2f' % (i_iter,N_inliers_max,N_inliers_max/N_data)
                print(output)

    # if N_inliers_max < 6:
    #     print('N_inliners=',N_inliers_max,' is less than 6. Failed to estimate the pose')
    #     R_best=np.zeros((3,3))
    #     T_best=np.zeros((3,1))
    #     M_best=np.zeros((3,4))

    # M = np.concatenate((R_best,T_best), axis=1)

    # estimate the pose using inliers only
    if N_inliers_max >= 6:
        R, T, M = estimate_pose_DLT(p_matched[:, i_inliers_best], p_W_matched[:,i_inliers_best], K)
    else:
        print('N_inliners=',N_inliers_max,' is less than 6. Failed to estimate the pose')
        R=np.zeros((3,3))
        T=np.zeros((3,1))
        M=np.zeros((3,4))


    return R, T, M, N_inliers_max, i_inliers_best


def draw_frame(ax, R, T, axis_length=1, line_width=1):
    C = -R.T@T.flatten()
    colors=('r','g','b')
    for i in range(3):
        ax.plot((C[0],C[0]+axis_length*R[i,0]),(C[1],C[1]+axis_length*R[i,1]),(C[2],C[2]+axis_length*R[i,2]), colors[i], linewidth=line_width)

def KLT(img_pre, img_cur, p_pre, W = 7, tol_bidir = 1, display = False):

    # KLT tracker: forward
    p_cur_T, bool_track, _ = cv2.calcOpticalFlowPyrLK(img_pre, img_cur, p_pre.T, None, winSize  = (2*W+1,2*W+1))
    p_cur = p_cur_T.T

    # KLT tracker: backward
    p_pre_rec_T, bool_track_backward, _ = cv2.calcOpticalFlowPyrLK(img_cur, img_pre, p_cur.T, None, winSize  = (2*W+1,2*W+1))
    p_pre_recovered = p_pre_rec_T.T

    # compute bi-directional error
    err_bidir = np.sum((p_pre - p_pre_recovered)**2, axis=0)

    # select inliers 
    index_track = np.where( (bool_track.flatten()>0) & (bool_track_backward.flatten()>0) & (err_bidir < tol_bidir) )[0]

    if display:
        p_pre_tracked = p_pre[:,index_track]
        p_cur_tracked = p_cur[:,index_track]

        # visualization
        plt.figure(dpi=150)
        plt.imshow(img_cur,cmap='gray')
        plt.axis('off')

        for i in range(len(index_track)):
            u_q, v_q = p_cur_tracked[0,i], p_cur_tracked[1,i] # matched keypoints on img_cur
            u_d, v_d = p_pre_tracked[0,i], p_pre_tracked[1,i] # matched keypoints on img_pre
            plt.plot(u_q, v_q, 'r+') 
            plt.plot(u_d, v_d, 'b+')
            plt.plot([u_d,u_q],[v_d,v_q],'r')
    
    return p_cur, index_track

def hom(p):
    # input (2 or 3) by n np.array
    # output (3 or 4) by n np.array for homogeneous coordinates
    assert(p.shape[0]==2 or p.shape[0]==3)
    return np.concatenate((p, np.ones((1,p.shape[1])) ), axis=0) 

def deh(p_h):
    # output (3 or 4) by n np.array for homogeneous coordinates
    # input (2 or 3) by n np.array for de-homonized coordinates
    n = p_h.shape[0] - 1
    assert(n==2 or n==3)
    
    return p_h[0:n,:]/p_h[n,:]

def VO_bootstrap(img0, img1, K, param, display = False):

    # PARAMETERS
    W_harris_patch = param['W_harris_patch']
    kappa_harris = param['kappa_harris']
    N_keypoint = param['N_keypoint']
    W_nms = param['W_nms']
    # KLT
    W_KLT = param['W_KLT']
    tol_KLT_bidir = param['tol_KLT_bidir']
    # find essential matrix
    tol_E = param['tol_E']
    tol_E_RANSAC_prob = param['tol_E_RANSAC_prob']
    # triangulation
    tol_TRI_mu = param['tol_TRI_mu']
    tol_TRI_rep = param['tol_TRI_rep']
    tol_TRI_depth = param['tol_TRI_depth']

    # # PARAMETERS
    # # keypoints
    # W_harris_patch, kappa_harris = 4, 0.08
    # N_keypoint, W_nms = 2000, 8
    # # KLT
    # W_KLT = 4
    # tol_KLT_bidir = 1
    # # find essential matrix
    # tol_E = 1
    # tol_E_RANSAC_prob = 0.99
    # # triangulation
    # tol_TRI_mu = 1e-3
    # tol_TRI_rep = 1

    # find keypoitns from img0
    harris_scores = harris_corner(img0, W_harris_patch, kappa_harris)
    keypoints0 = select_keypoints(harris_scores, N_keypoint, W_nms)
    p0 = np.array(keypoints0, dtype='float32').T
    p0 = p0[[1,0],:] 

    # track keypoints from img0 to img1
    p1, index_track = KLT(img0, img1, p0, W_KLT, tol_KLT_bidir, display = False)
    p0 = p0[:,index_track]
    p1 = p1[:,index_track]

    # compute essential matrix E
    E, bool_inliers = cv2.findEssentialMat(p0.T, p1.T, K, cv2.RANSAC, tol_E_RANSAC_prob, tol_E)
    index_inliers = np.where(bool_inliers.flatten() > 0)[0]
    p0 = p0[:, index_inliers]
    p1 = p1[:, index_inliers]

    # extract pose from E
    ret, R1, T1, bool_pose_inliers = cv2.recoverPose(E, p0.T, p1.T, K)
    index_inliers = np.where( bool_pose_inliers.flatten() > 0 )[0]
    p0 = p0[:, index_inliers]
    p1 = p1[:, index_inliers]

    # triangulation
    p0_h = hom(p0)
    p1_h = hom(p1)

    M0 = K @ np.concatenate( (np.identity(3), np.zeros((3,1))), axis=1 )
    M1 = K @ np.concatenate( (R1, T1), axis=1 )

    # p_W, index_inliers, mu_ratio, err = triangulation_robust(p0_h, p1_h, M0, M1, tol_TRI_mu, tol_TRI_rep)
    p_W, index_inliers, _, _, _ = triangulation_robust_depth(p0_h, p1_h, K, np.identity(3), np.zeros((3,1)),\
        K, R1, T1, tol_TRI_mu, tol_TRI_rep, tol_TRI_depth)

    # save keypoints and p_W from inliers
    keypoints = [(vu[1], vu[0]) for vu in p0[:,index_inliers].T]
    p_W = p_W[:,index_inliers]

    if display:
        fig = plt.figure(dpi=150,figsize=(6,10))

        fig.add_subplot(2, 1, 1)
        plt.imshow(img1,cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.axis('off')
        plt.plot(p0[0,:], p0[1,:], 'b+') 
        plt.plot(p1[0,:], p1[1,:], 'r+') 

        ax = fig.add_subplot(2, 1, 2, projection='3d')
        ax.plot(p_W[0,:],p_W[1,:],p_W[2,:],'b.',markersize=1)
        ax.set_xlim(-20,20)
        ax.set_ylim(-10,5)
        ax.set_zlim(-10,50)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elev=0., azim=-90)
        draw_frame(ax, np.identity(3), np.zeros((3,1)),axis_length=0.5)
        draw_frame(ax, R1, T1, axis_length=0.5)
        plt.tight_layout()

    return keypoints, p_W, R1, T1

from typing import NamedTuple
class state(NamedTuple):
    keypoints: list
    p_W: np.ndarray
    T_W_history: list

class candidate(NamedTuple):
    keypoints: list
    keypoints_org: list
    R_org: list
    T_org: list

def VO_localization_mapping(i_frame, img, img_rgb, img_pre, S_pre, C_pre, K, param, display_process=False):

    # PARAMETERS
    W_harris_patch = param['W_harris_patch']
    kappa_harris = param['kappa_harris']
    N_keypoint = param['N_keypoint']
    W_nms = param['W_nms']
    # KLT
    W_KLT = param['W_KLT']
    tol_KLT_bidir = param['tol_KLT_bidir']
    # triangulation
    tol_TRI_mu = param['tol_TRI_mu']
    tol_TRI_rep = param['tol_TRI_rep']
    tol_TRI_depth = param['tol_TRI_depth']
    # mapping
    tol_keypoints_new = param['tol_keypoints_new']

    # ## PARAMETERS
    # # Harris
    # W_harris_patch = 4 # size of Harris patch
    # kappa_harris = 0.08 # parameters of Harris score
    # W_nms = 8 # size of non-maximum suppresion for keypoints seclection
    # N_keypoint = 2000 # number of keypoints
    # # KLT
    # W_KLT = 4
    # tol_KLT_bidir = 1
    # # triangulation
    # tol_TRI_mu = 1e-3 # tolerance of trinagulation inlier for the singular value ratio
    # tol_TRI_rep = 0.5 # tolerance of trinagulation inlier for the reprojection error
    # # mapping
    # tol_keypoints_new = 20 # new keypoints should be district from the tracked keypoints by this distance

    ## LOCALIZATION

    # Track keypoints from img_pre to img
    p_pre = np.array(S_pre.keypoints, dtype='float32').T
    p_pre = p_pre[[1,0],:] 

    p, index_match = KLT(img_pre, img, p_pre, W_KLT, tol_KLT_bidir, display = False)
    
    p_pre_matched = p_pre[:,index_match]
    p_matched = p[:,index_match]
    p_W_matched = S_pre.p_W[:, index_match]
    
    # Estimate pose from 3D-to-2D correspondence using PnP

    retval, rvec, tvec, i_inliers = cv2.solvePnPRansac(p_W_matched.T, p_matched.T, K, None)
    i_inliers = i_inliers.flatten()
    N_inliers = i_inliers.shape[0]
    T = tvec
    R, _ = cv2.Rodrigues(rvec)
    T_W = (-R.T@T) # location of the vehicle in the W-frame

    # Alternativley, DLT RANSAC can be utlized, but it is less accurante than PnP
    # N_DLT_ iter = 1000 # number of RANSAC iteration
    # tol_DLT_inlier = 11 # tolerance of RANSIC DLT inliner in pixels
    # R, T, _, N_inliers, i_inliers_DLT = mae6292.estimate_pose_RANSAC_DLT( \
    #     p_matched_ho, p_W_matched_ho, K, N_DLT_iter, tol_DLT_inlier, display_iter=True)

    # Extract/save inliers of PnP
    p_pre_inliers = p_pre_matched[:,i_inliers]
    p_inliers = p_matched[:,i_inliers]   
    p_W_inliers = p_W_matched[:,i_inliers]  

    # Save the inliers to the state for the next iteration
    S_keypoints = [(vu[1], vu[0]) for vu in p_inliers.T]
    S_p_W = p_W_inliers.copy()

    ## MAPPING

    # Cetect new keypoints and compute descriptor for the current img
    harris_score = harris_corner(img, W_harris_patch, kappa_harris)
    keypoints_new = select_keypoints(harris_score, N_keypoint, W_nms)
    p_new = np.array(keypoints_new, dtype='float32').T
    p_new = p_new[[1,0],:] 

    index_C_pre_matched_inliers=[]
    index_C_pre_matched_outliers=[]

    # If there are the previous candidate keypoints,
    if len(C_pre.keypoints) > 0:
        C_keypoints =[]
        C_keypoints_org =[]
        C_R_org =[]
        C_T_org =[]

        # Track the previous candidate points to the current image
        C_p_pre = np.array(C_pre.keypoints, dtype='float32').T
        C_p_pre = C_p_pre[[1,0],:] 

        C_p, C_index_match = KLT(img_pre, img, C_p_pre, W_KLT, tol_KLT_bidir, display = False)
        
        C_p_pre_org = np.array(C_pre.keypoints_org, dtype='float32').T
        C_p_pre_org = C_p_pre_org[[1,0],:] 

        # Check if each matched candidate can be triangulated
        for i in C_index_match: 
            C_p_i = C_p[:,i]
            C_p_org_i = C_p_pre_org[:,i]
        
            # M_i = K @ np.concatenate((R, T), axis=1)
            # M_org = K @ np.concatenate((C_pre.R_org[i], C_pre.T_org[i]), axis=1)

            # p_W_i, index_pW_inliers, _, _ = triangulation_robust(\
            #     hom(C_p_i.reshape(2,1)), hom(C_p_org_i.reshape(2,1)), M_i, M_org, tol_TRI_mu, tol_TRI_rep)

            p_W_i, index_pW_inliers, _, _, _ = triangulation_robust_depth(hom(C_p_org_i.reshape(2,1)), hom(C_p_i.reshape(2,1)), \
                K, C_pre.R_org[i], C_pre.T_org[i], K, R, T, \
                tol_TRI_mu, tol_TRI_rep, tol_TRI_depth)


            # If the triangulated p_W is inlier then the corresponding keypoint is added to the state
            if index_pW_inliers.shape[0] > 0:
                index_C_pre_matched_inliers.append(i)
                S_keypoints.append((C_p_i[1], C_p_i[0]))
                S_p_W = np.concatenate( (S_p_W, p_W_i), axis=1)
            else:
            # Otherwise save it with the prior history as a tracked, continuing candidate
                index_C_pre_matched_outliers.append(i)
                C_keypoints.append((C_p_i[1], C_p_i[0]))
                C_keypoints_org.append(C_pre.keypoints_org[i])
                C_R_org.append(C_pre.R_org[i])
                C_T_org.append(C_pre.T_org[i])

        
        # Among the new keypoints, identify those which district from state inliers and tracked candidates
        C_p_matched = C_p[:,C_index_match]
        C_p_dist = scipy.spatial.distance.cdist(p_new.T, np.concatenate((p_inliers.T, C_p_matched.T), axis=0) , 'euclidean')
        C_index_distinct = np.where( np.min(C_p_dist, axis=1) > tol_keypoints_new )[0]
        
        # Distict keypoints are added as new candidates
        for i in C_index_distinct:
            C_keypoints.append((p_new[1,i], p_new[0,i]))
            C_keypoints_org.append((p_new[1,i], p_new[0,i]))
            C_R_org.append(R)
            C_T_org.append(T)

        print('S_pre=', p_pre.shape[1], ', KLT_matched=', p_pre_matched.shape[1], ', PnP_inliers=', p_pre_inliers.shape[1], \
            ', S_new = ',len(S_keypoints))
        print('C_pre=', C_p_pre.shape[1], ', KLT_matched=', len(C_index_match), ', TRI_inliers=', len(index_C_pre_matched_inliers), \
            ', C_new = ',len(C_keypoints))
        #print('C_added =', len(C_index_distinct))
        # print('S_new =', len(S_keypoints))#, 'S_pre_PnP_inliers+C_pre_TRI_inliers', p_pre_inliers.shape[1]+len(index_C_pre_matched_inliers))
        # print('C_new =', len(C_keypoints))#, 'matched_outliers+unmatched_distinct', len(index_C_pre_matched_outliers)+len(C_index_distinct))

    else:
        # if C_pre is empty, new keypoints distinct from p_inliers are saved as new candidates with the current pose
        p_dist = scipy.spatial.distance.cdist(p_new.T, p_inliers.T, 'euclidean')
        index_distinct = np.where( np.min(p_dist, axis=1) > tol_keypoints_new )[0]
        p_new = p_new[:,index_distinct]

        C_index_match =[]
        i_unmatched_outliers =[]
        C_keypoints = [(vu[1], vu[0]) for vu in p_new.T]
        C_keypoints_org = [(vu[1], vu[0]) for vu in p_new.T]
        C_R_org=[R for i in range(p_new.shape[1])]
        C_T_org=[T for i in range(p_new.shape[1])]
    

    # Save state and candidates
    S_T_W_history = S_pre.T_W_history.copy()
    S_T_W_history.append(T_W)
    S = state(S_keypoints, S_p_W, S_T_W_history)
    C = candidate(C_keypoints, C_keypoints_org, C_R_org, C_T_org)


    # Visualization
    if display_process:
        
        fig = plt.figure(dpi=300,figsize=(3,5))
        ax_img = fig.add_subplot(2, 1, 1)
        # plt.imshow(img,cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.axis('off')
        plt.scatter(p_inliers[0,:], p_inliers[1,:], marker='+', color='r', linewidth=0.5, s=3)
        plt.scatter(p_pre_inliers[0,:], p_pre_inliers[1,:], marker='+', color='b', linewidth=0.5, s=3)
        for i in range(N_inliers):
            u_q, v_q = p_inliers[0,i], p_inliers[1,i]
            u_d, v_d = p_pre_inliers[0,i], p_pre_inliers[1,i]
            plt.plot([u_d,u_q],[v_d,v_q],'r', linewidth=0.5)
        plt.imshow(img_rgb, zorder=1)
        plt.title('i_frame='+str(i_frame)+', N_p_W='+str(len(S_pre.keypoints))+', N_inliers='+str(N_inliers), fontsize=4)

        ax = fig.add_subplot(2, 1, 2, projection='3d')
        ax.plot(S_pre.p_W[0,:],S_pre.p_W[1,:],S_pre.p_W[2,:],'b.',markersize=0.5)
        # ax.plot(S_p_W[0,:],S_p_W[1,:],S_p_W[2,:],'g.',markersize=0.5)
        ax.plot(p_W_inliers[0,:],p_W_inliers[1,:],p_W_inliers[2,:],'r.',markersize=0.5)
        for i in range(1,len(S.T_W_history)):
            T_W_im = S.T_W_history[i-1].flatten()
            T_W_i = S.T_W_history[i].flatten() 
            ax.plot([T_W_im[0], T_W_i[0]], [T_W_im[1], T_W_i[1]], [T_W_im[2], T_W_i[2]], 'b', linewidth=0.5)
        
        ax.set_xlim(T_W[0]-20,T_W[0]+20)
        ax.set_zlim(T_W[2]-20,T_W[2]+20)
        ax.set_xlabel('x',fontsize=6)
        ax.set_ylabel('y',fontsize=6)
        ax.set_zlabel('z',fontsize=6)
        ax.xaxis.set_tick_params(labelsize=6)
        ax.yaxis.set_tick_params(labelsize=6)
        ax.zaxis.set_tick_params(labelsize=6)
        ax.view_init(elev=0., azim=-90)
        draw_frame(ax, np.identity(3), np.zeros((3,1)), axis_length=2, line_width=0.5)
        draw_frame(ax, R, T, axis_length=2, line_width=0.5)
        
    else:
        fig = None

    return R, T, S, C, fig
