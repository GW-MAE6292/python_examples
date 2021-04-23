import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.linalg as linalg
import scipy.signal
import scipy.spatial.distance
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
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

    N_negative_depth = np.zeros(4)
    for i in range(4):
        M2 = K2@np.concatenate( (R[:,:,i], T[:,i].reshape(-1,1)), axis=1 )
        P = triangulation( p1, p2, M1, M2)
        N_negative_depth[i] = len(np.where(P[2,:]<0)[0])

    i_opt = np.argmin(N_negative_depth)

    R=R[:,:,i_opt]
    T=T[:,i_opt].reshape(-1,1)

    return R, T

def eightpoint_algorithm(p1, p2, K1, K2):
    # p1, p2: 3 by n pixel homogeneous coordinates
    # K1, K2: 3 by 3 intrinsic parameters
    n = p1.shape[1]
    p_bar_1 = scipy.linalg.inv(K1)@p1
    p_bar_2 = scipy.linalg.inv(K1)@p2

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
            
            if display_iter:
                output = 'i_iter=%d, N_inliers=%d, w=%.2f' % (i_iter,N_inliers_max,N_inliers_max/N_data)
                print(output)

    # estimate the pose using inliers only
    if N_inliers_max >= 6:
        R, T, M = estimate_pose_DLT(p_matched[:, i_inliers_best], p_W_matched[:,i_inliers_best], K)
    else:
        print('N_inliners=',N_inliers_max,' is less than 6. Failed to estimate the pose')
        R=np.zeros((3,3))
        T=np.zeros((3,1))
        M=np.zeros((3,4))


    return R, T, M, N_inliers_max, i_inliers_best


def draw_frame(ax, R, T, axis_length=1):
    C = -R.T@T.flatten()
    colors=('r','g','b')
    for i in range(3):
        ax.plot((C[0],C[0]+axis_length*R[i,0]),(C[1],C[1]+axis_length*R[i,1]),(C[2],C[2]+axis_length*R[i,2]), colors[i])

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