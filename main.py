import cv2
import numpy as np
import os
import glob


def detectAndMatchFeature(img1, img2):
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=10, extended=True)

    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict()

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    goodMatches = []

    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            goodMatches.append(m)

    reg1 = cv2.drawMatches(img1, kp1, img2, kp2, goodMatches, None)
    cv2.imshow("matches", reg1)
    cv2.waitKey(0)

    dst_pts = np.float32([kp1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 2)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 2)

    return src_pts, dst_pts


def generateRandom(src_Pts, dest_Pts, N):
    r = np.random.choice(len(src_Pts), N)
    src = [src_Pts[i] for i in r]
    dest = [dest_Pts[i] for i in r]
    return np.asarray(src, dtype=np.float32), np.asarray(dest, dtype=np.float32)


def findH(src, dest, N):
    A = []
    for i in range(N):
        x, y = src[i][0], src[i][1]
        xp, yp = dest[i][0], dest[i][1]
        A.append([x, y, 1, 0, 0, 0, -x * xp, -xp * y, -xp])
        A.append([0, 0, 0, x, y, 1, -yp * x, -yp * y, -yp])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H


def ransacHomography(src_Pts, dst_Pts, ransacThreshold=0.995, maxIter=2000):
    maxI = 0
    maxSrcIn = []
    maxDstIn = []
    n = len(src_Pts)
    p1U = (np.append(src_Pts, np.ones((n, 1)), axis=1)).T
    for i in range(maxIter):
        srcP, destP = generateRandom(src_Pts, dst_Pts, 4)
        H = findH(srcP, destP, 4)
        p2e = H.dot(p1U)
        p2e = (p2e / p2e[2])[:2, :].T
        xx = np.linalg.norm(dst_Pts - p2e, axis=1)
        inlines = len(xx[xx < ransacThreshold])
        idx = np.argwhere(xx < ransacThreshold)
        idx = idx.reshape(1, len(idx))[0]
        if inlines > maxI:
            maxI = inlines
            maxSrcIn = src_Pts[idx]
            maxDstIn = dst_Pts[idx]
    FH = findH(maxSrcIn, maxDstIn, maxI)
    return FH


def stitchTwoImages(img1, img2, H, maxSize):
    height, width = maxSize
    img3 = np.zeros((height, width), dtype=np.uint8)
    img3[0:img1.shape[0], 0:img1.shape[1]] = img1
    X = np.mgrid[0:img2.shape[1], 0:img2.shape[0]].reshape(2, -1)
    X = np.insert(X, 2, values=1, axis=0)
    X = H.dot(X)
    X = np.around((X / X[2, :])[0:2, :].T)
    X = X.astype(np.int64)
    X[(X[:, 1] >= height) | (X[:, 1] < 0) | (X[:, 0] >= width) | (X[:, 0] < 0)] = 0
    img2 = img2.T
    img2 = img2.reshape(1, (img2.shape[0] * img2.shape[1]))[0]
    img3[X[:, 1], X[:, 0]] = img2
    return img3


def stitchConsecutively(p_files):
    images = [cv2.imread(p_files[i], cv2.IMREAD_GRAYSCALE) for i in range(len(p_files) - 1)]
    (j, x) = 0, 1
    while len(images) > 1:
        temp = []
        (i, k) = 0, int(len(images) / 2)
        for i in range(k):
            img1 = images.pop(0)
            img2 = images.pop(0)
            src_pts, dst_pts = detectAndMatchFeature(img1, img2)
            H = ransacHomography(src_pts, dst_pts)
            img3 = stitchTwoImages(img1, img2, H, (img1.shape[0], img1.shape[1] + x * 200))
            cv2.imshow("image " + str(j + i + 1), img3)
            cv2.waitKey(0)
            temp.append(img3)
        images = temp.copy()
        j += i + 1
        x += 1
    return images[0]


"""dir_path = os.path.dirname(os.path.realpath(__file__))
for t in range(9):
    pano = dir_path + "\pano" + str(t + 1)
    pano_file = glob.glob(pano + "\*.png")
    pano_img = stitchConsecutively(pano_file)"""
