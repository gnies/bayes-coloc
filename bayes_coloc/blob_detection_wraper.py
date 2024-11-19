from skimage.feature import blob_log
import numpy as np

def blob_detection(image, min_sigma=1.5, max_sigma=2, num_sigma=10, threshold=0.05):
    """ Detect points in an image using blob detection. This is based on the skimage blob_log function.
    Parameters
    ----------
    image : ndarray
        The image to detect points in.
    min_sigma : float
    max_sigma : float
    num_sigma : int
    threshold : float
    Returns
    -------
    res : ndarray
        An array of points detected in the image. Each row is a point, with the first column being the x coordinate and the second column being the y coordinate

    """
    image = (image - image.min()) / (image.max()-image.min())
    blobs = blob_log(image, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)

    y = blobs[:, 0]
    x = blobs[:, 1]
    res = np.vstack([x, y]).T
    return res

