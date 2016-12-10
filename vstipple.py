#
# jack morris 11/08/16
#

from __future__ import print_function, division, absolute_import
import itertools
import math
import numpy as np
import os
from PIL import Image
import random
from scipy import spatial
import sys
from time import gmtime, strftime


# global variables
NEG_COLOR = 255
POS_COLOR = 0
CONVERGENCE_LIMIT = 5 * 10**-1
DEFAULT_RESOLUTION = 1
MAGNIFICATION = 8


def voronoi_stipple(image):
    """compute the weighted voronoi stippling for an image"""
    pixels = np.array(image)
    imgx, imgy = pixels.shape

    num_cells = int(math.hypot(imgx, imgy) * MAGNIFICATION)

    showtime = strftime("%Y%m%d%H%M%S", gmtime())
    showtime += "-" + str(num_cells)

    print("(+) Creating " + str(num_cells) + 
          " stipples with convergence point " +
          str(CONVERGENCE_LIMIT) + ".")

    centroids = np.array(
        [np.random.randint(imgx, size=(num_cells,)),
         np.random.randint(imgy, size=(num_cells,))]
    )

    # precompute pixel densities
    rho = np.array(1 - pixels / 255.0)

    # make folder to save each snapshot
    folder_base = "output/_step/" + showtime + "/"
    os.makedirs(folder_base)

    # save initial image
    clear_image(pixels)
    draw_points(centroids, pixels)
    image.save(folder_base + "0.png", "PNG")

    # empty arrays for storing new centroid sums
    new_centroid_sums = np.zeros((3, num_cells))

    # Iterate to convergence
    iteration = 1
    resolution = DEFAULT_RESOLUTION
    while True:
        # Zero all sums.
        new_centroid_sums[:] = 0

        # Shade regions and add up centroid totals.
        sum_regions(centroids,
                    new_centroid_sums,
                    rho,
                    1.0 / resolution,
                    pixels.shape)

        # Compute new centroids.
        centroidal_delta = compute_centroids(centroids,
                                             new_centroid_sums,
                                             pixels.shape)
        # Print step difference.
        printr(str(iteration) +
               "     \tDifference: " +
               str(centroidal_delta) +
               ".\n")

        # Save a snapshot of the current image.
        clear_image(pixels)
        draw_points(centroids, pixels)
        image.save(folder_base + str(iteration) + ".png", "PNG")

        # If no pixels shifted, we have to increase resolution.
        if centroidal_delta == 0.0:
            resolution *= 2
            print("(+) Increasing resolution to " + str(resolution) + "x.")

        # Break if difference below convergence point.
        elif centroidal_delta < CONVERGENCE_LIMIT * resolution:
            break

        # Increase iteration count.
        iteration += 1

    # Final print statement.
    print("(+) Magnifying image and drawing final centroids.")
    return magnify_and_draw_points(centroids, pixels.shape)


def compute_centroids(centroids, new_centroid_sums, image_size):
    """calculate centroids for a weighted voronoi diagram"""
    # centroidal_delta = 0

    zero_rho = np.where(new_centroid_sums[2] == 0)[0]
    nonzero_rho = np.where(np.abs(new_centroid_sums[2]) > 0)

    centroids[0, zero_rho] = np.random.randint(image_size[0],
                                               size=(zero_rho.shape[0],))
    centroids[1, zero_rho] = np.random.randint(image_size[1],
                                               size=(zero_rho.shape[0],))
    normalizer = new_centroid_sums[2, nonzero_rho][None, :]
    new_centroid_sums[:2, nonzero_rho] /= normalizer

    diffs = new_centroid_sums[:2, nonzero_rho] - centroids[:, nonzero_rho]
    centroidal_delta = (diffs ** 2).sum()
    centroids[:, nonzero_rho] = new_centroid_sums[:2, nonzero_rho]
    # for i in range(len(centroids[0])):
    #
    #     if not new_centroid_sums[2][i]:
    #         # all pixels in region have rho = 0
    #         # send centroid somewhere else
    #         centroids[0][i] = random.randrange(image_size[0])
    #         centroids[1][i] = random.randrange(image_size[1])
    #     else:
    #         new_centroid_sums[0][i] /= new_centroid_sums[2][i]
    #         new_centroid_sums[1][i] /= new_centroid_sums[2][i]
    #         # print("centroidal_delta" + str(centroidal_delta))
    #         centroidal_delta += hypot_square(
    #             (new_centroid_sums[0][i] - centroids[0][i]),
    #             (new_centroid_sums[1][i] - centroids[1][i])
    #         )
    #         centroids[0][i] = new_centroid_sums[0][i]
    #         centroids[1][i] = new_centroid_sums[1][i]

    return centroidal_delta


def sum_regions(centroids, new_centroid_sums, rho, res_step, size):
    """create weighted voronoi diagram and add up for new centroids"""

    # construct 2-dimensional tree from generating points
    tree = spatial.KDTree(centroids.T)

    imgx, imgy = size
    x_range = np.arange(res_step/2.0, imgx, res_step)
    y_range = np.arange(res_step/2.0, imgy, res_step)
    point_matrix = list(itertools.product(x_range, y_range))
    nearest_nbr_indices = tree.query(point_matrix)[1]
    x = np.array(point_matrix[0], dtype=np.int)
    y = np.array(point_matrix[1], dtype=np.int)
    r = rho[x, y]
    for point, nearest_nbr in zip(point_matrix, nearest_nbr_indices):
        x = point[0]
        y = point[1]
        r = rho[int(x), int(y)]
        new_centroid_sums[0][nearest_nbr] += r * x
        new_centroid_sums[1][nearest_nbr] += r * y
        new_centroid_sums[2][nearest_nbr] += r


def zip_points(p):
    """zip 2d array into tuples"""
    return list(zip(p[0], p[1]))


def printr(s):
    """carriage return and print"""
    sys.stdout.write("\r" + s)
    sys.stdout.flush()


def zero_lists(the_lists):
    """set each element in a set of lists to zero"""
    for the_list in the_lists:
        zero_list(the_list)


def zero_list(the_list):
    """set every element in a list to zero"""
    for x in range(len(the_list)):
        the_list[x] = 0


def clear_image(arr):
    """set all pixels in an image to its negative color"""
    arr[:] = NEG_COLOR


def magnify_and_draw_points(points, size):
    """increase whitespace between images and draw final nodes"""
    magnified_size = (size[0] * MAGNIFICATION, size[1] * MAGNIFICATION)
    blank_magnified_image = np.zeros(magnified_size, dtype=np.int)
    clear_image(blank_magnified_image)

    magnified_points = MAGNIFICATION * points

    draw_points(magnified_points, blank_magnified_image)

    return blank_magnified_image


def draw_points(points, arr):
    """draw a set of points on an image"""
    previous_origin = arr[0, 0]

    arr[points[0], points[1]] = POS_COLOR
    arr[0, 0] = previous_origin


def round_point(pt):
    """cast a tuple to integers"""
    return int(pt[0]), int(pt[1])


def hypot_square(d1, d2):
    """faster hypotenuse (don't need to square-root it)"""
    if d1 == 0 and d2 == 0:
        return 0
    elif d1 == 0:
        return d2 ** 2
    elif d2 == 0:
        return d1 ** 2
    return d1**2 + d2**2
