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
    pixels = image.load()
    putpixel = image.putpixel
    imgx, imgy = image.size

    num_cells = int(math.hypot(imgx, imgy) * MAGNIFICATION)

    showtime = strftime("%Y%m%d%H%M%S", gmtime())
    showtime += "-" + str(num_cells)

    print("(+) Creating " + str(num_cells) + 
          " stipples with convergence point " +
          str(CONVERGENCE_LIMIT) + ".")

    centroids = [
        [random.randrange(imgx) for _ in range(num_cells)],
        [random.randrange(imgy) for _ in range(num_cells)]
    ]

    # precompute pixel densities
    rho = [[0] * imgx for _ in range(imgy)]
    for y in range(imgy):
        for x in range(imgx):
            rho[y][x] = 1 - pixels[x, y]/255.0  # rho

    # make folder to save each snapshot
    folder_base = "output/_step/" + showtime + "/"
    os.makedirs(folder_base)

    # save initial image
    clear_image(image.size, putpixel)
    draw_points(zip_points(centroids), putpixel)
    image.save(folder_base + "0.png", "PNG")

    # empty arrays for storing new centroid sums
    new_centroid_sums = [
        [0] * num_cells,  # x component
        [0] * num_cells,  # y component
        [0] * num_cells   # density
    ]

    # Iterate to convergence
    iteration = 1
    resolution = DEFAULT_RESOLUTION
    while True:
        # Zero all sums.
        zero_lists(new_centroid_sums)

        # Shade regions and add up centroid totals.
        sum_regions(centroids,
                    new_centroid_sums,
                    rho,
                    1.0 / resolution,
                    image.size)

        # Compute new centroids.
        centroidal_delta = compute_centroids(centroids,
                                             new_centroid_sums,
                                             image.size)

        # Print step difference.
        printr(str(iteration) +
               "     \tDifference: " +
               str(centroidal_delta) +
               ".\n")

        # Save a snapshot of the current image.
        clear_image(image.size, putpixel)
        draw_points(zip_points(centroids), putpixel)
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
    return magnify_and_draw_points(zip_points(centroids), image.size)


def compute_centroids(centroids, new_centroid_sums, image_size):
    """calculate centroids for a weighted voronoi diagram"""
    centroidal_delta = 0

    for i in range(len(centroids[0])):

        if not new_centroid_sums[2][i]:
            # all pixels in region have rho = 0
            # send centroid somewhere else
            centroids[0][i] = random.randrange(image_size[0])
            centroids[1][i] = random.randrange(image_size[1])
        else:
            new_centroid_sums[0][i] /= new_centroid_sums[2][i]
            new_centroid_sums[1][i] /= new_centroid_sums[2][i]
            # print("centroidal_delta" + str(centroidal_delta))
            centroidal_delta += hypot_square(
                (new_centroid_sums[0][i] - centroids[0][i]),
                (new_centroid_sums[1][i] - centroids[1][i])
            )
            centroids[0][i] = new_centroid_sums[0][i]
            centroids[1][i] = new_centroid_sums[1][i]

    return centroidal_delta


def sum_regions(centroids, new_centroid_sums, rho, res_step, size):
    """create weighted voronoi diagram and add up for new centroids"""

    # construct 2-dimensional tree from generating points
    tree = spatial.KDTree(zip(centroids[0], centroids[1]))

    imgx, imgy = size
    x_range = np.arange(res_step/2.0, imgx, res_step)
    y_range = np.arange(res_step/2.0, imgy, res_step)
    point_matrix = list(itertools.product(x_range, y_range))
    nearest_nbr_indices = tree.query(point_matrix)[1]
    for i in range(len(point_matrix)):
        point = point_matrix[i]
        x = point[0]
        y = point[1]
        r = rho[int(y)][int(x)]
        nearest_nbr_index = nearest_nbr_indices[i]
        new_centroid_sums[0][nearest_nbr_index] += r * x
        new_centroid_sums[1][nearest_nbr_index] += r * y
        new_centroid_sums[2][nearest_nbr_index] += r
        #
        if i % 10 == 0:
            #
            perc = float(i) / len(point_matrix)
            printr("{:.2%}".format(perc))


def zip_points(p):
    """zip 2d array into tuples"""
    return zip(p[0], p[1])


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


def clear_image(size, putpixel):
    """set all pixels in an image to its negative color"""
    imgx, imgy = size
    for y in range(imgy):
        for x in range(imgx):
            putpixel((x, y), NEG_COLOR)


def magnify_and_draw_points(points, size):
    """increase whitespace between images and draw final nodes"""
    magnified_size = (size[0] * MAGNIFICATION, size[1] * MAGNIFICATION)
    blank_magnified_image = Image.new("L", magnified_size)
    putpixel = blank_magnified_image.putpixel
    clear_image(magnified_size, putpixel)

    magnified_points = [tuple(MAGNIFICATION*x for x in point) for point in points]

    draw_points(magnified_points, putpixel)

    return blank_magnified_image


def draw_points(points, putpixel):
    """draw a set of points on an image"""
    for i in range(len(points)):
        pt = round_point(points[i])
        if pt == (0, 0):
            # Skip pixels at origin - they'll break the TSP art
            continue
        putpixel(pt, POS_COLOR)


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
