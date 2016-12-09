#
# jack morris 11/26/16
#

from __future__ import print_function, division, absolute_import
from PIL import Image, ImageDraw

"""
reads in an image composed of positive pixels (nodes) and negative pixels (free
space)
creates an approximate solution using a TSP heuristic
outputs the image with the path traced out as one line
"""

# store image pos and neg color
NEG_COLOR = None
POS_COLOR = None
CIRCLE_RADIUS = 6 


def draw_dots_on(image, stretched=True):
    """connect_the_dots (this file's main method)"""
    nodes = read_in_nodes(image)
    print("Read in " + str(len(nodes)) + " nodes.")

    if not stretched:
        image = magnify_image(image.size, nodes, CIRCLE_RADIUS)
        nodes = [(x[0] * CIRCLE_RADIUS, x[1] * CIRCLE_RADIUS) for x in nodes]

    return draw_dots(nodes, image)


def magnify_image(original_size, nodes, stretch_factor):
    """stretch whitespace over an image"""
    magnified_size = (original_size[0] * stretch_factor,
                      original_size[1] * stretch_factor)
    magnified_image = Image.new("L", magnified_size)

    putpixel = magnified_image.putpixel

    clear_image(magnified_size, putpixel)

    return magnified_image


def clear_image(size, putpixel):
    """set all pixels in image to negative color"""
    imgx, imgy = size
    for y in range(imgy):
        for x in range(imgx):
            putpixel((x, y), NEG_COLOR)


def draw_points(points, putpixel):
    """draw a set of points on an image"""
    for i in range(len(points)):
        pt = (int(points[i][0]), int(points[i][1]))
        if pt == (0, 0):
            # Skip pixels at origin - they'll break the TSP art
            continue
        putpixel(pt, POS_COLOR)


def draw_dots(nodes, image):
    """draw a set of lines on an image"""

    # clear image
    imgx, imgy = image.size
    pixels = image.load()
    for y in range(imgy):
        for x in range(imgx):
            pixels[x, y] = NEG_COLOR

    # draw dots
    draw = ImageDraw.Draw(image)
    for node in nodes:
        draw_circle(draw, node, CIRCLE_RADIUS)

    # return
    return image


def draw_circle(image_draw, pt, radius):
    """draws a circle with pillow ImageDraw"""
    pt_1 = (pt[0] - radius, pt[1] - radius)
    pt_2 = (pt[0] + radius, pt[1] + radius)
    image_draw.ellipse([pt_1, pt_2], fill=POS_COLOR)


def read_in_nodes(image):
    """read in nodes from input image"""
    global NEG_COLOR, POS_COLOR

    width, height = image.size
    pixels = image.load()
    nodes = []
    NEG_COLOR = pixels[0, 0]
    print("Neg_color:" + str(NEG_COLOR))

    for i in range(width):
        for j in range(height):

            if pixels[i, j] != NEG_COLOR:

                if not POS_COLOR:
                    POS_COLOR = pixels[i, j]

                nodes.append((i, j))

    return nodes
