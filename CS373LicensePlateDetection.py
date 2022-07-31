import math
import sys
from pathlib import Path

from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png


class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):
    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue=0):
    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    scale_factor = None
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    min_and_max = computeMinAndMaxValues(pixel_array, image_width, image_height)
    min_value = min_and_max[0]
    max_value = min_and_max[1]
    if (max_value - min_value) != 0:
        scale_factor = 255.0 / (max_value - min_value)
    if max_value > min_value:
        for y in range(image_height):
            for x in range(image_width):
                value = round(int(pixel_array[y][x] - min_value) * scale_factor)
                if value < 0:
                    greyscale_pixel_array[y][x] = 0
                elif value > 255:
                    greyscale_pixel_array[y][x] = 255
                else:
                    greyscale_pixel_array[y][x] = value

    return greyscale_pixel_array


def computeMinAndMaxValues(pixel_array, image_width, image_height):
    min_value = 0
    max_value = 0

    for y in range(image_height):
        for x in range(image_width):
            if pixel_array[y][x] < min_value:
                min_value = pixel_array[y][x]
            if pixel_array[y][x] > max_value:
                max_value = pixel_array[y][x]
    return (min_value, max_value)


def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):
    res = [[0 for x in range(image_width)] for y in range(image_height)]

    for i in range(1, image_height - 2):
        for j in range(1, image_width - 2):
            box = []

            for eta in [-2, -1, 0, 1, 2]:
                for xi in [-2, -1, 0, 1, 2]:
                    box.append(pixel_array[i + eta][j + xi])

            mean = sum(box) / 25
            points_sum = 0

            for val in box:
                points_sum += (val - mean) ** 2

            variance = points_sum / 25
            std_dev = math.sqrt(variance)

            res[i][j] = round(std_dev, 3)

    return res


def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    pixelArray = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for k in range(image_width):
            if pixel_array[i][k] >= threshold_value:
                pixelArray[i][k] = 255
    return pixelArray


def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    res = [[0 for x in range(image_width)] for y in range(image_height)]
    for i in range(image_height):
        for j in range(image_width):

            if i + 1 > image_height - 1 or i - 1 < 0 or j + 1 > image_width - 1 or j - 1 < 0:
                res[i][j] = 0
            else:
                if (pixel_array[i - 1][j - 1] != 0 and pixel_array[i - 1][j] != 0 and pixel_array[i - 1][j + 1] != 0 and
                        pixel_array[i][j - 1] != 0 and pixel_array[i][j] != 0 and pixel_array[i][j + 1] != 0 and
                        pixel_array[i + 1][j - 1] != 0 and pixel_array[i + 1][j] != 0 and pixel_array[i + 1][
                            j + 1] != 0):
                    res[i][j] = 1
                else:
                    res[i][j] = 0
    return res


def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    res = [[0 for x in range(image_width)] for y in range(image_height)]
    for i in range(image_height):
        for j in range(image_width):

            if i + 1 > image_height - 1 or i - 1 < 0 or j + 1 > image_width - 1 or j - 1 < 0:
                res[i][j] = 0
            else:
                if (pixel_array[i - 1][j - 1] != 0 or pixel_array[i - 1][j] != 0 or pixel_array[i - 1][j + 1] != 0 or
                        pixel_array[i][j - 1] != 0 or pixel_array[i][j] != 0 or pixel_array[i][j + 1] != 0 or
                        pixel_array[i + 1][j - 1] != 0 or pixel_array[i + 1][j] != 0 or pixel_array[i + 1][j + 1] != 0):
                    res[i][j] = 1

    return res


def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    res = [[0 for _ in range(image_width)] for _ in range(image_height)]
    visited = {(x, y): False for x in range(image_height) for y in range(image_width)}
    components = {}
    component = 1
    queue = Queue()

    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] != 0 and not visited[(i, j)]:
                queue.enqueue((i, j))
                components[component] = 0

                while not queue.isEmpty():
                    current = queue.dequeue()

                    x, y = current[0], current[1]
                    visited[(x, y)] = True

                    res[x][y] = component
                    components[component] += 1

                    if x - 1 >= 0:
                        if pixel_array[x - 1][y] != 0 and not visited[(x - 1, y)]:
                            queue.enqueue((x - 1, y))
                            visited[(x - 1, y)] = True

                    if x + 1 < image_height:
                        if pixel_array[x + 1][y] != 0 and not visited[(x + 1, y)]:
                            queue.enqueue((x + 1, y))
                            visited[(x + 1, y)] = True

                    if y - 1 >= 0:
                        if pixel_array[x][y - 1] != 0 and not visited[(x, y - 1)]:
                            queue.enqueue((x, y - 1))
                            visited[(x, y - 1)] = True

                    if y + 1 < image_width:
                        if pixel_array[x][y + 1] != 0 and not visited[(x, y + 1)]:
                            queue.enqueue((x, y + 1))
                            visited[(x, y + 1)] = True

                component += 1

    return res, components


def getLargestComponentCoords(connected_component_image, component_dict):
    min_pixel = None
    max_pixel = None
    component_dict = sorted(component_dict.items(), key=lambda x: x[1], reverse=True)
    reversed_keys = [x[0] for x in component_dict]

    for key in reversed_keys:
        x_values = []
        y_values = []
        for i in range(len(connected_component_image)):
            for j in range(len(connected_component_image[i])):
                if connected_component_image[i][j] == key:
                    x_values.append(i)
                    y_values.append(j)

        min_pixel = [min(x_values), min(y_values)]
        max_pixel = [max(x_values), max(y_values)]

        bbox_width = max_pixel[1] - min_pixel[1]
        bbox_height = max_pixel[0] - min_pixel[0]
        aspect_ratio = bbox_width / bbox_height

        if 1.5 <= aspect_ratio <= 5:
            break

    return (min_pixel, max_pixel)


# This is our code skeleton that performs the license plate detection.
# Feel free to try it on your own images of cars, but keep in mind that with our algorithm developed in this lecture,
# we won't detect arbitrary or difficult to detect license plates!
def main():
    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    input_filename = "numberplate6.png"

    if command_line_arguments:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(input_filename.replace(".png", "_output.png"))
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Greyscale')
    axs1[0, 0].imshow(px_array_r, cmap='gray')

    # STUDENT IMPLEMENTATION here
    px_array = scaleTo0And255AndQuantize(px_array_r, image_width, image_height)
    px_array = computeStandardDeviationImage5x5(px_array, image_width, image_height)
    px_array = scaleTo0And255AndQuantize(px_array, image_width, image_height)

    axs1[0, 1].set_title('Contrast Stretching and Quantization')
    axs1[0, 1].imshow(px_array, cmap='gray')

    px_array = computeThresholdGE(px_array, 140, image_width, image_height)

    for x in range(3):
        px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)

    for y in range(3):
        px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)

    axs1[1, 0].set_title('After dilation and erosion')
    axs1[1, 0].imshow(px_array, cmap='gray')

    res, component_dict = computeConnectedComponentLabeling(px_array, image_width, image_height)
    min_pixel, max_pixel = getLargestComponentCoords(res, component_dict)

    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('License plate detection')
    axs1[1, 1].imshow(px_array_r, cmap='gray')
    rect = Rectangle((min_pixel[1], min_pixel[0]), max_pixel[1] - min_pixel[1], max_pixel[0] - min_pixel[0],
                     linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()
