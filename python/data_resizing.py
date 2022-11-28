# Requires Python3 -m pip install opencv-Python
import cv2
import glob
import os
import numpy as np
from pathlib import Path
import sys
import struct
import matplotlib.pyplot as plt


'''
    RESIZE DATABASE FLYING THINGS DATABASE
'''
def make_image_directories():
    newFiles = "Data/SmallerThings"
    oldDirectory = "Data/FlyingThings/"
    pathExtendImages = "Data/FlyingThings/frames_cleanpass/*"
    extra_level = "/*"

    os.mkdir(newFiles+"/frames_cleanpass")
    extra_level = "/*"
    while(glob.glob(pathExtendImages+extra_level)):
        image_directory = glob.glob(pathExtendImages)
        for directory in image_directory:
            offset = len(oldDirectory)
            directory = directory[offset:]
            path = os.path.join(newFiles, directory)
            os.mkdir(path) 
            print("Directory '% s' created" % directory) 
        pathExtendImages += extra_level

def make_disparity_directories():
    newFiles = "Data/SmallerThings"
    oldDirectory = "Data/FlyingThings/"
    pathExtendDisparity = "Data/FlyingThings/disparity/*"
    extra_level = "/*"

    os.mkdir(newFiles+"/disparity")
    while(glob.glob(pathExtendDisparity+extra_level)):
        disparity_directory = glob.glob(pathExtendDisparity)
        for directory in disparity_directory:
            offset = len(oldDirectory)
            directory = directory[offset:]
            path = os.path.join(newFiles, directory)
            os.mkdir(path) 
            print("Directory '% s' created" % directory) 
        pathExtendDisparity += extra_level


def resize_images():
    newFiles = "Data/SmallerThings"
    oldDirectory = "Data/FlyingThings/"
    pathExtendImages = "Data/FlyingThings/frames_cleanpass/*"
    extra_level = "/*"

    while(glob.glob(pathExtendImages+extra_level)):
        pathExtendImages += extra_level

    for image_file in glob.glob(pathExtendImages):
        offset = len(oldDirectory)
        new_file = image_file[offset:]
        path = os.path.join(newFiles, new_file)
        img = cv2.imread(image_file)
        resized = cv2.resize(img, (img.shape[1]//4,img.shape[0]//4))

        cv2.imwrite(path, resized)

def resize_disparities():
    newFiles = "Data/SmallerThings"
    oldDirectory = "Data/FlyingThings/"
    pathExtendDisparity = "Data/FlyingThings/disparity/*"
    extra_level = "/*"

    while(glob.glob(pathExtendDisparity+extra_level)):
        pathExtendDisparity += extra_level

    for image_file in glob.glob(pathExtendDisparity):
        offset = len(oldDirectory)
        new_file = image_file[offset:]
        path = os.path.join(newFiles, new_file)
        img = read_pfm(image_file)
        img = smallisize(img)
        # Normalizing: disparity is 4 times smaller when there are 4 times less
        # pixels in the line!
        img /= 4
        write_pfm(path, img)



'''
    PFM HANDLING  
'''
def read_pfm(filename):
    with Path(filename).open('rb') as pfm_file:

        line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
        assert line1 in ('PF', 'Pf')
        
        channels = 3 if "PF" in line1 else 1
        width, height = (int(s) for s in line2.split())
        scale_endianess = float(line3)
        bigendian = scale_endianess > 0
        scale = abs(scale_endianess)

        buffer = pfm_file.read()
        samples = width * height * channels
        assert len(buffer) == samples * 4
        
        fmt = f'{"<>"[bigendian]}{samples}f'
        decoded = struct.unpack(fmt, buffer)
        shape = (height, width, 3) if channels == 3 else (height, width)
        return np.flipud(np.reshape(decoded, shape)) * scale

def write_pfm(file, image, scale=1):
    image = np.fliplr(np.rot90(np.rot90(image)))
    file = open(file, 'wb')
    file.write('Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image.tofile(file)

def resize_pfm(path):
    img = read_pfm(path)
    img = smallisize(img)
    # Normalizing: disparity is 4 times smaller when there are 4 times less
    # pixels in the line!
    img /= 4
    return img

def smallisize(img):
    new_image = []
    lines = len(img)//4
    columns = len(img[0])//4
    for y in range(lines):
        line = []
        for x in range(columns):
            avg = 0
            for i in range(4):
                for j in range(4):
                    avg += img[y*4+i][x*4+j]
            avg /= 16
            line.append(avg)
        new_image.append(line)

    new_image = np.matrix(new_image, dtype='float32')
    return new_image





'''
    TESTS
'''
def test_resize():
    imageL = cv2.imread("Data/SmallerThings/frames_cleanpass/TEST/A/0000/left/0010.png")
    imageR = cv2.imread("Data/SmallerThings/frames_cleanpass/TEST/A/0000/right/0010.png")
    disparity = resize_pfm("Data/FlyingThings/disparity/TEST/A/0000/left/0010.pfm")
    
    defaced = []
    width = len(imageR[0])
    length = len(imageR)
    for i in range(length):
        new_line = []
        for j in range(width):
            displacement = int (disparity[i,j])
            horiz = (j+displacement)%240
            new_line.append(imageL[i][horiz])
        defaced.append(new_line)
    defaced = np.array(defaced)

    
    cv2.imwrite("out/test/preprocessing-tests/defaced.png", defaced)
    cv2.imwrite("out/test/preprocessing-tests/left.png", imageL)
    cv2.imwrite("out/test/preprocessing-tests/right.png", imageR)
    plt.imshow(disparity)
    plt.show()

def test_read_pfm():
    disparity = read_pfm("Data/SmallerThings/disparity/TEST/A/0008/left/0010.pfm")
    img = cv2.imread("Data/SmallerThings/frames_cleanpass/TEST/A/0008/left/0010.png")
    cv2.imwrite("out/test/preprocessing-tests/test_read_pfm_img.png", img)
    plt.imshow(disparity)
    plt.show()

def test_database_size():
    newFiles = "Data/SmallerThings"
    oldDirectory = "Data/FlyingThings/"
    pathExtendImages = "Data/FlyingThings/frames_cleanpass/*"
    pathExtendDisparity = "Data/FlyingThings/disparity/*"
    extra_level = "/*"

    while(glob.glob(pathExtendDisparity+extra_level)):
        disparity_directory = glob.glob(pathExtendDisparity)
        for directory in disparity_directory:
            offset = len(oldDirectory)
            directory = directory[offset:]
            path_old = os.path.join(oldDirectory, directory)
            path_new = os.path.join(newFiles, directory)
            old_size = len(glob.glob(path_old))
            new_size = len(glob.glob(path_new))
            assert old_size==new_size
        pathExtendDisparity += extra_level


    extra_level = "/*"
    while(glob.glob(pathExtendImages+extra_level)):
        image_directory = glob.glob(pathExtendImages)
        for directory in image_directory:
            offset = len(oldDirectory)
            directory = directory[offset:]
            path_old = os.path.join(oldDirectory, directory)
            path_new = os.path.join(newFiles, directory)
            old_size = len(glob.glob(path_old))
            new_size = len(glob.glob(path_new))
            assert old_size==new_size
        pathExtendImages += extra_level
    
    print("TEST PASSED")


