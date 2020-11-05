SIZE = 10
SIDEDIV = 3


import numpy as np
from math import *
import cv2



"""Functions for downloading and reading MNIST data."""
import math
import gzip
import os
from six.moves.urllib.request import urlretrieve
import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm


SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


def maybe_download(filename, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)[0]
        rows = _read32(bytestream)[0]
        cols = _read32(bytestream)[0]
        #print('check', magic, num_images, rows, cols, rows * cols * num_images)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    #print('check num_labels',num_labels)
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename, one_hot=False):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)[0]
        #print('check', magic, num_items)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels


class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                       labels.shape))
            self._num_examples = images.shape[0]
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            #fake_image = [1.0 for _ in xrange(784)]
            fake_image = [1.0 for _ in range(784)]
            fake_label = 0
            #return [fake_image for _ in xrange(batch_size)], [
            #    fake_label for _ in xrange(batch_size)]
            return [fake_image for _ in range(batch_size)], [
                fake_label for _ in range(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir, fake_data=False, one_hot=False):
    class DataSets(object):
        pass
    data_sets = DataSets()
    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    VALIDATION_SIZE = 5000
    local_file = maybe_download(TRAIN_IMAGES, train_dir)
    train_images = extract_images(local_file)
    local_file = maybe_download(TRAIN_LABELS, train_dir)
    train_labels = extract_labels(local_file, one_hot=one_hot)
    local_file = maybe_download(TEST_IMAGES, train_dir)
    test_images = extract_images(local_file)
    local_file = maybe_download(TEST_LABELS, train_dir)
    test_labels = extract_labels(local_file, one_hot=one_hot)
    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]
    data_sets.train = DataSet(train_images, train_labels)
    data_sets.validation = DataSet(validation_images, validation_labels)
    data_sets.test = DataSet(test_images, test_labels)
    print('check data_sets train', data_sets.train, train_images.shape)
    print('check data_sets validation', data_sets.validation, validation_images.shape)
    print('check data_sets test', data_sets.test, test_images.shape)
    return data_sets


#trainim = "train-images-idx3-ubyte.gz"
#testim = "t10k-images-idx3-ubyte.gz"
#trainlab = "train-labels-idx1-ubyte.gz"
#testlab = "t10k-labels-idx1-ubyte.gz"
#===============================================================================
# maybe_download(trainim, "MNIST_data") #train-images-idx3-ubyte.gz: training set images (9912422 bytes)
# maybe_download(trainlab, "MNIST_data") #train-labels-idx1-ubyte.gz: training set labels (28881 bytes)
# maybe_download(testim, "MNIST_data") #t10k-images-idx3-ubyte.gz: test set images (1648877 bytes)
# maybe_download(testlab, "MNIST_data") #t10k-labels-idx1-ubyte.gz: test set labels (4542 bytes)
#===============================================================================


# FILTER
def REDUCEPIXELSTOTHEMAX__(imagematrix2828, plotimages=False):

    im = imagematrix2828.reshape(28,28)
    
    
    #===============================================================================
    # COUNT ZEROS (=BLANK PIXEL) IN THE RESIZED IM
    #===============================================================================
    count_zeros_stra = 0
    count_zeros_reve = 0
    for i in range(np.shape(im)[0]):
        for j in range(np.shape(im)[1]):
            if im[i][j] == 0:
                count_zeros_stra += 1
            else:
                break
            if im[np.shape(im)[0]-1-i][np.shape(im)[1]-1-j] == 0:
                count_zeros_reve += 1
            else:
                break
    
    
    #print('count_zeros_stra:',count_zeros_stra,'count_zeros_reve',count_zeros_reve)
    each_side_blanks = min(count_zeros_stra,count_zeros_reve) / np.shape(im)[0]
    #print('each_side_blanks',each_side_blanks,'over',np.shape(im)[0]*np.shape(im)[1],'pixels')
    
    #CALCULATE NEW RESIM MATRIX
    c_zeros_stra = 0
    c_zeros_reve = 0
    resim = np.empty((np.shape(im)[0]-floor(each_side_blanks),np.shape(im)[1]-floor(each_side_blanks)))
    #print('shape(resim)',np.shape(resim))
    #===========================================================================
    # FOR THIS SET, TYPICALLY SIDES ARE ~12, SO WE FORCE IT (CHECK ALWAYS THE PRINT BEFORE
    #===========================================================================
    resim = np.empty((SIZE, SIZE))

    sidediv = SIDEDIV #FROM 3 TO 5
    counti = 0
    for i in range(np.shape(im)[0]-sidediv):
        if i > each_side_blanks/sidediv and i < np.shape(im)[0] - each_side_blanks/sidediv:
            
            countj = 0
            for j in range(np.shape(im)[1]-sidediv):
                if countj >= np.shape(resim)[1]:
                    break
                
                if j > each_side_blanks/sidediv and j < np.shape(im)[1] - each_side_blanks/sidediv:
                    resim[counti][countj] = (im[i+1][j+1]+1.5*im[i+sidediv][j]+1.5*im[i][j+sidediv]+2*im[i+sidediv][j+sidediv]) / 6
                    #print(max(im[i+1][j+1],im[i+sidediv][j],im[i][j+sidediv],im[i+sidediv][j+sidediv]),(0.00123456 + min(im[i+1][j+1],im[i+sidediv][j],im[i][j+sidediv],im[i+sidediv][j+sidediv])))
                    #resim[counti][countj] = log2(0.00123456 + max(im[i+1][j+1],im[i+sidediv][j],im[i][j+sidediv],im[i+sidediv][j+sidediv])/(0.00123456 + min(im[i+1][j+1],im[i+sidediv][j],im[i][j+sidediv],im[i+sidediv][j+sidediv])))
                    countj += 1

            counti += 1
            if counti >= np.shape(resim)[0]:
                break
    
    if plotimages == True:
        plt.imshow(resim, cmap = plt.cm.binary)
        plt.show()

    return resim


# FILTER
def REDUCEPIXELSTOTHEMAX(imageindex, plotimages=False):

    im = data_sets.train.images[imageindex].reshape(28,28)

    
    #===============================================================================
    # COUNT ZEROS (=BLANK PIXEL) IN THE RESIZED IM
    #===============================================================================
    count_zeros_stra = 0
    count_zeros_reve = 0
    for i in range(np.shape(im)[0]):
        for j in range(np.shape(im)[1]):
            if im[i][j] == 0:
                count_zeros_stra += 1
            else:
                break
            if im[np.shape(im)[0]-1-i][np.shape(im)[1]-1-j] == 0:
                count_zeros_reve += 1
            else:
                break
    
    
    #print('count_zeros_stra:',count_zeros_stra,'count_zeros_reve',count_zeros_reve)
    each_side_blanks = min(count_zeros_stra,count_zeros_reve) / np.shape(im)[0]
    #print('each_side_blanks',each_side_blanks,'over',np.shape(im)[0]*np.shape(im)[1],'pixels')
    
    #CALCULATE NEW RESIM MATRIX
    c_zeros_stra = 0
    c_zeros_reve = 0
    resim = np.empty((np.shape(im)[0]-floor(each_side_blanks),np.shape(im)[1]-floor(each_side_blanks)))
    #print('shape(resim)',np.shape(resim))
    #===========================================================================
    # FOR THIS SET, TYPICALLY SIDES ARE ~12, SO WE FORCE IT (CHECK ALWAYS THE PRINT BEFORE
    #===========================================================================
    resim = np.empty((SIZE, SIZE))

    sidediv = SIDEDIV #FROM 3 TO 5
    counti = 0
    for i in range(np.shape(im)[0]-sidediv):
        if i > each_side_blanks/sidediv and i < np.shape(im)[0] - each_side_blanks/sidediv:
            
            countj = 0
            for j in range(np.shape(im)[1]-sidediv):
                if countj >= np.shape(resim)[1]:
                    break
                
                if j > each_side_blanks/sidediv and j < np.shape(im)[1] - each_side_blanks/sidediv:
                    resim[counti][countj] = (im[i+1][j+1]+1.5*im[i+sidediv][j]+1.5*im[i][j+sidediv]+2*im[i+sidediv][j+sidediv]) / 6
                    #print(max(im[i+1][j+1],im[i+sidediv][j],im[i][j+sidediv],im[i+sidediv][j+sidediv]),(0.00123456 + min(im[i+1][j+1],im[i+sidediv][j],im[i][j+sidediv],im[i+sidediv][j+sidediv])))
                    #resim[counti][countj] = log2(0.00123456 + max(im[i+1][j+1],im[i+sidediv][j],im[i][j+sidediv],im[i+sidediv][j+sidediv])/(0.00123456 + min(im[i+1][j+1],im[i+sidediv][j],im[i][j+sidediv],im[i+sidediv][j+sidediv])))
                    countj += 1

            counti += 1
            if counti >= np.shape(resim)[0]:
                break
    
    if plotimages == True:
        plt.imshow(resim, cmap = plt.cm.binary)
        plt.show()

    return resim





data_sets = read_data_sets("MNIST_data")
print('shape(data_sets.train.images):',np.shape(data_sets.train.images),'shape(data_sets.train.labels):',np.shape(data_sets.train.labels))


def main(arraynumbers, start_index, max_data, NUMCLASSES=3):
    
    storei = 999
    DATASET_1 = []
    LABELS_1 = []
    count = 0
    for i in range(5000):
        if data_sets.train.labels[i] == arraynumbers[0]:
            if i < start_index:
                continue
            DATASET_1.append(REDUCEPIXELSTOTHEMAX(i))
            LABELS_1.append(data_sets.train.labels[i])
            count += 1
        if count >= max_data:
            break
    print('shape(DATASET_1)',np.shape(DATASET_1),'np.shape(LABELS_1)',np.shape(LABELS_1))
    storei = i


    DATASET_2 = []
    LABELS_2 = []
    count = 0
    for i in range(5000):
        if data_sets.train.labels[i] ==  arraynumbers[1]:
            if i == storei or i < start_index:
                continue
            DATASET_2.append(REDUCEPIXELSTOTHEMAX(i))
            LABELS_2.append(data_sets.train.labels[i])
            count += 1
        if count >= max_data:
            break
    print('shape(DATASET_2)',np.shape(DATASET_2),'np.shape(LABELS_2)',np.shape(LABELS_2))
    
    
    if len(arraynumbers) > 2:
        
        DATASET_3 = []
        LABELS_3 = []
        count = 0
        for i in range(5000):
            if data_sets.train.labels[i] ==  arraynumbers[2]:
                if i == storei or i < start_index:
                    continue
                DATASET_3.append(REDUCEPIXELSTOTHEMAX(i))
                LABELS_3.append(data_sets.train.labels[i])
                count += 1
            if count >= max_data:
                break
        print('shape(DATASET_3)',np.shape(DATASET_3),'np.shape(LABELS_3)',np.shape(LABELS_3))
    
    
    #===============================================================================
    # #===============================================================================
    # # CHECK
    # #===============================================================================
    # randx = np.random.randint(max_data)
    # plt.imshow(DATASET_1[randx], cmap = plt.cm.binary)
    # plt.show()
    # print(LABELS_1[randx])
    # 
    # randx = np.random.randint(max_data)
    # plt.imshow(DATASET_3[randx], cmap = plt.cm.binary)
    # plt.show()
    # print(LABELS_3[randx])
    #===============================================================================
    
    if NUMCLASSES == 3:
    
        MERGED_MATRIX = np.append(DATASET_1, DATASET_2, axis=0)
        MERGED_MATRIX = np.append(MERGED_MATRIX, DATASET_3, axis=0)
        MERGED_LABELS = np.append(LABELS_1, LABELS_2, axis=0)
        MERGED_LABELS = np.append(MERGED_LABELS, LABELS_3, axis=0)
        print('shape(MERGED_MATRIX)',np.shape(MERGED_MATRIX))
        #randompick = 133
        #plt.imshow(MERGED_MATRIX[randompick], cmap = plt.cm.binary)
        #plt.show()
        #print(MERGED_LABELS[randompick])
        
        
        unitfinalsamplingsize = max_data
        FLATTEN_MERG_MATR = []
        for i in range(unitfinalsamplingsize):  # COLLECT INTO FLATTENED 100 (10x10) ARRAY 1
            FLATTEN_MERG_MATR.append(MERGED_MATRIX[i].flatten())
        for i in range(unitfinalsamplingsize, unitfinalsamplingsize*2):  # COLLECT INTO FLATTENED 100 (10x10) ARRAY 2
            FLATTEN_MERG_MATR.append(MERGED_MATRIX[i].flatten())
        for i in range(np.shape(MERGED_MATRIX)[0]-1,np.shape(MERGED_MATRIX)[0]-1-unitfinalsamplingsize,-1):  # COLLECT INTO FLATTENED 100 (10x10) ARRAY 3
            FLATTEN_MERG_MATR.append(MERGED_MATRIX[i].flatten())
    
    
    elif NUMCLASSES == 2:
    
        MERGED_MATRIX = np.append(DATASET_1, DATASET_2, axis=0)
        MERGED_LABELS = np.append(LABELS_1, LABELS_2, axis=0)
        print('shape(MERGED_MATRIX)',np.shape(MERGED_MATRIX))
        #randompick = 133
        #plt.imshow(MERGED_MATRIX[randompick], cmap = plt.cm.binary)
        #plt.show()
        #print(MERGED_LABELS[randompick])
        
        
        unitfinalsamplingsize = max_data
        FLATTEN_MERG_MATR = []
        for i in range(unitfinalsamplingsize):  # COLLECT INTO FLATTENED 100 (10x10) ARRAY 1
            FLATTEN_MERG_MATR.append(MERGED_MATRIX[i].flatten())
        for i in range(np.shape(MERGED_MATRIX)[0]-1,np.shape(MERGED_MATRIX)[0]-1-unitfinalsamplingsize,-1):  # COLLECT INTO FLATTENED 100 (10x10) ARRAY 3
            FLATTEN_MERG_MATR.append(MERGED_MATRIX[i].flatten())    
    
    
    print('shape(FLATTEN_MERG_MATR)',np.shape(FLATTEN_MERG_MATR))



    return MERGED_MATRIX, MERGED_LABELS, FLATTEN_MERG_MATR



#main(arraynumbers=MNIST_NUMBERS, start_index=0, max_data=50)