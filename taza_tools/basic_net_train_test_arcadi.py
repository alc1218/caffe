__author__ = "Arcadi Llanza Carmona"
__version__ = "1.0"

cluster = True

if cluster:
    # CLUSTER PARAMS
    root = '/home/arcadi/'
    caffe_root = root + 'libraries/caffe/'  # this file should be run from {caffe_root}/
    wide_eyes_folder = root + 'wide_eyes_tech/'
    trData_filename = wide_eyes_folder + 'shuffled_train_cifashionDB.txt'
    teData_filename = wide_eyes_folder + 'shuffled_test_cifashionDB.txt'
    solver_config_path = wide_eyes_folder + 'solver_alexnet.prototxt'
    image_mean_filename = 'custom_image_mean.png'
else:
    # MAC PARAMS
    root = '/Users/arcadillanzacarmona/' 
    caffe_root = root + 'caffe_github/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
    wide_eyes_folder = root + 'caffe_github/wide_eyes_tech/'
    trData_filename = caffe_root + 'shuffled_train_cifashionDB.txt'
    teData_filename = caffe_root + 'shuffled_test_cifashionDB.txt'
    solver_config_path = wide_eyes_folder + 'bvlc_alexnet/solver.prototxt'
    image_mean_filename = 'custom_image_mean.png'

import multiprocessing as mtp
from pylab import *
import pylab
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
import numpy as np
import cv2
import random as random
from PIL import Image
import os.path
import itertools

# GLOBAL VARIABLES
prob_data_augmentation = 0.5
prob_rotation = 0.25
prob_crop = 0.50
prob_blur = 0.75
prob_flip = 1.0

# SOLVER PARAMETERS
niter = 100000
test_interval = 5
total_test_iters = 100
batch_size = 64
gpu_id = 0

def main():

    if cluster:
        # CLUSTER
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()
        num_cores = mtp.cpu_count() / 2
    else:
        # MAC
        caffe.set_mode_cpu()
        num_cores = mtp.cpu_count()

    workers = mtp.Pool(num_cores)
    print 'threads: ' + str(num_cores)
    
    ### load the solver and create train and test nets
    solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
    solver = caffe.get_solver(solver_config_path)

    # losses will also be stored in the log
    train_loss = zeros(niter)
    test_acc = zeros(int(np.ceil(niter / test_interval)))

    if cluster:
        # CLUSTER
        trData = [("." +  a.split(' ')[0], a.split(' ')[1]) for a in open(trData_filename)]
        teData = [("." +  a.split(' ')[0], a.split(' ')[1]) for a in open(teData_filename)]
    else:
        # MAC
        trData = [(".." + a.split(' ')[0], a.split(' ')[1]) for a in open(trData_filename)]
        teData = [(".." + a.split(' ')[0], a.split(' ')[1]) for a in open(teData_filename)]

    # SPLIT TRAINING DATA FOR DIFFERENT WORKERS
    #st_tr = range(0,len(trData),len(trData)/num_cores)
    #trData_splited = [trData[s:s+len(trData)/num_cores] for s in st_tr]
    trData_splited = []
    amount_of_trFiles_per_worker = (len(trData)/num_cores)
    
    for idx_worker in range(num_cores):
        trData_splited.append([])
        for idx_image in range(amount_of_trFiles_per_worker):
            trData_splited[idx_worker].append(trData[idx_worker * amount_of_trFiles_per_worker + idx_image][0])
    # TODO: ARCADI: WE WILL HAVE TO APPEND THE LAST trData IMAGES

    if os.path.isfile(image_mean_filename):
        print 'Retrieving an image mean from disc'
        im_mean = cv2.imread(image_mean_filename, 1)
    else:
        print 'Computing an image mean'

        im_mean = np.zeros([227, 227, 3], float32)

        # EXECUTE WORKERS TO STORE IMAGE VALUES
        jobs = workers.map(computeImageMeanMTP, trData_splited)

        # REDUCE WORKERS INFORMATION TO OBTAIN THE IMAGE MEAN
        for index_job in range(jobs):
            im_mean += jobs[index_job][0]

        im_mean[:,:,0] = im_mean[:,:,0] / len(trData)
        im_mean[:,:,1] = im_mean[:,:,1] / len(trData)
        im_mean[:,:,2] = im_mean[:,:,2] / len(trData)

        # SAVE IMAGE MEAN
        #Rescale to 0-255 and convert to uint8
        rescaled = (255.0 / im_mean.max() * (im_mean - im_mean.min())).astype(np.uint8)

        data = Image.fromarray(rescaled)

        data.save(image_mean_filename)

    batch_train_idx = 0
    batch_test_idx = 0

    # the main solver loop
    for it in range(1, niter):
        #print 'Iteration', it, 'training...'

        if (batch_train_idx + 1) * batch_size >= len(trData):
            batch_train_idx = 0
            np.random.shuffle(trData)

        images_train = np.zeros([batch_size, 3, 227, 227], np.float32)
        y_train = np.zeros([batch_size, 1], np.float32)

        # SPLIT BATCH PATHS FOR DIFFERENT WORKERS
        batch_splited = []
        amount_of_trFiles_per_worker = (batch_size/num_cores)

        for idx_worker in range(num_cores):
            batch_splited.append([])
            for idx_image in range(amount_of_trFiles_per_worker):
                batch_splited[idx_worker].append(trData[batch_train_idx * batch_size + idx_worker * amount_of_trFiles_per_worker + idx_image])

        # EXECUTE WORKERS TO READ IMAGES
        jobs = workers.map(packed_readImageMTP, itertools.izip(batch_splited, itertools.repeat(im_mean)))

        # REDUCE WORKERS INFORMATION TO FEED THE NET
        index = 0
        for index_job in range(len(jobs)):
            for index_images in range(len(jobs[index_job][0])):
                images_train[index,:,:,:] = jobs[index_job][0][index_images]
                y_train[index] = jobs[index_job][1][index_images]
                index += 1

        # FEED THE NET WITH IMAGES
        solver.net.blobs['data'].reshape(*(images_train.shape))
        solver.net.blobs['data'].data[:] = images_train[:]

        # FEED THE NET WITH LABELS
        solver.net.blobs['label'].reshape(*(y_train.shape))
        solver.net.blobs['label'].data[:] = y_train[:]

        # COMPUTE ONE STEP OF THE NET (BACKPROPAGATION ALGORITHM)
        solver.step(1)  # SGD by Caffe

        if it % (test_interval + 1) == 0:
            #print 'Iteration', it, 'testing...'

            np.random.shuffle(teData)

            correct_predicted = 0

            for _ in xrange(total_test_iters):

                if (batch_test_idx + 1) * batch_size >= len(teData):
                    batch_test_idx = 0
                    np.random.shuffle(teData)

                images_test = np.zeros([batch_size, 3, 227, 227], np.float32)
                y_test = np.zeros([batch_size, 1], np.float32)

                # SPLIT BATCH PATHS FOR DIFFERENT WORKERS
                batch_splited = []
                amount_of_teFiles_per_worker = (batch_size/num_cores)

                for idx_worker in range(num_cores):
                    batch_splited.append([])
                    for idx_image in range(amount_of_teFiles_per_worker):
                        batch_splited[idx_worker].append(teData[batch_test_idx * batch_size + idx_worker * amount_of_teFiles_per_worker + idx_image])

                # EXECUTE WORKERS TO READ IMAGES
                jobs = workers.map(packed_readTestImageMTP, itertools.izip(batch_splited, itertools.repeat(im_mean)))

                # REDUCE WORKERS INFORMATION TO FEED THE NET
                index = 0
                for index_job in range(len(jobs)):
                    for index_images in range(len(jobs[index_job][0])):
                        images_test[index,:,:,:] = jobs[index_job][0][index_images]
                        y_test[index] = jobs[index_job][1][index_images]
                        index += 1

                solver.net.blobs['data'].reshape(*(images_test.shape))
                solver.net.blobs['data'].data[:] = images_test[:]

                solver.net.blobs['label'].reshape(*(y_test.shape))
                solver.net.blobs['label'].data[:] = y_test[:]

                forward_test = solver.net.forward()
                predictions = solver.net.blobs['fc8'].data.argmax(1) 

                for index, prediction in enumerate(predictions):
                    if y_test[index] == prediction:
                        correct_predicted += 1

                batch_test_idx += 1

            accuracy = float(correct_predicted)/float((total_test_iters + 1) * (batch_size + 1))

            print 'Accuracy: ' + "%.2f" % accuracy

        batch_train_idx += 1

    _, ax1 = subplots()
    ax2 = ax1.twinx()
    ax1.plot(arange(niter), train_loss)
    ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('test accuracy')
    ax2.set_title('Custom Test Accuracy: {:.2f}'.format(test_acc[-1]))

    pylab.show()

def computeImageMeanMTP(data):

    im_mean = np.zeros([227, 227, 3], float32)

    for imFilename in data:

        im = cv2.imread(imFilename)
        im32f = cv2.resize(im, (227,227)).astype(np.float32)

        im_mean[:,:,0] += im32f[:,:,0]
        im_mean[:,:,1] += im32f[:,:,1]
        im_mean[:,:,2] += im32f[:,:,2]

    return im_mean

def packed_readImageMTP(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return readImageMTP(*a_b)

def readImageMTP(data, im_mean):

    images = np.zeros([len(data), 3, 227, 227], np.float32)
    y = np.zeros([len(data), 1], np.float32)

    for idx, sample in enumerate(data):

        imFilename = sample[0]
        y[idx,0] = int(sample[1])

        im = cv2.imread(imFilename, 1)
        im = cv2.resize(im, (227,227)).astype(np.float32)

        # DATA AUGMENTATION
        if random.random() > prob_data_augmentation:
            prob_random = random.random()
            if prob_random <= prob_rotation:

                rows,cols,channels = im.shape

                items = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
                angle = random.sample(items, 1)

                M = cv2.getRotationMatrix2D((cols/2,rows/2),angle[0],1)
                im = cv2.warpAffine(im,M,(cols,rows))

            elif prob_random <= prob_crop:
                
                padding = random.randint(5, 20)

                im = im[padding:im.shape[0]-padding, padding:im.shape[0]-padding]

            elif prob_random <= prob_blur:

                items = [0, 1, 2]
                blur_type = random.sample(items, 1)

                if blur_type[0] == 0: # AVERAGING BLUR

                    ksize = random.randrange(3, 9, 2)  # Odd integer from 3 to 9
                    im = cv2.blur(im,(ksize,ksize))

                elif blur_type[0] == 1: # GAUSSIAN BLUR

                    ksize = random.randrange(3, 9, 2)  # Odd integer from 3 to 9
                    im = cv2.GaussianBlur(im,(ksize,ksize),0)

                elif blur_type[0] == 2: # MEDIAN BLUR

                    ksize = random.randrange(3, 5, 2)  # Odd integer from 3 to 5
                    im = cv2.medianBlur(im,ksize)

            elif prob_random <= prob_flip:
                im = np.fliplr(im)
                
                '''
                #Rescale to 0-255 and convert to uint8
                rescaled = (255.0 / im.max() * (im - im.min())).astype(np.uint8)

                data = Image.fromarray(rescaled)

                data.save("test_mirror.png")

                asdf
                '''


        im32f = cv2.resize(im, (227,227)).astype(np.float32)
        im32f[:,:,0] -= im_mean[:,:,0] #128
        im32f[:,:,1] -= im_mean[:,:,1] #128
        im32f[:,:,2] -= im_mean[:,:,2] #128

        images[idx,:,:,:] = im32f.transpose([2,0,1]) # We are looking for --> [Channel, Width, Height]

    return images, y

def packed_readTestImageMTP(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return readTestImageMTP(*a_b)

def readTestImageMTP(data, im_mean):

    images = np.zeros([len(data), 3, 227, 227], np.float32)
    y = np.zeros([len(data), 1], np.float32)

    for idx, sample in enumerate(data):

        imFilename = sample[0]
        y[idx,0] = int(sample[1])

        im = cv2.imread(imFilename, 1)
        im32f = cv2.resize(im, (227,227)).astype(np.float32)
        im32f[:,:,0] -= im_mean[:,:,0] #128
        im32f[:,:,1] -= im_mean[:,:,1] #128
        im32f[:,:,2] -= im_mean[:,:,2] #128

        images[idx,:,:,:] = im32f.transpose([2,0,1]) # We are looking for --> [Channel, Width, Height]

    return images, y

if __name__ == "__main__":
    main()
