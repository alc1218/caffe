import caffe
import numpy as np
import sys

input_file = 'train60_227x227_image_mean.binaryproto'
output_file = 'train60_227x227_image_mean.npy'

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( input_file , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save( output_file , out )