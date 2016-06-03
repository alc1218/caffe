rm -rf wide_eyes_tech/train
rm -rf wide_eyes_tech/test

rm wide_eyes_tech/train_image_mean.binaryproto

build/tools/convert_imageset -backend lmdb -resize_height 227 -resize_width 227 . shuffled_train_cifashionDB.txt wide_eyes_tech/train
build/tools/convert_imageset -backend lmdb -resize_height 227 -resize_width 227 . shuffled_test_cifashionDB.txt wide_eyes_tech/test

build/tools/compute_image_mean -backend lmdb wide_eyes_tech/train wide_eyes_tech/train_image_mean.binaryproto
