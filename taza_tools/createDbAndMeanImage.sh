rm -rf wide_eyes_tech/train_lmdb
rm -rf wide_eyes_tech/test_lmdb

rm wide_eyes_tech/train_image_mean_lmdb.binaryproto

build/tools/convert_imageset -backend lmdb -resize_height 227 -resize_width 227 . shuffled_train_cifashionDB.txt wide_eyes_tech/train_lmdb
build/tools/convert_imageset -backend lmdb -resize_height 227 -resize_width 227 . shuffled_test_cifashionDB.txt wide_eyes_tech/test_lmdb

build/tools/compute_image_mean -backend lmdb wide_eyes_tech/train_lmdb wide_eyes_tech/train_image_mean_lmdb.binaryproto
