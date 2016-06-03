rm -rf wide_eyes_tech/train_leveldb
rm -rf wide_eyes_tech/test_leveldb

rm wide_eyes_tech/train_image_mean_leveldb.binaryproto

build/tools/convert_imageset -backend leveldb -resize_height 227 -resize_width 227 . wide_eyes_tech/cifashion/shuffled_train_cifashionDB.txt wide_eyes_tech/train_leveldb
build/tools/convert_imageset -backend leveldb -resize_height 227 -resize_width 227 . wide_eyes_tech/cifashion/shuffled_test_cifashionDB.txt wide_eyes_tech/test_leveldb

build/tools/compute_image_mean -backend leveldb wide_eyes_tech/train_leveldb wide_eyes_tech/train_image_mean_leveldb.binaryproto
