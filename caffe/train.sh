WEIGHTS=/home/caozhangjie/run-czj/icml-hash/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel

./build/tools/caffe train \
    -solver ./models/san/office/solver.prototxt\
    -weights $WEIGHTS\
    -gpu 6
