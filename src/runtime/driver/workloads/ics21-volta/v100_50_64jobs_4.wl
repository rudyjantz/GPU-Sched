/home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet classifier train /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/cifar-cporter.data /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/cifar_small.cfg
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/srad/srad_v1/srad 100 0.5 11000 11000
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/nw/needle 16384 10
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/srad/srad_v2/srad 8192 8192 0 127 0 127 0.5 2
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/dwt2d/dwt2d /home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/data/dwt2d/rgb.bmp -d 8192x8192 -f -5 -l 3
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/lavaMD/lavaMD -boxes1d 120
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/srad/srad_v2/srad 16384 16384 0 127 0 127 0.5 2
cat /home/ubuntu/GPU-Sched/Benchmarks/darknet/image-names-medium-full-v100.txt | /home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet detect /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/yolov3-tiny.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/yolov3-tiny.weights
/home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet classifier train /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/cifar-cporter.data /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/cifar_small.cfg
/home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet rnn generate /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/rnn.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/shakespeare.weights -len 100000
cat /home/ubuntu/GPU-Sched/Benchmarks/darknet/image-names-large-full-v100.txt | /home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet classifier predict /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/imagenet1k-cporter.data /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/darknet19.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/darknet19.weights
cat /home/ubuntu/GPU-Sched/Benchmarks/darknet/image-names-medium-full-v100.txt | /home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet detect /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/yolov3-tiny.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/yolov3-tiny.weights
/home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet classifier train /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/cifar-cporter.data /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/cifar_small.cfg
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/srad/srad_v2/srad 8192 8192 0 127 0 127 0.5 2
cat /home/ubuntu/GPU-Sched/Benchmarks/darknet/image-names-large-full-v100.txt | /home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet classifier predict /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/imagenet1k-cporter.data /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/darknet19.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/darknet19.weights
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/srad/srad_v1/srad 100 0.5 15000 15000
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/backprop/backprop 16777216
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/srad/srad_v1/srad 100 0.5 20000 20000
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/srad/srad_v1/srad 100 0.5 20000 20000
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/nw/needle 16384 10
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/srad/srad_v2/srad 16384 16384 0 127 0 127 0.5 2
cat /home/ubuntu/GPU-Sched/Benchmarks/darknet/image-names-medium-full-v100.txt | /home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet detect /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/yolov3-tiny.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/yolov3-tiny.weights
/home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet rnn generate /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/rnn.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/shakespeare.weights -len 100000
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/nw/needle 16384 10
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/srad/srad_v2/srad 16384 16384 0 127 0 127 0.5 2
cat /home/ubuntu/GPU-Sched/Benchmarks/darknet/image-names-large-full-v100.txt | /home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet classifier predict /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/imagenet1k-cporter.data /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/darknet19.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/darknet19.weights
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/backprop/backprop 16777216
cat /home/ubuntu/GPU-Sched/Benchmarks/darknet/image-names-large-full-v100.txt | /home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet classifier predict /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/imagenet1k-cporter.data /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/darknet19.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/darknet19.weights
cat /home/ubuntu/GPU-Sched/Benchmarks/darknet/image-names-medium-full-v100.txt | /home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet detect /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/yolov3-tiny.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/yolov3-tiny.weights
/home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet classifier train /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/cifar-cporter.data /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/cifar_small.cfg
/home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet rnn generate /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/rnn.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/shakespeare.weights -len 100000
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/dwt2d/dwt2d /home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/data/dwt2d/rgb.bmp -d 8192x8192 -f -5 -l 3
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/srad/srad_v2/srad 16384 16384 0 127 0 127 0.5 2
/home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet classifier train /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/cifar-cporter.data /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/cifar_small.cfg
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/backprop/backprop 8388608
/home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet rnn generate /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/rnn.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/shakespeare.weights -len 100000
/home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet classifier train /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/cifar-cporter.data /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/cifar_small.cfg
/home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet rnn generate /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/rnn.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/shakespeare.weights -len 100000
cat /home/ubuntu/GPU-Sched/Benchmarks/darknet/image-names-large-full-v100.txt | /home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet classifier predict /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/imagenet1k-cporter.data /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/darknet19.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/darknet19.weights
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/srad/srad_v2/srad 16384 16384 0 127 0 127 0.5 2
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/nw/needle 16384 10
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/srad/srad_v2/srad 16384 16384 0 127 0 127 0.5 2
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/nw/needle 16384 10
cat /home/ubuntu/GPU-Sched/Benchmarks/darknet/image-names-medium-full-v100.txt | /home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet detect /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/yolov3-tiny.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/yolov3-tiny.weights
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/srad/srad_v2/srad 16384 16384 0 127 0 127 0.5 2
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/backprop/backprop 67108864
cat /home/ubuntu/GPU-Sched/Benchmarks/darknet/image-names-large-full-v100.txt | /home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet classifier predict /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/imagenet1k-cporter.data /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/darknet19.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/darknet19.weights
cat /home/ubuntu/GPU-Sched/Benchmarks/darknet/image-names-medium-full-v100.txt | /home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet detect /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/yolov3-tiny.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/yolov3-tiny.weights
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/nw/needle 32768 10
cat /home/ubuntu/GPU-Sched/Benchmarks/darknet/image-names-medium-full-v100.txt | /home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet detect /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/yolov3-tiny.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/yolov3-tiny.weights
/home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet rnn generate /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/rnn.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/shakespeare.weights -len 100000
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/lavaMD/lavaMD -boxes1d 110
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/srad/srad_v1/srad 100 0.5 11000 11000
/home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet rnn generate /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/rnn.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/shakespeare.weights -len 100000
/home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet classifier train /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/cifar-cporter.data /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/cifar_small.cfg
/home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet rnn generate /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/rnn.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/shakespeare.weights -len 100000
cat /home/ubuntu/GPU-Sched/Benchmarks/darknet/image-names-large-full-v100.txt | /home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet classifier predict /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/imagenet1k-cporter.data /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/darknet19.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/darknet19.weights
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/srad/srad_v1/srad 100 0.5 11000 11000
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/dwt2d/dwt2d /home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/data/dwt2d/rgb.bmp -d 16384x16384 -f -5 -l 3
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/lavaMD/lavaMD -boxes1d 120
/home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet classifier train /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/cifar-cporter.data /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/cifar_small.cfg
cat /home/ubuntu/GPU-Sched/Benchmarks/darknet/image-names-large-full-v100.txt | /home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet classifier predict /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/imagenet1k-cporter.data /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/darknet19.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/darknet19.weights
cat /home/ubuntu/GPU-Sched/Benchmarks/darknet/image-names-medium-full-v100.txt | /home/ubuntu/GPU-Sched/Benchmarks/darknet/darknet detect /home/ubuntu/GPU-Sched/Benchmarks/darknet/cfg/yolov3-tiny.cfg /home/ubuntu/GPU-Sched/Benchmarks/darknet/weights/yolov3-tiny.weights
/home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/backprop/backprop 16777216