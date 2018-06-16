#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/product_layer.hpp"
#include "caffe/util/math_functions.hpp"



namespace caffe {

template <typename Dtype>
__global__ void ProductForward(const int nthreads, const int dim, const Dtype* w, const Dtype* bottom_data, Dtype* top_data){
  CUDA_KERNEL_LOOP(index, nthreads){
    int i = index / dim;
    top_data[index] += bottom_data[index] * w[i];
  }
}
template <typename Dtype>
void ProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count;
  int dim;
  for(int i = 0; i < bottom_pair_num_; i++){
    dim = bottom[i*2]->count(1);
    count = bottom[i*2]->count();
    ProductForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, dim, bottom[i*2+1]->gpu_data(), bottom[i*2]->gpu_data(), top[0]->mutable_gpu_data());
  }
}


template <typename Dtype>
__global__ void ProductBackward(const int nthreads, const int dim, const Dtype* w, const Dtype* bottom_data, const Dtype* top_diff, Dtype* bottom_diff, Dtype* w_diff){
  CUDA_KERNEL_LOOP(index, nthreads){
    int i = index / dim;
    bottom_diff[index] += top_diff[index] * w[i];
    w_diff[i] += top_diff[index] * bottom_data[index];
  }
}
template <typename Dtype>
void ProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int count;
  int dim;

  for(int i = 0; i < bottom_pair_num_; i++){
    dim = bottom[i*2]->count(1);
    count = bottom[i*2]->count();
    ProductBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, dim, bottom[i*2+1]->gpu_data(), bottom[i*2]->gpu_data(), top[0]->gpu_diff(), bottom[i*2]->mutable_gpu_diff(), bottom[i*2+1]->mutable_gpu_diff());
    caffe_gpu_scal(bottom[i*2+1]->count(), loss_weight_, bottom[i*2+1]->mutable_gpu_diff());
    caffe_gpu_scal(count, Dtype(-1), bottom[i*2]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ProductLayer);

}  // namespace caffe
