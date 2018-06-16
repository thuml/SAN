#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/entropy_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ForwardGPU(const int nthreads, const Dtype* prob, 
        const Dtype* log_data, const Dtype threshold, const Dtype prob_pow, Dtype* loss_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if(prob[index] < threshold){
        loss_data[index] = Dtype(0);
    }
    else{
        loss_data[index] = pow(prob[index], prob_pow) * log_data[index];
    }
  }
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* log_data = normalized_bottom_data_.mutable_gpu_data(); 
    
    caffe_gpu_log(data_num_ * label_num_, bottom_data, log_data);

    int nthreads = label_num_ * data_num_;
    Dtype loss;
    Dtype* loss_data = bottom[0]->mutable_gpu_diff();
    ForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), 
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, log_data, Dtype(0.00001), prob_pow_, loss_data);
    caffe_gpu_asum(label_num_ * data_num_, loss_data, &loss);
    loss = -loss;
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void EntropyDiff(const int nthreads, const Dtype* data, 
        const Dtype* log_data, const Dtype threshold, 
        const int data_num, const int ignore_label, const int label_num, 
        const Dtype prob_pow, Dtype* count, Dtype* diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    count[index] = Dtype(1) / label_num;
    if(data[index] < threshold){
        diff[index] = Dtype(0);
    }
    else{
        diff[index] = -(
                        pow(data[index], prob_pow - Dtype(1.0)) + 
                        prob_pow * pow(data[index], prob_pow - Dtype(1.0)) * log_data[index]
                       );
    }
  }
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* log_data = normalized_bottom_data_.mutable_gpu_data();
  Dtype* count = normalized_bottom_data_.mutable_gpu_diff();
  int nthreads = data_num_ * label_num_;

  const Dtype* bottom_data = bottom[0]->gpu_data();
  if (propagate_down[0]) {
      EntropyDiff<Dtype><<<CAFFE_GET_BLOCKS(nthreads), 
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, log_data, 
                threshold_, data_num_, ignore_label_, label_num_, prob_pow_, count, bottom_diff);

      Dtype count_num;
      caffe_gpu_asum(nthreads, count, &count_num);
      count_num = count_num > 0 ? count_num : Dtype(1);
      caffe_gpu_scal(nthreads, loss_weight_ / count_num, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EntropyLossLayer);

}  // namespace caffe
