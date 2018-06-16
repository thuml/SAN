#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/aggregate_weight_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void AggregateWeightForward(const int nthreads, const int dim, const int data_num, const Dtype* source, Dtype* target) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int data_id = index / dim;
    int label_id = index % dim;
    target[label_id * data_num + data_id] = source[index];
  }
}

template <typename Dtype>
__global__ void AggregateWeightForClass(const int nthreads, const int data_num, const Dtype* source, Dtype* target) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    for (int i = data_num / 2; i < data_num; i++){
      target[index] += source[i*nthreads+index];
    }
  }
}

template <typename Dtype>
void AggregateWeightLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* temp_weight = weight_blob_.mutable_gpu_data();
  Dtype* unnormalized_weight = class_blob_.mutable_gpu_data();
  Dtype* normalized_weight = class_blob_.mutable_gpu_diff();
  caffe_gpu_set(weight_blob_.count(), Dtype(0), temp_weight);
  AggregateWeightForward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(bottom[0]->count(), num_output_, num_data_, bottom_data, temp_weight);
  //AggregateWeightForClass<Dtype><<<CAFFE_GET_BLOCKS(num_output_), CAFFE_CUDA_NUM_THREADS>>>(num_output_, num_data_, bottom_data, temp_weight);
  //caffe_gpu_memcpy(sizeof(Dtype)*num_output_, temp_weight, weight_blob_.mutable_cpu_data());
  //Dtype sum_;
  //caffe_gpu_asum(num_output_, temp_weight, &sum_);
  for(int i = 0; i < num_output_; i++){
    //weight_blob_.mutable_cpu_data()[i] =weight_blob_.mutable_cpu_data()[i] * 10 / sum_;
    caffe_gpu_memcpy(sizeof(Dtype)*num_data_, temp_weight+i*num_data_, top[i]->mutable_gpu_data());
  }

  AggregateWeightForClass<Dtype><<<CAFFE_GET_BLOCKS(num_output_), CAFFE_CUDA_NUM_THREADS>>>(num_output_, num_data_, bottom_data, unnormalized_weight);
  Dtype sum_weight;
  caffe_gpu_asum(class_blob_.count(), unnormalized_weight, &sum_weight);

  caffe_gpu_memcpy(class_blob_.count() * sizeof(Dtype), unnormalized_weight, normalized_weight);
  caffe_gpu_scal(class_blob_.count(), Dtype(num_output_) / sum_weight, normalized_weight);
  
  for(int i = 0; i < num_output_; i++){
    caffe_gpu_memcpy(sizeof(Dtype), normalized_weight+i, top[i]->mutable_gpu_data()+num_data_);
  }
  //caffe_gpu_memcpy(sizeof(Dtype) * weight_blob_.count(), temp_weight, weight_blob_.mutable_cpu_data());
  /*Dtype sum_;
  for(int i = 0; i < num_data_; i++){
    sum_ = 0;
    for(int j = 0; j < num_output_; j++){
      sum_ += weight_blob_.cpu_data()[j*num_data_+i];
    }
    LOG(INFO) << "sum: " << sum_;
  }*/
}

template <typename Dtype>
void AggregateWeightLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AggregateWeightLayer);

}  // namespace caffe
