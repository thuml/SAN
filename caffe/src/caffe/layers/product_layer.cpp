#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  loss_weight_ = this->layer_param_.product_param().loss_weight();
  bottom_pair_num_ = bottom.size() / 2;
}

template <typename Dtype>
void ProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void ProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void ProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
  }
}

#ifdef CPU_ONLY
STUB_GPU(ProductLayer);
#endif

INSTANTIATE_CLASS(ProductLayer);
REGISTER_LAYER_CLASS(Product);

}  // namespace caffe
