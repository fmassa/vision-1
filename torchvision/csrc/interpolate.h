#pragma once
#include <torch/extension.h>
#include <ATen/DimVector.h>
#include <ATen/native/TensorIterator.h>


template <typename scalar_t, typename func_t>
void cpu_interpolate_kernel(at::TensorIterator& iter, int64_t indexed_stride,
                      const func_t& f, bool serial_execution=false)
{
  //int ntensor = iter.ntensors();
  // When launch the index parallel version, set a relative samll grain size less than the INTERNAL::GRAIN_SIZE
  // to make the whole available thread numbers get more balanced work load and a better cache location.
  // The grain size here is chosen by the op benchmark to overcome the thread launch overhead
  const int index_parallel_grain_size = 3000;
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    //auto indexer = Indexer(ntensor - 2, &data[2], &strides[2], index_size, index_stride);
    char* dst = data[0];
    char* src = data[1];
    char* idx0 = data[2];
    char* idx1 = data[3];
    char* w0lambda = data[4];
    char* w1lambda = data[5];
    //std::cout << "now at\n";
    
    for (int64_t i = 0; i < n; i++) {
      //int64_t offset = indexer.get(i);

      //std::cout << i << " " << n << " " << strides[2] << " " << indexed_stride << "\n";
      int64_t offset1 = *(int64_t*)&idx0[i * strides[2]] * indexed_stride;
      int64_t offset2 = *(int64_t*)&idx1[i * strides[3]] * indexed_stride;
      //std::cout << offset1 << " " << offset2 << "\n";
      f(dst + strides[0] * i, src + strides[1] * i, offset1, offset2, w0lambda + strides[4] * i, w1lambda + strides[5] * i);
    }
  };
  //std::cout << "entereing here\n";
  if (serial_execution) {
    iter.serial_for_each(loop, {0, iter.numel()});
  } else {
    iter.for_each(loop, index_parallel_grain_size);
  }
}

//template <typename scalar_t>
//at::Tensor interpolate_cpu_kernel(
at::Tensor interpolate_cpu(
    const at::Tensor& input,
    int64_t size,
    int64_t dim){
  
  auto shape = at::DimVector(input.sizes());
  shape[dim] = size;
  auto strides = at::DimVector(input.strides());
  strides[dim] = 0;

  auto restrided_input = input.as_strided(shape, strides);
  
  auto i = input.size(dim);
  auto scale = float(i) / size;
  auto wr = ((at::arange(size).to(at::kFloat) + 0.5) * scale - 0.5).clamp(0);
  auto w = wr.floor();
  auto wlambda = wr - w;
  auto w0lambda = 1 - wlambda;
  auto idx0 = w.to(at::kLong);
  auto idx1 = (idx0 + 1).clamp(0, i-1);

  auto s = at::DimVector(input.dim(), 1);
  s[dim] = -1;
  w0lambda = w0lambda.reshape(s);
  wlambda = wlambda.reshape(s);

  idx0 = idx0.reshape(s);
  idx1 = idx1.reshape(s);

  //auto output = at::empty(shape, input.options());
  //return output;
  
  at::TensorIteratorConfig config;
  config.check_all_same_dtype(false)
        .declare_static_dtype_and_device(input.scalar_type(), input.device())
        .add_output(at::Tensor())
        .add_input(restrided_input);

  config.add_input(idx0);
  config.add_input(idx1);
  config.add_input(w0lambda);
  config.add_input(wlambda);

  auto iter = config.build();

  int64_t element_size_bytes = input.element_size();
  int64_t indexed_stride = input.stride(dim) * element_size_bytes;
  //std::cout << indexed_stride << " " << idx0 << "\n";

  AT_DISPATCH_ALL_TYPES(
    iter.dtype(), "index_cpu", [&] {
    cpu_interpolate_kernel<scalar_t>(iter, indexed_stride, [](char* dst, char* src, int64_t offset1, int64_t offset2, char* v0, char* v1) {
      //std::cout << offset1 << " " <<  *(scalar_t*)v0 << " " << offset2 << " " <<  *(scalar_t*)v1 << "\n";
      *(scalar_t*)dst = *(scalar_t*)(src + offset1) * *(scalar_t*)v0 + *(scalar_t*)(src + offset2) * *(scalar_t*)v1;
    });
  });
  //std::cout << "here\n";

  return iter.output();
  //return input.index_select(dim, idx0) * w0lambda + input.index_select(dim, idx1) * wlambda;
}

/*
at::Tensor interpolate_cpu(
    const at::Tensor& input,
    int64_t size,
    int64_t dim) {
  auto result = at::empty({0}, input.options());

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "interpolate", [&] {
    result = interpolate_cpu_kernel<scalar_t>(input, size, dim);
  });
  return result;
}*/
