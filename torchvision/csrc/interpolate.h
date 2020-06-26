#pragma once
#include <torch/extension.h>
#include <ATen/DimVector.h>
#include <ATen/native/TensorIterator.h>


struct Indexer {
  Indexer(int64_t num_indexers, char** indexers, const int64_t* indexer_strides,
          IntArrayRef original_strides)
    : num_indexers(num_indexers)
    , indexers(indexers)
    , indexer_strides(indexer_strides)
    , original_strides(original_strides.data()) {
  }

  int64_t num_indexers;
  char** indexers;
  const int64_t* indexer_strides;
  const int64_t* original_strides;

  int64_t get(int64_t idx) {
    int64_t offset = 0;
    for (int j = 0; j < num_indexers; j++) {
      int64_t value = *(int64_t*)&indexers[j][idx * indexer_strides[j]];
      offset += value * original_strides[j];
    }
    return offset;
  }
};



template <typename scalar_t, typename func_t>
void cpu_interpolate_kernel(at::TensorIterator& iter, std::vector<int64_t> indexed_strides,
                      const func_t& f, bool serial_execution=false)
{
  int ntensor = iter.ntensors();
  int n_dims = (ntensor - 2) / 4;
  // When launch the index parallel version, set a relative samll grain size less than the INTERNAL::GRAIN_SIZE
  // to make the whole available thread numbers get more balanced work load and a better cache location.
  // The grain size here is chosen by the op benchmark to overcome the thread launch overhead
  const int index_parallel_grain_size = 3000;
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    //auto indexer = Indexer(ntensor - 2, &data[2], &strides[2], index_size, index_stride);
    auto indexer1 = Indexer(n_dims, &data[2], &strides[2], indexed_strides[0]);
    auto indexer2 = Indexer(n_dims, &data[4], &strides[4], indexed_strides[1]);
    char* dst = data[0];
    char* src = data[1];
    char* idx0 = data[2];
    char* idx1 = data[3];
    char* w0lambda = data[4];
    char* w1lambda = data[5];
    
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


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_helper(const at::Tensor& input,
        int64_t size,
        int64_t dim) {
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

  return {idx0, idx1, w0lambda, wlambda};
}

//template <typename scalar_t>
//at::Tensor interpolate_cpu_kernel(
at::Tensor interpolate_cpu(
    const at::Tensor& input,
    std::vector<int64_t> sizes,
    std::vector<int64_t> dims){
  
  auto shape = at::DimVector(input.sizes());
  auto strides = at::DimVector(input.strides());

  for(int64_t i=0; i < dims.size(); i++){
    strides[dims[i]] = 0;
    shape[dims[i]] = sizes[i];
  }
  /*
  for(auto dim : dims)
    strides[dim] = 0;
  for(auto size: sizes):
    shape[dim] = size;
  */
  auto restrided_input = input.as_strided(shape, strides);
  
  std::vector<at::Tensor> idx0, idx1, w0lambda, wlambda;
  for(int64_t i=0; i < dims.size(); i++) {
    auto res = _helper(input, sizes[i], dims[i]);
    idx0.push_back(std::get<0>(res));
    idx1.push_back(std::get<1>(res));
    w0lambda.push_back(std::get<2>(res));
    wlambda.push_back(std::get<3>(res));
  }

  at::TensorIteratorConfig config;
  config.check_all_same_dtype(false)
        .declare_static_dtype_and_device(input.scalar_type(), input.device())
        .add_output(at::Tensor())
        .add_input(restrided_input);

  for(auto& i : idx0)
    config.add_input(i);
  for(auto& i : idx1)
    config.add_input(i);
  for(auto& i : w0lambda)
    config.add_input(i);
  for(auto& i : wlambda)
    config.add_input(i);

  auto iter = config.build();

  int64_t element_size_bytes = input.element_size();
  std::vector<int64_t> indexed_strides(dims.size());
  for(int64_t i=0; i < dims.size(); i++)
    indexed_strides[i] = input.stride(dims[i]) * element_size_bytes;

  AT_DISPATCH_ALL_TYPES(
    iter.dtype(), "index_cpu", [&] {
    cpu_interpolate_kernel<scalar_t>(iter, indexed_strides, [](char* dst, char* src, int64_t offset1, int64_t offset2, char* v0, char* v1) {
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
