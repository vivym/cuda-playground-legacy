#include <torch/extension.h>
#include <thrust/device_ptr.h>
#include <magma_auxiliary.h>
#include <magma_svbatched.h>
#include <magma_dvbatched.h>
#include "vbmm/vbmm.h"
#include "thrust_allocator.h"

namespace cuda_playground {
namespace vbmm {

namespace magma {
  template <typename scalar_t>
  void magmablas_gemm_vbatched(
      magma_trans_t transA, magma_trans_t transB, 
      magma_int_t* m, magma_int_t* n, magma_int_t* k,
      scalar_t alpha,
      scalar_t const * const * dA_array, magma_int_t* ldda,
      scalar_t const * const * dB_array, magma_int_t* lddb,
      scalar_t beta,
      scalar_t **dC_array, magma_int_t* lddc, 
      magma_int_t batch_count, magma_queue_t queue);

  template <>
  void magmablas_gemm_vbatched<float>(
      magma_trans_t transA, magma_trans_t transB, 
      magma_int_t* m, magma_int_t* n, magma_int_t* k,
      float alpha,
      float const * const * dA_array, magma_int_t* ldda,
      float const * const * dB_array, magma_int_t* lddb,
      float beta,
      float **dC_array, magma_int_t* lddc, 
      magma_int_t batch_count, magma_queue_t queue) {
    magmablas_sgemm_vbatched(
      transA, transB,
      m, n, k,
      alpha,
      dA_array, ldda,
      dB_array, lddb,
      beta,
      dC_array, lddc,
      batch_count, queue);
  }

  template <>
  void magmablas_gemm_vbatched<double>(
      magma_trans_t transA, magma_trans_t transB, 
      magma_int_t* m, magma_int_t* n, magma_int_t* k,
      double alpha,
      double const * const * dA_array, magma_int_t* ldda,
      double const * const * dB_array, magma_int_t* lddb,
      double beta,
      double **dC_array, magma_int_t* lddc, 
      magma_int_t batch_count, magma_queue_t queue) {
    magmablas_dgemm_vbatched(
      transA, transB,
      m, n, k,
      alpha,
      dA_array, ldda,
      dB_array, lddb,
      beta,
      dC_array, lddc,
      batch_count, queue);
  }
}

template <typename scalar_t, typename index_t>
void vbmm_cuda_magma_impl(
    const VBMatrices& A,
    const VBMatrices& B,
    VBMatrices& C,
    float alpha, float beta,
    bool transA, bool transB) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ThrustAllocator allocator;
  auto policy = thrust::cuda::par(allocator).on(stream);

  auto batch_size = A.get_batch_size();
  auto options = A.get_data().options();
  
  const at::Tensor &m = transA ? A.get_n() : A.get_m();
  const at::Tensor &k = transA ? A.get_m() : A.get_n();
  const at::Tensor &n = transB ? B.get_m() : B.get_n();

  if (!C.is_defined()) {
    C.init(batch_size, A.get_m(), B.get_n(), options);
  }

  auto dA_array = A.get_addresses();
  auto dB_array = B.get_addresses();
  auto dC_array = C.get_addresses();

  auto dA_array_ptr = reinterpret_cast<scalar_t **>(dA_array.template data_ptr<int64_t>());
  auto dB_array_ptr = reinterpret_cast<scalar_t **>(dB_array.template data_ptr<int64_t>());
  auto dC_array_ptr = reinterpret_cast<scalar_t **>(dC_array.template data_ptr<int64_t>());

  auto d_m = m.data_ptr<index_t>();
  auto d_k = k.data_ptr<index_t>();
  auto d_n = n.data_ptr<index_t>();

  auto ldda = transA ? d_m : d_k;
  auto lddb = transB ? d_k : d_n;

  magma_device_t device;
  magma_queue_t queue;
  magma_getdevice(&device);
  magma_queue_create(device, &queue);

  auto magma_transA = transA ? MagmaTrans : MagmaNoTrans;
  auto magma_transB = transB ? MagmaTrans : MagmaNoTrans;
  magma::magmablas_gemm_vbatched<scalar_t>(
      magma_transB, magma_transA,
      d_n, d_m, d_k,
      alpha,   // alpha
      dB_array_ptr,
      lddb,
      dA_array_ptr,
      ldda,
      beta,    // beta
      dC_array_ptr,
      d_n,
      batch_size,
      queue);

  magma_queue_destroy(queue);
}

void vbmm_cuda_magma(
    const VBMatrices& A,
    const VBMatrices& B,
    VBMatrices& C,
    float alpha, float beta,
    bool transA, bool transB) {
  AT_DISPATCH_FLOATING_TYPES(A.get_scalar_type(), "vbmm_cuda_magma", [&] {
    vbmm_cuda_magma_impl<scalar_t, VBMatrices::index_t>(
        A, B, C, alpha, beta, transA, transB);
  });
}

} // namespace vbmm

} // namespace cuda_playground
