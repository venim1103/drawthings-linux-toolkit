#include "nnc/ccv_nnc_8i_rowwise.c"

__attribute__((visibility("default")))
size_t dt_moss_i4x_encode(const void* input, const size_t rows, const size_t columns,
  void* output, const size_t capacity)
{
  if (!input || !output || !rows || !columns || columns % 256 ||
      columns > 33554432 || rows > 33554432 / columns)
    return 0;
  if (capacity < ccv_nnc_8i_rowwise_x_data_size(CCV_NNC_QX_8I_ROWWISE_Q4_K,
      CCV_16F, rows * columns, columns))
    return 0;
  return ccv_nnc_quantize_8i_rowwise_x(input, CCV_16F, CCV_TENSOR_CPU_MEMORY,
    rows * columns, columns,
    CCV_NNC_QX_8I_ROWWISE_Q4_K | CCV_NNC_QX_8I_ROWWISE_HADAMARD_256,
    0, 0, output, capacity);
}

__attribute__((visibility("default")))
int dt_moss_i4x_decode_rotated(const void* input, const size_t length,
  const size_t rows, const size_t columns, void* output)
{
  if (!input || !output || !length || !rows || !columns || columns % 256 ||
      columns > 33554432 || rows > 33554432 / columns)
    return -1;
  if (length != ccv_nnc_8i_rowwise_x_data_size(CCV_NNC_QX_8I_ROWWISE_Q4_K,
      CCV_16F, rows * columns, columns))
    return -1;
  ccv_nnc_dequantize_8i_rowwise_x(input, CCV_16F, CCV_TENSOR_CPU_MEMORY,
    length, columns, CCV_NNC_QX_8I_ROWWISE_Q4_K, output, rows * columns);
  return 0;
}
