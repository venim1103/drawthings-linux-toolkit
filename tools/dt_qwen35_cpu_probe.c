#include <stdlib.h>
#include <string.h>
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"

static void* workspace;

void* ccv_nnc_stream_context_get_workspace(ccv_nnc_stream_context_t* context,
  const size_t size, const int memory)
{
  (void)context;
  assert(memory == CCV_TENSOR_CPU_MEMORY);
  free(workspace);
  workspace = calloc(1, size);
  assert(workspace);
  return workspace;
}

#include "nnc/cmd/gated_delta/ccv_nnc_gated_delta_cpu_ref.c"
#include "nnc/cmd/rmsnorm_gated/ccv_nnc_rmsnorm_gated_cpu_ref.c"

static ccv_nnc_tensor_t tensor(float* data, const int* shape, const int rank)
{
  ccv_nnc_tensor_t result = {0};
  result.data.f32 = data;
  result.info.type = CCV_TENSOR_CPU_MEMORY;
  result.info.format = CCV_TENSOR_FORMAT_NHWC;
  result.info.datatype = CCV_32F;
  memcpy(result.info.dim, shape, sizeof(int) * rank);
  return result;
}

int dt_probe_gated_delta(const int* dimensions, float* queries, float* keys,
  float* values, float* decay, float* beta, float* initial, float* output,
  float* final_state, const int log_decay, const int checkpoint_count,
  const int channel_decay)
{
  const int batch = dimensions[0], sequence = dimensions[1];
  const int key_heads = dimensions[2], value_heads = dimensions[3];
  const int key_dim = dimensions[4], value_dim = dimensions[5];
  const int query_shape[] = {batch, sequence, key_heads, key_dim};
  const int value_shape[] = {batch, sequence, value_heads, value_dim};
  const int gate_shape[] = {batch, sequence, value_heads, key_dim};
  const int state_shape[] = {batch, value_heads, value_dim, key_dim};
  const int final_shape[] = {batch, value_heads * (checkpoint_count + 1), value_dim, key_dim};
  ccv_nnc_tensor_t tensors[] = {
    tensor(queries, query_shape, 4), tensor(keys, query_shape, 4),
    tensor(values, value_shape, 4), tensor(decay, gate_shape, channel_decay ? 4 : 3),
    tensor(beta, gate_shape, 3), tensor(initial, state_shape, 4),
    tensor(output, value_shape, 4), tensor(final_state, final_shape, 4)
  };
  ccv_nnc_tensor_t* inputs[] = {tensors, tensors + 1, tensors + 2, tensors + 3, tensors + 4, tensors + 5};
  ccv_nnc_tensor_t* outputs[] = {tensors + 6, tensors + 7};
  const ccv_nnc_cmd_t command = {.info = {.gated_delta = {
    .log_decay = log_decay, .state_checkpoint_count = checkpoint_count}}};
  const ccv_nnc_hint_t hint = {0};
  return _ccv_nnc_gated_delta_forw(command, hint, 0, inputs, 6, outputs, 2, 0);
}

int dt_probe_rmsnorm(const int rows, const int columns, float* values,
  float* gate, float* weight, float* output)
{
  const int shape[] = {rows, columns};
  const int weight_shape[] = {1, columns};
  ccv_nnc_tensor_t tensors[] = {
    tensor(values, shape, 2), tensor(gate, shape, 2),
    tensor(weight, weight_shape, 2), tensor(output, shape, 2)
  };
  ccv_nnc_tensor_t* inputs[] = {tensors, tensors + 1, tensors + 2};
  ccv_nnc_tensor_t* outputs[] = {tensors + 3};
  const ccv_nnc_cmd_t command = {.info = {.rmsnorm_gated = {
    .epsilon = 1e-6, .elementwise_affine = 1, .count = 1, .axis = {1}}}};
  const ccv_nnc_hint_t hint = {0};
  const int result = _ccv_nnc_rmsnorm_gated_forw(command, hint, 0, inputs, 3, outputs, 1, 0);
  free(workspace);
  workspace = 0;
  return result;
}

int dt_probe_native_store(void* initialize, void* open_database, void* read_tensor,
  void* close_database, void* free_tensor, const char* path)
{
  ((void (*)(void))initialize)();
  void* database = 0;
  if (((int (*)(const char*, void**, int, const char*))open_database)(path, &database, 1, 0) != 0)
    return 1;
  ccv_nnc_tensor_t* result = 0;
  const int status = ((int (*)(void*, const char*, const ccv_nnc_tensor_io_option_t*,
    int, const ccv_nnc_tensor_param_t*, ccv_nnc_tensor_t**))read_tensor)(
      database, "split_matrix", 0, 0, 0, &result);
  int failed = status != CCV_IO_FINAL || !result;
  if (!failed)
  {
    failed = result->info.datatype != CCV_32F || result->info.format != CCV_TENSOR_FORMAT_NHWC
      || result->info.dim[0] != 3 || result->info.dim[1] != 4;
    for (int index = 0; index < 12; index++)
      failed |= result->data.f32[index] != (float)index;
  }
  if (result)
    ((void (*)(ccv_nnc_tensor_t*))free_tensor)(result);
  ((int (*)(void*))close_database)(database);
  return failed;
}