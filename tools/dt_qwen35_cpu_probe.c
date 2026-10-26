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

int dt_probe_native_init(void* initialize)
{
  ((void (*)(void))initialize)();
  return 0;
}

uint32_t dt_probe_native_backend(void* select_backend, const uint32_t command,
  const int memory, const int datatype)
{
  const ccv_nnc_cmd_t wrapped = {.cmd = command};
  return ((uint32_t (*)(ccv_nnc_cmd_t, int, int, int))select_backend)(
    wrapped, memory, CCV_TENSOR_FORMAT_NHWC, datatype);
}

int dt_probe_native_nonfinite(const void* tensor_pointer, void* synchronize, void* transfer)
{
  const ccv_nnc_tensor_t* source = tensor_pointer;
  const int datatype = source->info.datatype;
  if (datatype != CCV_32F && datatype != CCV_16F && datatype != CCV_16BF) return -30;
  const int count = ccv_nnc_tensor_count(source->info);
  if (!count) return 0;
  const size_t size = (size_t)count * (datatype == CCV_32F ? 4 : 2);
  void* values = malloc(size);
  if (!values) return -31;
  int status = 0;
  if (CCV_TENSOR_GET_MEMORY(source->info.type) == CCV_TENSOR_GPU_MEMORY)
  {
    status = ((int (*)(void))synchronize)();
    if (!status) status = ((int (*)(void*, const void*, size_t, int))transfer)(values, source->data.u8, size, 2);
  } else {
    memcpy(values, source->data.u8, size);
  }
  int nonfinite = 0;
  if (!status)
    for (int index = 0; index < count; index++)
      nonfinite += datatype == CCV_32F ? !isfinite(((const float*)values)[index]) :
        datatype == CCV_16F ? ((((const uint16_t*)values)[index] & 0x7c00) == 0x7c00) :
        ((((const uint16_t*)values)[index] & 0x7f80) == 0x7f80);
  free(values);
  return status ? -32 : nonfinite;
}

int dt_probe_native_sdpa_contract(void* forward, const int mode)
{
  ccv_nnc_tensor_t tensors[7] = {0};
  for (int index = 0; index < 7; index++)
  {
    tensors[index].info.type = CCV_TENSOR_GPU_MEMORY;
    tensors[index].info.format = CCV_TENSOR_FORMAT_NHWC;
    tensors[index].info.datatype = CCV_16F;
    if ((mode <= 5 && mode == index + 1) || (mode == 7 && index == 3))
    {
      const int shape[] = {1, 1, 1, 8};
      memcpy(tensors[index].info.dim, shape, sizeof(shape));
    }
  }
  ccv_nnc_tensor_t* inputs[] = {tensors, tensors + 1, tensors + 2, 0, tensors + 6};
  ccv_nnc_tensor_t* outputs[] = {tensors + 3, tensors + 4, tensors + 5};
  const ccv_nnc_cmd_t command = {.cmd = CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD,
    .algorithm = -1, .info = {.scaled_dot_product_attention = {.scale = 1}}};
  const ccv_nnc_hint_t hint = {0};
  return ((int (*)(ccv_nnc_cmd_t, ccv_nnc_hint_t, int, ccv_nnc_tensor_t* const*, int,
    ccv_nnc_tensor_t* const*, int, ccv_nnc_stream_context_t*))forward)(
      command, hint, 0, inputs, mode >= 6 ? 5 : 3, outputs, mode >= 6 ? 3 : mode == 5 ? 2 : 1, 0);
}

int dt_probe_native_sdpa_execution(void* allocate_tensor, void* release_tensor, void* execute_command)
{
  typedef ccv_nnc_tensor_t* (*allocate_fn)(void*, ccv_nnc_tensor_param_t, int);
  typedef int (*execute_fn)(ccv_nnc_cmd_t, ccv_nnc_hint_t, int, ccv_nnc_tensor_t* const*, int,
    ccv_nnc_tensor_t* const*, int, ccv_nnc_stream_context_t*);
  const int shapes[4][4] = {{2,3,4,32},{2,5,2,32},{2,5,2,32},{2,3,4,32}};
  const uint16_t values[] = {0, 0x3400, 0x3800, 0x3a00, 0x3c00};
  uint16_t buffers[4][768] = {{0}};
  ccv_nnc_tensor_t host[4] = {0};
  ccv_nnc_tensor_t* host_inputs[4];
  ccv_nnc_tensor_t* device[4] = {0};
  int status = 0;
  for (int index = 0; index < 4; index++)
  {
    host[index].info.type = CCV_TENSOR_CPU_MEMORY;
    host[index].info.format = CCV_TENSOR_FORMAT_NHWC;
    host[index].info.datatype = CCV_16F;
    memcpy(host[index].info.dim, shapes[index], sizeof(shapes[index]));
    host[index].data.u8 = (unsigned char*)buffers[index];
    host_inputs[index] = host + index;
    ccv_nnc_tensor_param_t parameters = host[index].info;
    parameters.type = CCV_TENSOR_GPU_MEMORY;
    device[index] = ((allocate_fn)allocate_tensor)(0, parameters, 0);
    if (!device[index]) { status = -20; break; }
  }
  for (int batch = 0; batch < 2; batch++)
    for (int token = 0; token < 5; token++)
      for (int head = 0; head < 2; head++)
        for (int channel = 0; channel < 32; channel++)
          buffers[2][((batch * 5 + token) * 2 + head) * 32 + channel] =
            (values[token] + (batch && token ? 0x400 : 0)) | (head ? 0x8000 : 0);
  for (int index = 0; index < 768; index++) buffers[3][index] = 0x7e00;
  const ccv_nnc_hint_t hint = {0};
  const ccv_nnc_cmd_t transfer = {.cmd = CCV_NNC_DATA_TRANSFER_FORWARD, .algorithm = -1};
  const ccv_nnc_cmd_t attention = {.cmd = CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD,
    .algorithm = -1, .info = {.scaled_dot_product_attention = {.scale = 1}}};
  if (!status) status = ((execute_fn)execute_command)(transfer, hint, 0, host_inputs, 4, device, 4, 0);
  if (!status) status = ((execute_fn)execute_command)(attention, hint, 0, device, 3, device + 3, 1, 0);
  if (!status) status = ((execute_fn)execute_command)(transfer, hint, 0, device + 3, 1, host_inputs + 3, 1, 0);
  if (!status)
    for (int index = 0; index < 768; index++)
    {
      const int expected_magnitude = index < 384 ? 0x3800 : 0x3c00;
      const int expected_sign = (index / 32) % 4 >= 2 ? 0x8000 : 0;
      const int magnitude = buffers[3][index] & 0x7fff;
      if ((magnitude & 0x7c00) == 0x7c00 || (buffers[3][index] & 0x8000) != expected_sign ||
          abs(magnitude - expected_magnitude) > 2)
        status = -21;
    }
  for (int index = 0; index < 4; index++)
    if (device[index]) ((void (*)(ccv_nnc_tensor_t*))release_tensor)(device[index]);
  return status;
}

int dt_probe_native_qwen_execution(void* allocate_tensor, void* release_tensor, void* execute_command)
{
  typedef ccv_nnc_tensor_t* (*allocate_fn)(void*, ccv_nnc_tensor_param_t, int);
  typedef void (*release_fn)(ccv_nnc_tensor_t*);
  typedef int (*execute_fn)(ccv_nnc_cmd_t, ccv_nnc_hint_t, int, ccv_nnc_tensor_t* const*, int,
    ccv_nnc_tensor_t* const*, int, ccv_nnc_stream_context_t*);
  const int shapes[8][4] = {{1,3,2,3},{1,3,2,3},{1,3,4,5},{1,3,4,0},
    {1,3,4,0},{1,4,5,3},{1,3,4,5},{1,12,5,3}};
  float buffers[8][180] = {{0}};
  ccv_nnc_tensor_t host[8];
  ccv_nnc_tensor_t* host_inputs[8];
  ccv_nnc_tensor_t* device[8] = {0};
  int status = 0;
  for (int index = 0; index < 8; index++)
  {
    const int rank = index == 3 || index == 4 ? 3 : 4;
    host[index] = tensor(buffers[index], shapes[index], rank);
    host_inputs[index] = host + index;
    const int count = ccv_nnc_tensor_count(host[index].info);
    for (int element = 0; element < count; element++)
      buffers[index][element] = index >= 6 ? NAN : index == 3 ? -0.1f : index == 4 ? 0.5f : 0.01f * ((element % 11) - 5);
    ccv_nnc_tensor_param_t parameters = host[index].info;
    parameters.type = CCV_TENSOR_GPU_MEMORY;
    device[index] = ((allocate_fn)allocate_tensor)(0, parameters, 0);
    if (!device[index]) { status = -10; break; }
  }
  const ccv_nnc_hint_t hint = {0};
  const ccv_nnc_cmd_t transfer = {.cmd = CCV_NNC_DATA_TRANSFER_FORWARD, .algorithm = -1};
  const ccv_nnc_cmd_t delta = {.cmd = CCV_NNC_GATED_DELTA_FORWARD, .algorithm = -1,
    .info = {.gated_delta = {.log_decay = 1, .state_checkpoint_count = 2}}};
  if (!status)
    status = ((execute_fn)execute_command)(transfer, hint, 0, host_inputs, 8, device, 8, 0);
  if (!status)
    status = ((execute_fn)execute_command)(delta, hint, 0, device, 6, device + 6, 2, 0);
  if (!status)
    status = ((execute_fn)execute_command)(transfer, hint, 0, device + 6, 2, host_inputs + 6, 2, 0);
  float expected_output[60], expected_state[180];
  ccv_nnc_tensor_t reference[] = {tensor(expected_output, shapes[6], 4), tensor(expected_state, shapes[7], 4)};
  ccv_nnc_tensor_t* expected[] = {reference, reference + 1};
  if (!status)
    status = _ccv_nnc_gated_delta_forw(delta, hint, 0, host_inputs, 6, expected, 2, 0);
  if (!status)
    for (int index = 0; index < 180; index++)
      if (!isfinite(buffers[7][index]) || fabsf(buffers[7][index] - expected_state[index]) > 2e-6f)
        status = -11;
  if (!status)
    for (int index = 0; index < 60; index++)
      if (!isfinite(buffers[6][index]) || fabsf(buffers[6][index] - expected_output[index]) > 2e-6f)
        status = -12;
  if (!status)
  {
    const int norm_shapes[4][2] = {{3,4},{3,4},{1,4},{3,4}};
    ccv_nnc_tensor_t norm_host[4];
    ccv_nnc_tensor_t* norm_inputs[4];
    for (int index = 0; index < 4; index++)
    {
      norm_host[index] = tensor(buffers[index], norm_shapes[index], 2);
      norm_inputs[index] = norm_host + index;
      memset(device[index]->info.dim, 0, sizeof(device[index]->info.dim));
      memcpy(device[index]->info.dim, norm_shapes[index], sizeof(norm_shapes[index]));
      for (int element = 0; element < ccv_nnc_tensor_count(norm_host[index].info); element++)
        buffers[index][element] = index == 3 ? NAN : 0.1f * ((element % 9) - 4);
    }
    const ccv_nnc_cmd_t norm = {.cmd = CCV_NNC_RMSNORM_GATED_FORWARD, .algorithm = -1,
      .info = {.rmsnorm_gated = {.epsilon = 1e-6, .elementwise_affine = 1, .count = 1, .axis = {1}}}};
    status = ((execute_fn)execute_command)(transfer, hint, 0, norm_inputs, 4, device, 4, 0);
    if (!status) status = ((execute_fn)execute_command)(norm, hint, 0, device, 3, device + 3, 1, 0);
    if (!status) status = ((execute_fn)execute_command)(transfer, hint, 0, device + 3, 1, norm_inputs + 3, 1, 0);
    ccv_nnc_tensor_t expected_norm = tensor(expected_output, norm_shapes[3], 2);
    ccv_nnc_tensor_t* expected_norm_outputs[] = {&expected_norm};
    if (!status) status = _ccv_nnc_rmsnorm_gated_forw(norm, hint, 0, norm_inputs, 3, expected_norm_outputs, 1, 0);
    if (!status)
      for (int element = 0; element < 12; element++)
        if (!isfinite(buffers[3][element]) || fabsf(buffers[3][element] - expected_output[element]) > 2e-6f)
          status = -13;
    free(workspace);
    workspace = 0;
  }
  for (int index = 0; index < 8; index++)
    if (device[index]) ((release_fn)release_tensor)(device[index]);
  return status;
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