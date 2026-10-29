#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"

static void* workspace;
static size_t capture_limit = 64 * 1024 * 1024;

int dt_probe_native_set_capture_limit(const size_t limit)
{
  if (!limit || limit > 128 * 1024 * 1024) return -34;
  capture_limit = limit;
  return 0;
}

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
  if (size > 64 * 1024 * 1024) return -34;
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
  float maximum_f32 = 0;
  uint16_t maximum_16 = 0;
  if (!status)
    for (int index = 0; index < count; index++)
    {
      nonfinite += datatype == CCV_32F ? !isfinite(((const float*)values)[index]) :
        datatype == CCV_16F ? ((((const uint16_t*)values)[index] & 0x7c00) == 0x7c00) :
        ((((const uint16_t*)values)[index] & 0x7f80) == 0x7f80);
      if (datatype == CCV_32F)
      {
        const float magnitude = fabsf(((const float*)values)[index]);
        if (isfinite(magnitude) && magnitude > maximum_f32) maximum_f32 = magnitude;
      } else {
        const uint16_t magnitude = ((const uint16_t*)values)[index] & 0x7fff;
        const int finite = datatype == CCV_16F ? magnitude < 0x7c00 : magnitude < 0x7f80;
        if (finite && magnitude > maximum_16) maximum_16 = magnitude;
      }
    }
  if (!status)
    fprintf(stderr, "DT_TENSOR_RANGE count=%d datatype=%d nonfinite=%d maximum_f32=%g maximum_16=0x%04x\n",
      count, datatype, nonfinite, maximum_f32, maximum_16);
  free(values);
  return status ? -32 : nonfinite;
}

int dt_probe_native_tensor_dump(const void* tensor_pointer, void* synchronize, void* transfer,
  const char* path)
{
  const ccv_nnc_tensor_t* source = tensor_pointer;
  const int datatype = source->info.datatype;
  if (datatype != CCV_32F && datatype != CCV_16F && datatype != CCV_16BF) return -30;
  if (!CCV_IS_TENSOR_CONTIGUOUS(source)) return -33;
  const size_t count = ccv_nnc_tensor_count(source->info);
  const size_t size = count * (datatype == CCV_32F ? 4 : 2);
  if (!count || size > capture_limit) return -34;
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
  if (status) { free(values); return -32; }
  const int descriptor = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
  if (descriptor < 0) { free(values); return -35; }
  FILE* output = fdopen(descriptor, "wb");
  if (!output) { close(descriptor); unlink(path); free(values); return -35; }
  uint32_t header[17] = {0, CCV_TENSOR_CPU_MEMORY, source->info.format, datatype, 0};
  memcpy(header + 5, source->info.dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
  if (fwrite(header, 1, sizeof(header), output) != sizeof(header) ||
      fwrite(values, 1, size, output) != size) status = -35;
  if (fclose(output)) status = -35;
  if (status) unlink(path);
  free(values);
  return status;
}

const char* dt_probe_native_command_metadata(const void* command_pointer, void* command_name)
{
  static char metadata[1024];
  const ccv_nnc_cmd_t* command = command_pointer;
  const char* name = ((const char* (*)(uint32_t))command_name)(command->cmd);
  if (!name) return 0;
  const int is_norm = command->cmd == CCV_NNC_RMSNORM_FORWARD;
  const int is_blas = command->cmd == CCV_NNC_ADD_FORWARD || command->cmd == CCV_NNC_MUL_FORWARD ||
    command->cmd == CCV_NNC_SCALAR_MUL_FORWARD;
  const int is_gemm = command->cmd == CCV_NNC_GEMM_FORWARD;
  snprintf(metadata, sizeof(metadata),
    "{\"code\":%u,\"name\":\"%s\",\"scalars\":[%.9g,%.9g],\"soft_cap\":%.9g,\"rms_epsilon\":%.9g,\"rms_scale\":%.9g,\"rms_axis_count\":%d,\"rms_axes\":[%d,%d,%d],\"attention_scale\":%.9g,\"transpose_a\":[%d,%d],\"transpose_b\":[%d,%d]}",
    command->cmd, name, is_blas ? command->info.blas.a[0] : 0, is_blas ? command->info.blas.a[1] : 0,
    command->cmd == CCV_NNC_TANH_FORWARD ? command->info.tanh.soft_cap : 0,
    is_norm ? command->info.rmsnorm.epsilon : 0, is_norm ? command->info.rmsnorm.scale : 0,
    is_norm ? command->info.rmsnorm.count : 0,
    is_norm ? command->info.rmsnorm.axis[0] : 0, is_norm ? command->info.rmsnorm.axis[1] : 0,
    is_norm ? command->info.rmsnorm.axis[2] : 0,
    command->cmd == CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD ? command->info.scaled_dot_product_attention.scale : 0,
    is_gemm ? command->info.blas.transpose_a[0] : 0, is_gemm ? command->info.blas.transpose_a[1] : 0,
    is_gemm ? command->info.blas.transpose_b[0] : 0, is_gemm ? command->info.blas.transpose_b[1] : 0);
  return metadata;
}

const char* dt_probe_native_tensor_metadata(const void* tensor_pointer)
{
  static char metadata[1024];
  const ccv_nnc_tensor_t* source = tensor_pointer;
  snprintf(metadata, sizeof(metadata),
    "{\"data\":%llu,\"memory\":%d,\"format\":%d,\"datatype\":%d,\"reserved\":%d,\"contiguous\":%d,\"dimensions\":[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d]}",
    (unsigned long long)(uintptr_t)source->data.u8, source->info.type, source->info.format,
    source->info.datatype, source->info.reserved, !!CCV_IS_TENSOR_CONTIGUOUS(source),
    source->info.dim[0], source->info.dim[1], source->info.dim[2], source->info.dim[3],
    source->info.dim[4], source->info.dim[5], source->info.dim[6], source->info.dim[7],
    source->info.dim[8], source->info.dim[9], source->info.dim[10], source->info.dim[11]);
  return metadata;
}

int dt_probe_native_decoded_tensor_dump(const void* tensor_pointer, void* synchronize,
  void* transfer, void* allocate_tensor, void* release_tensor, void* decode_tensor,
  void* prefetch_tensor, void* execute_command, const char* path, void* stream)
{
  const ccv_nnc_tensor_t* source = tensor_pointer;
  const int quantized = CCV_GET_DATA_TYPE(source->info.datatype) == CCV_QX;
  if (!quantized && CCV_IS_TENSOR_CONTIGUOUS(source))
    return dt_probe_native_tensor_dump(source, synchronize, transfer, path);
  if (quantized && !CCV_IS_TENSOR_CONTIGUOUS(source)) return -33;
  ccv_nnc_tensor_param_t parameters = source->info;
  parameters.datatype = quantized ? (parameters.datatype & 0xff) << 12 : parameters.datatype;
  parameters.reserved = 0;
  const size_t size = (size_t)ccv_nnc_tensor_count(parameters) * (parameters.datatype == CCV_32F ? 4 : 2);
  if (!size || size > capture_limit ||
      (parameters.datatype != CCV_32F && parameters.datatype != CCV_16F && parameters.datatype != CCV_16BF)) return -34;
  ccv_nnc_tensor_t* dense = ((ccv_nnc_tensor_t* (*)(void*, ccv_nnc_tensor_param_t, int))allocate_tensor)(0, parameters, 0);
  if (!dense) return -31;
  ((void (*)(ccv_nnc_tensor_t*, ccv_nnc_stream_context_t*))prefetch_tensor)((ccv_nnc_tensor_t*)source, stream);
  int status = 0;
  if (quantized)
    ((void (*)(const void*, ccv_nnc_tensor_param_t, void*, ccv_nnc_stream_context_t*))decode_tensor)(
      source->data.u8, source->info, dense->data.u8, stream);
  else {
    ccv_nnc_tensor_t* inputs[] = {(ccv_nnc_tensor_t*)source};
    ccv_nnc_tensor_t* outputs[] = {dense};
    const ccv_nnc_cmd_t transform = {.cmd = CCV_NNC_FORMAT_TRANSFORM_FORWARD, .algorithm = -1};
    const ccv_nnc_hint_t hint = {0};
    status = ((int (*)(ccv_nnc_cmd_t, ccv_nnc_hint_t, int, ccv_nnc_tensor_t* const*, int,
      ccv_nnc_tensor_t* const*, int, ccv_nnc_stream_context_t*))execute_command)(transform, hint, 0, inputs, 1, outputs, 1, stream);
  }
  if (!status) status = dt_probe_native_tensor_dump(dense, synchronize, transfer, path);
  ((void (*)(ccv_nnc_tensor_t*))release_tensor)(dense);
  return status;
}

int dt_probe_native_is_final_flow_step(const void* command_pointer)
{
  if (!command_pointer) return 0;
  const ccv_nnc_cmd_t* command = command_pointer;
  return command->cmd == CCV_NNC_ADD_FORWARD && command->info.blas.a[0] == 1 &&
    command->info.blas.a[1] == -1;
}

int dt_probe_native_sdpa_parameters(const void* command_pointer)
{
  const ccv_nnc_cmd_t* command = command_pointer;
  if (command->cmd != CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD) return -37;
  fprintf(stderr, "DT_SDPA_PARAMETERS scale=%g causal=%d varlen=%d\n",
    command->info.scaled_dot_product_attention.scale,
    command->info.scaled_dot_product_attention.is_causal,
    command->info.scaled_dot_product_attention.is_varlen);
  return 0;
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

int dt_probe_native_embedding_execution(void* allocate_tensor, void* release_tensor,
  void* execute_command, void* select_backend, void* quantize, void* dequantize)
{
  if (dt_probe_native_backend(select_backend, CCV_NNC_INDEX_SELECT_FORWARD,
      CCV_TENSOR_GPU_MEMORY, CCV_QX) != CCV_NNC_BACKEND_GPU_REF) return -40;
  typedef ccv_nnc_tensor_t* (*allocate_fn)(void*, ccv_nnc_tensor_param_t, int);
  typedef int (*execute_fn)(ccv_nnc_cmd_t, ccv_nnc_hint_t, int, ccv_nnc_tensor_t* const*, int,
    ccv_nnc_tensor_t* const*, int, ccv_nnc_stream_context_t*);
  const int datatypes[] = {CCV_32F, CCV_16F, CCV_16BF};
  const int formats[] = {CCV_NNC_QX_8I_ROWWISE_Q5_K, CCV_NNC_QX_8I_ROWWISE_Q6_K};
  const int indices[] = {2, 0, 2, 1};
  const float dense_f32[] = {1, 2, 4};
  const uint16_t dense_f16[] = {0x3c00, 0x4000, 0x4400};
  const uint16_t dense_bf16[] = {0x3f80, 0x4000, 0x4080};
  for (int format_index = 0; format_index < 2; format_index++)
    for (int datatype_index = 0; datatype_index < 3; datatype_index++)
    {
      unsigned char packed[1024] = {0};
      unsigned char dense[4096] = {0}, reference[4096] = {0};
      for (int element = 0; element < 768; element++)
      {
        const int row = (element / 256 + element / 13 + element % 7) % 3;
        if (datatypes[datatype_index] == CCV_32F) ((float*)dense)[element] = dense_f32[row];
        else ((uint16_t*)dense)[element] = datatypes[datatype_index] == CCV_16F ? dense_f16[row] : dense_bf16[row];
      }
      const size_t packed_size = ((size_t (*)(const void*, int, int, size_t, size_t, int,
        const float*, size_t, void*, size_t))quantize)(dense, datatypes[datatype_index],
          CCV_TENSOR_CPU_MEMORY, 768, 256, formats[format_index], 0, 0, packed, sizeof(packed));
      if (!packed_size || packed_size > sizeof(packed)) return -43;
      ((void (*)(const void*, int, int, size_t, size_t, int, void*, size_t))dequantize)(packed,
        datatypes[datatype_index], CCV_TENSOR_CPU_MEMORY, packed_size, 256, formats[format_index], reference, 768);
      unsigned char output[4096];
      memset(output, 0xff, sizeof(output));
      ccv_nnc_tensor_t host[3] = {0};
      ccv_nnc_tensor_t* host_pointers[] = {host, host + 1, host + 2};
      ccv_nnc_tensor_t* device[3] = {0};
      const int shapes[3][2] = {{3, 256}, {4, 0}, {4, 256}};
      int status = 0;
      for (int index = 0; index < 3; index++)
      {
        host[index].info.type = CCV_TENSOR_CPU_MEMORY;
        host[index].info.format = CCV_TENSOR_FORMAT_NHWC;
        host[index].info.datatype = index == 0 ? CCV_QX | CCV_NNC_QX_8I_ROWWISE_X |
          (datatypes[datatype_index] >> 12) : index == 1 ? CCV_32S : datatypes[datatype_index];
        host[index].info.reserved = index == 0 ? formats[format_index] : 0;
        memcpy(host[index].info.dim, shapes[index], sizeof(shapes[index]));
        host[index].data.u8 = index == 0 ? packed : index == 1 ? (unsigned char*)indices : output;
        ccv_nnc_tensor_param_t parameters = host[index].info;
        parameters.type = CCV_TENSOR_GPU_MEMORY;
        device[index] = ((allocate_fn)allocate_tensor)(0, parameters, 0);
        if (!device[index]) { status = -41; break; }
      }
      const ccv_nnc_hint_t hint = {0};
      const ccv_nnc_cmd_t transfer = {.cmd = CCV_NNC_DATA_TRANSFER_FORWARD, .algorithm = -1};
      const ccv_nnc_cmd_t lookup = {.cmd = CCV_NNC_INDEX_SELECT_FORWARD, .algorithm = -1};
      if (!status) status = ((execute_fn)execute_command)(transfer, hint, 0, host_pointers, 3, device, 3, 0);
      if (!status) status = ((execute_fn)execute_command)(lookup, hint, 0, device, 2, device + 2, 1, 0);
      if (!status) status = ((execute_fn)execute_command)(transfer, hint, 0, device + 2, 1, host_pointers + 2, 1, 0);
      if (!status)
        for (int element = 0; element < 1024; element++)
        {
          const int row = indices[element / 256];
          const int reference_element = row * 256 + element % 256;
            const uint16_t actual_bits = ((const uint16_t*)output)[element];
            const uint16_t reference_bits = ((const uint16_t*)reference)[reference_element];
            const int exponent_mask = datatypes[datatype_index] == CCV_16F ? 0x7c00 : 0x7f80;
          if (datatypes[datatype_index] == CCV_32F ?
              ((const float*)output)[element] != ((const float*)reference)[reference_element] :
              (actual_bits & exponent_mask) == exponent_mask ||
              (reference_bits & exponent_mask) == exponent_mask ||
              (actual_bits & 0x8000) != (reference_bits & 0x8000) ||
              abs((int)actual_bits - reference_bits) > 2)
          {
            fprintf(stderr, "DT_EMBEDDING_MISMATCH format=%d datatype=%d element=%d actual_f32=%g actual_16=0x%04x reference_16=0x%04x\n",
              formats[format_index], datatypes[datatype_index], element,
              datatypes[datatype_index] == CCV_32F ? ((const float*)output)[element] : 0,
              datatypes[datatype_index] == CCV_32F ? 0 : ((const uint16_t*)output)[element],
              datatypes[datatype_index] == CCV_32F ? 0 : ((const uint16_t*)reference)[reference_element]);
            status = -42;
            break;
          }
        }
      for (int index = 0; index < 3; index++)
        if (device[index]) ((void (*)(ccv_nnc_tensor_t*))release_tensor)(device[index]);
      if (status) return status;
    }
  return 0;
}

static float dt_probe_float_value(const unsigned char* data, const int datatype, const int element);

static void dt_probe_regular_hadamard_256(float* values)
{
  for (int stride = 1; stride < 256; stride *= 4)
    for (int block = 0; block < 256; block += stride * 4)
      for (int offset = 0; offset < stride; offset++)
      {
        float* group = values + block + offset;
        const float first = group[0], second = group[stride];
        const float third = group[stride * 2], fourth = group[stride * 3];
        group[0] = first + second + third - fourth;
        group[stride] = first + second - third + fourth;
        group[stride * 2] = first - second + third + fourth;
        group[stride * 3] = -first + second + third + fourth;
      }
  for (int element = 0; element < 256; element++)
    values[element] *= 0.0625f;
}

int dt_probe_native_packed_gemm(void* allocate_tensor, void* release_tensor, void* execute_command,
  void* quantize, void* dequantize)
{
  typedef ccv_nnc_tensor_t* (*allocate_fn)(void*, ccv_nnc_tensor_param_t, int);
  typedef int (*execute_fn)(ccv_nnc_cmd_t, ccv_nnc_hint_t, int, ccv_nnc_tensor_t* const*, int,
    ccv_nnc_tensor_t* const*, int, ccv_nnc_stream_context_t*);
  const int formats[] = {CCV_NNC_QX_8I_ROWWISE_Q4_K,
    CCV_NNC_QX_8I_ROWWISE_Q4_K | CCV_NNC_QX_8I_ROWWISE_HADAMARD_256,
    CCV_NNC_QX_8I_ROWWISE_Q5_K, CCV_NNC_QX_8I_ROWWISE_Q6_K};
  const uint16_t levels[] = {0x3400, 0x3800, 0x3a00, 0x3c00, 0xbc00, 0xb800, 0x4000, 0x4400};
  const float activation_values[] = {0.25f, 0.5f, 0.75f, 1, -1, -0.5f, 2, 4};
  for (int format_index = 0; format_index < (int)(sizeof(formats) / sizeof(formats[0])); format_index++)
  {
    uint16_t dense[32 * 256], reference[32 * 256], packed_reference[32 * 256];
    uint16_t activations[16 * 256], output[16 * 32], dense_output[16 * 32];
    unsigned char packed[8192] = {0};
    for (int index = 0; index < 32 * 256; index++)
      dense[index] = levels[((index / 256) + (index % 256) / 32 + index % 7) % 8];
    for (int index = 0; index < 16 * 256; index++)
      activations[index] = levels[((index / 256) + (index % 256) / 16) % 8];
    memset(output, 0xff, sizeof(output));
    const size_t packed_size = ((size_t (*)(const void*, int, int, size_t, size_t, int,
      const float*, size_t, void*, size_t))quantize)(dense, CCV_16F, CCV_TENSOR_CPU_MEMORY,
        32 * 256, 256, formats[format_index], 0, 0, packed, sizeof(packed));
    if (!packed_size || packed_size > sizeof(packed)) return -50;
    ((void (*)(const void*, int, int, size_t, size_t, int, void*, size_t))dequantize)(packed,
      CCV_16F, CCV_TENSOR_CPU_MEMORY, packed_size, 256, formats[format_index], reference, 32 * 256);
    ((void (*)(const void*, int, int, size_t, size_t, int, void*, size_t))dequantize)(packed,
      CCV_16F, CCV_TENSOR_CPU_MEMORY, packed_size, 256,
      formats[format_index] & ~CCV_NNC_QX_8I_ROWWISE_HADAMARD_256, packed_reference, 32 * 256);
    ccv_nnc_tensor_t host[3] = {0};
    ccv_nnc_tensor_t* host_pointers[] = {host, host + 1, host + 2};
    ccv_nnc_tensor_t* device[3] = {0};
    const int shapes[3][2] = {{16, 256}, {32, 256}, {16, 32}};
    int status = 0;
    for (int index = 0; index < 3; index++)
    {
      host[index].info.type = CCV_TENSOR_CPU_MEMORY;
      host[index].info.format = CCV_TENSOR_FORMAT_NHWC;
      host[index].info.datatype = index == 1 ? CCV_QX | CCV_NNC_QX_8I_ROWWISE_X | (CCV_16F >> 12) : CCV_16F;
      host[index].info.reserved = index == 1 ? formats[format_index] : 0;
      memcpy(host[index].info.dim, shapes[index], sizeof(shapes[index]));
      host[index].data.u8 = index == 0 ? (unsigned char*)activations : index == 1 ? packed : (unsigned char*)output;
      ccv_nnc_tensor_param_t parameters = host[index].info;
      parameters.type = CCV_TENSOR_GPU_MEMORY;
      device[index] = ((allocate_fn)allocate_tensor)(0, parameters, 0);
      if (!device[index]) { status = -51; break; }
    }
    const ccv_nnc_hint_t hint = {0};
    const ccv_nnc_cmd_t transfer = {.cmd = CCV_NNC_DATA_TRANSFER_FORWARD, .algorithm = -1};
    const ccv_nnc_cmd_t gemm = {.cmd = CCV_NNC_GEMM_FORWARD, .algorithm = -1,
      .info = {.blas = {.transpose_b = {0, 1}}}};
    if (!status) status = ((execute_fn)execute_command)(transfer, hint, 0, host_pointers, 3, device, 3, 0);
    if (!status) status = ((execute_fn)execute_command)(gemm, hint, 0, device, 2, device + 2, 1, 0);
    if (!status) status = ((execute_fn)execute_command)(transfer, hint, 0, device + 2, 1, host_pointers + 2, 1, 0);
    if (!status)
    {
      ((void (*)(ccv_nnc_tensor_t*))release_tensor)(device[1]);
      ccv_nnc_tensor_param_t parameters = host[1].info;
      parameters.type = CCV_TENSOR_GPU_MEMORY;
      parameters.datatype = CCV_16F;
      parameters.reserved = 0;
      device[1] = ((allocate_fn)allocate_tensor)(0, parameters, 0);
      if (!device[1]) status = -51;
      host[1].info = parameters;
      host[1].info.type = CCV_TENSOR_CPU_MEMORY;
      host[1].data.u8 = (unsigned char*)reference;
      host[2].data.u8 = (unsigned char*)dense_output;
      if (!status) status = ((execute_fn)execute_command)(transfer, hint, 0, host_pointers + 1, 1, device + 1, 1, 0);
      if (!status) status = ((execute_fn)execute_command)(gemm, hint, 0, device, 2, device + 2, 1, 0);
      if (!status) status = ((execute_fn)execute_command)(transfer, hint, 0, device + 2, 1, host_pointers + 2, 1, 0);
    }
    if (!status)
      for (int row = 0; row < 16; row++)
      {
        float packed_activations[256];
        for (int reduction = 0; reduction < 256; reduction++)
          packed_activations[reduction] = activation_values[(row + reduction / 16) % 8];
        if (formats[format_index] & CCV_NNC_QX_8I_ROWWISE_HADAMARD_256)
          dt_probe_regular_hadamard_256(packed_activations);
        float maximum_activation = 0;
        for (int reduction = 0; reduction < 256; reduction++)
          maximum_activation = fmaxf(maximum_activation, fabsf(packed_activations[reduction]));
        for (int column = 0; column < 32; column++)
        {
          float expected = 0;
          float expected_dense = 0;
          for (int reduction = 0; reduction < 256; reduction++)
          {
            const float activation = activation_values[(row + reduction / 16) % 8];
            const float weight = dt_probe_float_value((const unsigned char*)reference, CCV_16F, column * 256 + reduction);
            const float packed_weight = dt_probe_float_value((const unsigned char*)packed_reference, CCV_16F, column * 256 + reduction);
            expected_dense += weight * activation;
            expected += packed_weight * nearbyintf(packed_activations[reduction] * (127.0f / maximum_activation)) * (maximum_activation / 127);
          }
          const uint16_t bits = output[row * 32 + column];
          const int exponent = (bits >> 10) & 31;
          const float magnitude = exponent ? ldexpf((bits & 1023) + 1024, exponent - 25) : ldexpf(bits & 1023, -24);
          const float actual = bits & 0x8000 ? -magnitude : magnitude;
          const uint16_t control_bits = dense_output[row * 32 + column];
          const int control_exponent = (control_bits >> 10) & 31;
          const float control_magnitude = control_exponent ? ldexpf((control_bits & 1023) + 1024, control_exponent - 25) : ldexpf(control_bits & 1023, -24);
          const float control = control_bits & 0x8000 ? -control_magnitude : control_magnitude;
            if (exponent == 31 || control_exponent == 31 || !isfinite(expected) || !isfinite(expected_dense) ||
              fabsf(actual - expected) > 0.003f * fmaxf(1, fabsf(expected)) ||
              fabsf(control - expected_dense) > 0.003f * fmaxf(1, fabsf(expected_dense)))
          {
            fprintf(stderr, "DT_PACKED_GEMM_MISMATCH format=%d row=%d column=%d packed=%g dense=%g expected=%g\n",
              formats[format_index], row, column, actual, control, expected);
            status = -52;
            break;
          }
        }
      }
    for (int index = 0; index < 3; index++)
      if (device[index]) ((void (*)(ccv_nnc_tensor_t*))release_tensor)(device[index]);
    if (status) return status;
  }
  return 0;
}

static float dt_probe_float_value(const unsigned char* data, const int datatype, const int element)
{
  if (datatype == CCV_32F) return ((const float*)data)[element];
  const uint16_t bits = ((const uint16_t*)data)[element];
  float value;
  if (datatype == CCV_16BF)
  {
    const uint32_t expanded = (uint32_t)bits << 16;
    memcpy(&value, &expanded, 4);
  } else {
    const int exponent = (bits >> 10) & 31;
    value = exponent == 31 ? NAN : exponent ? ldexpf((bits & 1023) + 1024, exponent - 25) : ldexpf(bits & 1023, -24);
    if (bits & 0x8000) value = -value;
  }
  return value;
}

int dt_probe_native_mapped_gemm(void* allocate_tensor, void* release_tensor, void* execute_command,
  void* allocate_managed, void* release_memory, void* peek_error)
{
  typedef ccv_nnc_tensor_t* (*allocate_fn)(void*, ccv_nnc_tensor_param_t, int);
  typedef int (*execute_fn)(ccv_nnc_cmd_t, ccv_nnc_hint_t, int, ccv_nnc_tensor_t* const*, int,
    ccv_nnc_tensor_t* const*, int, ccv_nnc_stream_context_t*);
  const int datatypes[] = {CCV_32F, CCV_16F, CCV_16BF};
  const uint16_t half_inputs[] = {0x3400, 0xb800, 0x3c00};
  const uint16_t half_weights[] = {0x3000, 0x3400, 0x3600, 0x3800, 0x3900};
  const float input_values[] = {0.25f, -0.5f, 1};
  const int shapes[4][2] = {{3, 8}, {5, 8}, {5, 0}, {3, 5}};
  for (int datatype_index = 0; datatype_index < 3; datatype_index++)
    for (int weight_transposed = 0; weight_transposed < 2; weight_transposed++)
      for (int managed = 0; managed < 2; managed++)
        for (int with_bias = 0; with_bias < 2; with_bias++)
  {
    unsigned char data[4][64 * 4];
    memset(data, 0xff, sizeof(data));
    for (int index = 0; index < 3; index++)
    {
      const int count = index == 0 ? 24 : index == 1 ? 40 : 5;
      for (int element = 0; element < count; element++)
      {
        const int weight_column = weight_transposed ? element / 8 : element % 5;
        const float value = index == 0 ? input_values[element / 8] : index == 1 ? 0.125f * (weight_column + 1) : 0.125f;
        if (datatypes[datatype_index] == CCV_32F) ((float*)data[index])[element] = value;
        else if (datatypes[datatype_index] == CCV_16F)
          ((uint16_t*)data[index])[element] = index == 0 ? half_inputs[element / 8] : index == 1 ? half_weights[weight_column] : 0x3000;
        else {
          uint32_t bits;
          memcpy(&bits, &value, sizeof(bits));
          ((uint16_t*)data[index])[element] = bits >> 16;
        }
      }
    }
    ccv_nnc_tensor_t host[4] = {0};
    ccv_nnc_tensor_t* host_pointers[] = {host, host + 1, host + 2, host + 3};
    ccv_nnc_tensor_t* device[4] = {0};
    void* memory = 0;
    int status = 0;
    for (int index = 0; index < 4; index++)
    {
      host[index].info.type = CCV_TENSOR_CPU_MEMORY;
      host[index].info.format = CCV_TENSOR_FORMAT_NHWC;
      host[index].info.datatype = datatypes[datatype_index];
      memcpy(host[index].info.dim, shapes[index], sizeof(shapes[index]));
      if (index == 1 && !weight_transposed)
      {
        host[index].info.dim[0] = 8;
        host[index].info.dim[1] = 5;
      }
      host[index].data.u8 = data[index];
      ccv_nnc_tensor_param_t parameters = host[index].info;
      parameters.type = CCV_TENSOR_GPU_MEMORY;
      if (managed && index == 1)
      {
        memory = ((void* (*)(int, size_t))allocate_managed)(0, 40 * (datatypes[datatype_index] == CCV_32F ? 4 : 2));
        if (!memory) { status = -90; break; }
      }
      device[index] = ((allocate_fn)allocate_tensor)(index == 1 ? memory : 0, parameters, 0);
      if (!device[index]) { status = -90; break; }
      if (managed && index == 1) device[index]->type |= CCV_MAPPED_MEM;
    }
    const ccv_nnc_hint_t hint = {0};
    const ccv_nnc_cmd_t transfer = {.cmd = CCV_NNC_DATA_TRANSFER_FORWARD, .algorithm = -1};
    const ccv_nnc_cmd_t gemm = {.cmd = CCV_NNC_GEMM_FORWARD, .algorithm = -1,
      .info = {.blas = {.transpose_b = {0, weight_transposed ? 1 : 0}}}};
    if (!status) status = ((execute_fn)execute_command)(transfer, hint, 0, host_pointers, 3, device, 3, 0);
    if (!status) status = ((execute_fn)execute_command)(gemm, hint, 0, device, with_bias ? 3 : 2, device + 3, 1, 0);
    if (!status) status = ((execute_fn)execute_command)(transfer, hint, 0, device + 3, 1, host_pointers + 3, 1, 0);
    if (!status)
      for (int row = 0; row < 3; row++)
        for (int column = 0; column < 5; column++)
        {
          const float expected = input_values[row] * (column + 1) + (with_bias ? 0.125f : 0);
          if (dt_probe_float_value(data[3], datatypes[datatype_index], row * 5 + column) != expected) status = -91;
        }
    if (!status && ((int (*)(void))peek_error)()) status = -92;
    for (int index = 0; index < 4; index++)
      if (device[index]) ((void (*)(ccv_nnc_tensor_t*))release_tensor)(device[index]);
    if (memory) ((void (*)(int, void*))release_memory)(0, memory);
    if (status) return status;
  }
  return 0;
}

int dt_probe_native_empty_forward(void* execute_command, void* synchronize, void* peek_error,
  void* clear_error, void* set_device, void* device_count)
{
  typedef int (*execute_fn)(ccv_nnc_cmd_t, ccv_nnc_hint_t, int, ccv_nnc_tensor_t* const*, int,
    ccv_nnc_tensor_t* const*, int, ccv_nnc_stream_context_t*);
  const int datatypes[] = {CCV_32F, CCV_16F, CCV_16BF};
  const uint32_t operations[] = {CCV_NNC_DATATYPE_CONVERSION_FORWARD, CCV_NNC_CMUL_FORWARD,
    CCV_NNC_GEMM_FORWARD, CCV_NNC_TANH_FORWARD, CCV_NNC_TANH_FORWARD};
  int devices = 0;
  if (((int (*)(int*))device_count)(&devices) || devices < 1 || ((int (*)(int))set_device)(0)) return -72;
  ((int (*)(void))clear_error)();
  const ccv_nnc_hint_t hint = {0};
  for (int operation_index = 0; operation_index < 5; operation_index++)
    for (int datatype_index = 0; datatype_index < 3; datatype_index++)
      for (int output_datatype_index = 0; output_datatype_index < 3; output_datatype_index++)
      {
        const int binary_operation = operation_index == 1 || operation_index == 2;
        if (binary_operation && output_datatype_index != datatype_index) continue;
        if (operation_index >= 3 && (datatype_index == 2 || output_datatype_index == 2)) continue;
        const int tensor_size = binary_operation ? 3 : 2;
        for (int mask = 0; mask < (1 << tensor_size) - 1; mask++)
        {
          ccv_nnc_tensor_t tensors[3] = {0};
          for (int index = 0; index < tensor_size; index++)
          {
            tensors[index].info.type = CCV_TENSOR_GPU_MEMORY;
            tensors[index].info.format = CCV_TENSOR_FORMAT_NHWC;
            tensors[index].info.datatype = !binary_operation && index == 1 ? datatypes[output_datatype_index] : datatypes[datatype_index];
            if (mask & (1 << index))
              tensors[index].info.dim[0] = 2;
          }
          ccv_nnc_tensor_t* inputs[] = {tensors, tensors + 1};
          ccv_nnc_tensor_t* outputs[] = {tensors + tensor_size - 1};
          ccv_nnc_cmd_t command = {.cmd = operations[operation_index], .algorithm = -1};
          if (operation_index == 2) command.info.blas.transpose_b[1] = 1;
          if (operation_index == 3) command.info.tanh.soft_cap = 5;
          const int expected = mask == 0 || (operation_index == 2 && mask == 2) ? CCV_NNC_EXEC_SUCCESS : CCV_NNC_EXEC_INVALID;
          int status = ((execute_fn)execute_command)(command, hint, 0, inputs, binary_operation ? 2 : 1, outputs, 1, 0);
          if (status != expected || ((int (*)(void))peek_error)() || ((int (*)(void))synchronize)())
          {
            ((int (*)(void))clear_error)();
            return -70;
          }
          if (expected == CCV_NNC_EXEC_SUCCESS)
          {
            const int injected = ((int (*)(int))set_device)(devices);
            if (!injected || ((int (*)(void))peek_error)() != injected) return -72;
            status = ((execute_fn)execute_command)(command, hint, 0, inputs, binary_operation ? 2 : 1, outputs, 1, 0);
            const int preserved = ((int (*)(void))peek_error)();
            ((int (*)(void))clear_error)();
            if (status != expected || preserved != injected) return -71;
          }
        }
      }
  return 0;
}

int dt_probe_native_memory_advice(void* allocate_managed, void* release_memory, void* advise_memory,
  void* transfer, void* synchronize, void* peek_error, void* clear_error, void* set_device, void* device_count,
  void* prefetch_tensor)
{
  const uint32_t expected[] = {0x11223344, 0xdeadbeef, 0, 0xffffffff};
  uint32_t actual[4] = {0};
  int devices = 0;
  if (((int (*)(int*))device_count)(&devices) || devices < 1 || ((int (*)(int))set_device)(0)) return -80;
  ((int (*)(void))clear_error)();
  void* memory = ((void* (*)(int, size_t))allocate_managed)(0, sizeof(expected));
  if (!memory) return -80;
  ccv_nnc_tensor_t tensor = {0};
  tensor.type = CCV_MAPPED_MEM;
  tensor.info.type = CCV_TENSOR_GPU_MEMORY;
  tensor.info.format = CCV_TENSOR_FORMAT_NHWC;
  tensor.info.datatype = CCV_32F;
  tensor.info.dim[0] = 4;
  tensor.data.u8 = memory;
  int status = ((int (*)(void*, const void*, size_t, int))transfer)(memory, expected, sizeof(expected), 1);
  if (!status)
  {
    ((void (*)(int, void*, size_t))advise_memory)(0, memory, sizeof(expected));
    if (((int (*)(void))peek_error)() || ((int (*)(void))synchronize)()) status = -81;
  }
  if (!status)
  {
    ((void (*)(ccv_nnc_tensor_t*, const ccv_nnc_stream_context_t*))prefetch_tensor)(&tensor, 0);
    if (((int (*)(void))peek_error)() || ((int (*)(void))synchronize)()) status = -84;
  }
  if (!status)
  {
    status = ((int (*)(void*, const void*, size_t, int))transfer)(actual, memory, sizeof(actual), 2);
    if (!status && memcmp(expected, actual, sizeof(actual))) status = -82;
  }
  if (!status)
  {
    const int injected = ((int (*)(int))set_device)(devices);
    if (!injected || ((int (*)(void))peek_error)() != injected) status = -83;
    else {
      ((void (*)(int, void*, size_t))advise_memory)(0, memory, sizeof(expected));
      if (((int (*)(void))peek_error)() != injected) status = -83;
      ((void (*)(ccv_nnc_tensor_t*, const ccv_nnc_stream_context_t*))prefetch_tensor)(&tensor, 0);
      if (((int (*)(void))peek_error)() != injected) status = -85;
    }
  }
  ((int (*)(void))clear_error)();
  ((void (*)(int, void*))release_memory)(0, memory);
  return status;
}

int dt_probe_native_scaled_rmsnorm(void* allocate_tensor, void* release_tensor, void* execute_command)
{
  typedef ccv_nnc_tensor_t* (*allocate_fn)(void*, ccv_nnc_tensor_param_t, int);
  typedef int (*execute_fn)(ccv_nnc_cmd_t, ccv_nnc_hint_t, int, ccv_nnc_tensor_t* const*, int,
    ccv_nnc_tensor_t* const*, int, ccv_nnc_stream_context_t*);
  const float scales[] = {1, 0.125f, 1.0f / 128, 0.0883883461f, 0, -0.125f};
  const int datatypes[] = {CCV_32F, CCV_16F, CCV_16BF};
  const float values[] = {0, 0.25f, -0.5f, 1};
  const uint16_t half[] = {0, 0x3400, 0xb800, 0x3c00};
  const uint16_t bfloat[] = {0, 0x3e80, 0xbf00, 0x3f80};
  for (int datatype_index = 0; datatype_index < 3; datatype_index++)
    for (int scale_index = 0; scale_index < 6; scale_index++)
      for (int affine = 0; affine < 2; affine++)
    {
      unsigned char data[4][6 * 128 * 4];
      memset(data, 0xff, sizeof(data));
      for (int element = 0; element < 6 * 128; element++)
      {
        const int index = element < 128 ? 0 : element % 4;
        if (datatypes[datatype_index] == CCV_32F) ((float*)data[0])[element] = values[index];
        else ((uint16_t*)data[0])[element] = datatypes[datatype_index] == CCV_16F ? half[index] : bfloat[index];
      }
      for (int element = 0; element < 128; element++)
        if (datatypes[datatype_index] == CCV_32F) ((float*)data[3])[element] = 2;
        else ((uint16_t*)data[3])[element] = 0x4000;
      ccv_nnc_tensor_t host[4] = {0};
      ccv_nnc_tensor_t* host_pointers[] = {host, host + 1, host + 2, host + 3};
      ccv_nnc_tensor_t* device[4] = {0};
      int status = 0;
      for (int index = 0; index < 4; index++)
      {
        host[index].info.type = CCV_TENSOR_CPU_MEMORY;
        host[index].info.format = CCV_TENSOR_FORMAT_NHWC;
        host[index].info.datatype = datatypes[datatype_index];
        const int shape[] = {1, index == 3 ? 1 : 3, index == 3 ? 1 : 2, index == 2 ? 1 : 128};
        memcpy(host[index].info.dim, shape, sizeof(shape));
        host[index].data.u8 = data[index];
        ccv_nnc_tensor_param_t parameters = host[index].info;
        parameters.type = CCV_TENSOR_GPU_MEMORY;
        device[index] = ((allocate_fn)allocate_tensor)(0, parameters, 0);
        if (!device[index]) { status = -60; break; }
      }
      const ccv_nnc_hint_t hint = {0};
      const ccv_nnc_cmd_t transfer = {.cmd = CCV_NNC_DATA_TRANSFER_FORWARD, .algorithm = -1};
      const ccv_nnc_cmd_t norm = {.cmd = CCV_NNC_RMSNORM_FORWARD, .algorithm = -1,
        .info = {.rmsnorm = {.epsilon = 1e-6f / 128, .elementwise_affine = affine,
          .scale = scales[scale_index], .count = 1, .axis = {3}}}};
      ccv_nnc_tensor_t* norm_inputs[] = {device[0], device[3]};
      if (!status) status = ((execute_fn)execute_command)(transfer, hint, 0, host_pointers, 4, device, 4, 0);
      if (!status) status = ((execute_fn)execute_command)(norm, hint, 0, norm_inputs, affine ? 2 : 1, device + 1, 2, 0);
      if (!status) status = ((execute_fn)execute_command)(transfer, hint, 0, device + 1, 2, host_pointers + 1, 2, 0);
      if (!status)
        for (int element = 0; element < 6 * 128; element++)
        {
          const float variance = element < 128 ? 0 : (0.25f * 0.25f + 0.5f * 0.5f + 1) / 4;
          const float expected = values[element < 128 ? 0 : element % 4] * scales[scale_index] * (affine ? 2 : 1) / sqrtf(variance + 1e-6f / 128);
          const float actual = dt_probe_float_value(data[1], datatypes[datatype_index], element);
          const float tolerance = datatypes[datatype_index] == CCV_32F ? 2e-6f : datatypes[datatype_index] == CCV_16F ? 0.002f : 0.015f;
          if (!isfinite(actual) || fabsf(actual - expected) > tolerance * fmaxf(fabsf(expected), 0.01f)) status = -61;
        }
      if (!status)
        for (int row = 0; row < 6; row++)
        {
          const float variance = row == 0 ? 0 : (0.25f * 0.25f + 0.5f * 0.5f + 1) / 4;
          const float expected = 1 / sqrtf(variance + 1e-6f / 128);
          const float actual = dt_probe_float_value(data[2], datatypes[datatype_index], row);
          const float tolerance = datatypes[datatype_index] == CCV_32F ? 2e-6f : datatypes[datatype_index] == CCV_16F ? 0.002f : 0.015f;
          if (!isfinite(actual) || fabsf(actual - expected) > tolerance * expected) status = -62;
        }
      for (int index = 0; index < 4; index++)
        if (device[index]) ((void (*)(ccv_nnc_tensor_t*))release_tensor)(device[index]);
      if (status) return status;
    }
  return 0;
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