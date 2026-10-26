extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>
#include <cuda_bf16.h>

#ifdef HAVE_CUDA
static __device__ float _ccv_nnc_rmsnorm_gated_load(const void* data, const size_t index, const int datatype)
{
	switch (datatype) {
		case CCV_16F: return __half2float(((const __half*)data)[index]);
		case CCV_16BF: return __bfloat162float(((const __nv_bfloat16*)data)[index]);
		default: return ((const float*)data)[index];
	}
}

static __global__ void _ccv_nnc_rmsnorm_gated_kernel(const int columns, const float epsilon,
	const void* values, const int value_datatype, const void* gate, const int gate_datatype,
	const void* scale, const int scale_datatype, void* output)
{
	__shared__ float partial[128];
	const size_t offset = (size_t)blockIdx.x * columns;
	float squared = 0;
	for (int column = threadIdx.x; column < columns; column += blockDim.x)
	{
		const float value = _ccv_nnc_rmsnorm_gated_load(values, offset + column, value_datatype);
		squared += value * value;
	}
	partial[threadIdx.x] = squared;
	__syncthreads();
	for (int stride = 64; stride > 0; stride >>= 1)
	{
		if (threadIdx.x < stride)
			partial[threadIdx.x] += partial[threadIdx.x + stride];
		__syncthreads();
	}
	const float inverse = rsqrtf(partial[0] / columns + epsilon);
	for (int column = threadIdx.x; column < columns; column += blockDim.x)
	{
		const float value = _ccv_nnc_rmsnorm_gated_load(values, offset + column, value_datatype);
		const float gate_value = _ccv_nnc_rmsnorm_gated_load(gate, offset + column, gate_datatype);
		const float exponential = expf(-fabsf(gate_value));
		const float sigmoid = gate_value >= 0 ? 1.f / (1.f + exponential) : exponential / (1.f + exponential);
		const float weight = scale ? _ccv_nnc_rmsnorm_gated_load(scale, column, scale_datatype) : 1.f;
		const float result = value * inverse * weight * gate_value * sigmoid;
		if (value_datatype == CCV_16F)
			((__half*)output)[offset + column] = __float2half_rn(result);
		else if (value_datatype == CCV_16BF)
			((__nv_bfloat16*)output)[offset + column] = __float2bfloat16_rn(result);
		else
			((float*)output)[offset + column] = result;
	}
}

static int _ccv_nnc_rmsnorm_gated_supported_type(const int datatype)
{
	return datatype == CCV_32F || datatype == CCV_16F || datatype == CCV_16BF;
}

static int _ccv_nnc_rmsnorm_gated_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint,
	const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size,
	ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (input_size != (cmd.info.rmsnorm_gated.elementwise_affine ? 3 : 2) || output_size != 1)
		return CCV_NNC_EXEC_INVALID;
	const ccv_nnc_tensor_view_t* const values = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const gate = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const scale = input_size == 3 ? (const ccv_nnc_tensor_view_t*)inputs[2] : 0;
	ccv_nnc_tensor_view_t* const output = (ccv_nnc_tensor_view_t*)outputs[0];
	if (!values || !gate || !output || (input_size == 3 && !scale))
		return CCV_NNC_EXEC_INVALID;
	if (CCV_TENSOR_GET_MEMORY(values->info.type) != CCV_TENSOR_GPU_MEMORY ||
		CCV_TENSOR_GET_MEMORY(gate->info.type) != CCV_TENSOR_GPU_MEMORY ||
		CCV_TENSOR_GET_MEMORY(output->info.type) != CCV_TENSOR_GPU_MEMORY ||
		(scale && CCV_TENSOR_GET_MEMORY(scale->info.type) != CCV_TENSOR_GPU_MEMORY))
		return CCV_NNC_EXEC_INVALID;
	const int rank = ccv_nnc_tensor_nd(values->info.dim);
	if (rank == 0)
		return ccv_nnc_tensor_nd(gate->info.dim) == 0 && ccv_nnc_tensor_nd(output->info.dim) == 0 ? CCV_NNC_EXEC_SUCCESS : CCV_NNC_EXEC_INVALID;
	if (cmd.info.rmsnorm_gated.count != 1 || cmd.info.rmsnorm_gated.axis[0] != rank - 1 ||
		!CCV_IS_TENSOR_CONTIGUOUS(values) || !CCV_IS_TENSOR_CONTIGUOUS(gate) || !CCV_IS_TENSOR_CONTIGUOUS(output) ||
		!_ccv_nnc_rmsnorm_gated_supported_type(values->info.datatype) ||
		!_ccv_nnc_rmsnorm_gated_supported_type(gate->info.datatype) || output->info.datatype != values->info.datatype ||
		!isfinite(cmd.info.rmsnorm_gated.epsilon) || cmd.info.rmsnorm_gated.epsilon <= 0)
		return CCV_NNC_EXEC_INVALID;
	for (int axis = 0; axis < CCV_NNC_MAX_DIM_ALLOC; axis++)
		if (values->info.dim[axis] != gate->info.dim[axis] || values->info.dim[axis] != output->info.dim[axis])
			return CCV_NNC_EXEC_INVALID;
	const int columns = values->info.dim[rank - 1];
	if (scale && (!CCV_IS_TENSOR_CONTIGUOUS(scale) || !_ccv_nnc_rmsnorm_gated_supported_type(scale->info.datatype) || ccv_nnc_tensor_count(scale->info) != columns))
		return CCV_NNC_EXEC_INVALID;
	const size_t rows = ccv_nnc_tensor_count(values->info) / columns;
	if (rows > 0x7fffffffU)
		return CCV_NNC_EXEC_INVALID;
	_ccv_nnc_rmsnorm_gated_kernel<<<rows, 128, 0, ccv_nnc_stream_context_get_stream(stream_context)>>>(
		columns, cmd.info.rmsnorm_gated.epsilon, values->data.u8, values->info.datatype,
		gate->data.u8, gate->info.datatype, scale ? scale->data.u8 : 0,
		scale ? scale->info.datatype : CCV_32F, output->data.u8);
	return cudaPeekAtLastError() == cudaSuccess ? CCV_NNC_EXEC_SUCCESS : CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_RMSNORM_GATED_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_rmsnorm_gated_forw;
}
#endif