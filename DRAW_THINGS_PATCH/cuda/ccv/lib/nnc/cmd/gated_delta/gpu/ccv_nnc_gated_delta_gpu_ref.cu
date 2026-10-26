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
static __device__ float _ccv_nnc_gated_delta_load(const void* data, const size_t index, const int datatype)
{
	if (datatype == CCV_16F) return __half2float(((const __half*)data)[index]);
	if (datatype == CCV_16BF) return __bfloat162float(((const __nv_bfloat16*)data)[index]);
	return ((const float*)data)[index];
}

static __global__ void _ccv_nnc_gated_delta_kernel(const int sequence, const int key_heads,
	const int value_heads, const int key_dim, const int value_dim, const int log_decay,
	const int channel_decay, const int checkpoints, const void* queries, const int query_type,
	const void* keys, const int key_type, const void* values, const int value_type,
	const float* decay, const void* beta, const int beta_type, const float* initial,
	void* output, float* final_state)
{
	__shared__ float partial[1024];
	const int value_row = blockIdx.x % value_dim;
	const int value_head = (blockIdx.x / value_dim) % value_heads;
	const int batch = blockIdx.x / ((size_t)value_dim * value_heads);
	const int key_head = value_head / (value_heads / key_heads);
	const int column = threadIdx.x;
	const size_t state_offset = (((size_t)batch * value_heads + value_head) * value_dim + value_row) * key_dim;
	float state = column < key_dim ? initial[state_offset + column] : 0.f;
	for (int timestep = 0; timestep < sequence; timestep++)
	{
		const size_t key_offset = (((size_t)batch * sequence + timestep) * key_heads + key_head) * key_dim;
		const size_t gate_offset = ((size_t)batch * sequence + timestep) * value_heads + value_head;
		const float key = column < key_dim ? _ccv_nnc_gated_delta_load(keys, key_offset + column, key_type) : 0.f;
		if (column < key_dim)
		{
			const float gate = decay[channel_decay ? gate_offset * key_dim + column : gate_offset];
			state *= log_decay ? expf(gate) : gate;
		}
		partial[column] = state * key;
		__syncthreads();
		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
		{
			if (column < stride) partial[column] += partial[column + stride];
			__syncthreads();
		}
		const size_t value_offset = gate_offset * value_dim + value_row;
		const float delta = (_ccv_nnc_gated_delta_load(values, value_offset, value_type) - partial[0]) *
			_ccv_nnc_gated_delta_load(beta, gate_offset, beta_type);
		__syncthreads();
		state += delta * key;
		partial[column] = column < key_dim ? state * _ccv_nnc_gated_delta_load(queries, key_offset + column, query_type) : 0.f;
		__syncthreads();
		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
		{
			if (column < stride) partial[column] += partial[column + stride];
			__syncthreads();
		}
		if (column == 0)
		{
			if (value_type == CCV_16F) ((__half*)output)[value_offset] = __float2half_rn(partial[0]);
			else if (value_type == CCV_16BF) ((__nv_bfloat16*)output)[value_offset] = __float2bfloat16_rn(partial[0]);
			else ((float*)output)[value_offset] = partial[0];
		}
		const int history = sequence - 1 - timestep;
		if (column < key_dim && history > 0 && history <= checkpoints)
		{
			const size_t checkpoint_offset = ((((size_t)batch * (checkpoints + 1) + history) * value_heads + value_head) * value_dim + value_row) * key_dim;
			final_state[checkpoint_offset + column] = state;
		}
	}
	if (column < key_dim)
	{
		const size_t final_offset = (((size_t)batch * (checkpoints + 1) * value_heads + value_head) * value_dim + value_row) * key_dim;
		final_state[final_offset + column] = state;
	}
}

static int _ccv_nnc_gated_delta_type(const int datatype)
{
	return datatype == CCV_32F || datatype == CCV_16F || datatype == CCV_16BF;
}

static int _ccv_nnc_gated_delta_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint,
	const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size,
	ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (input_size != 6 || output_size != 2) return CCV_NNC_EXEC_INVALID;
	for (int index = 0; index < 8; index++)
	{
		const ccv_nnc_tensor_view_t* const tensor = (const ccv_nnc_tensor_view_t*)(index < 6 ? inputs[index] : outputs[index - 6]);
		if (!tensor || !CCV_IS_TENSOR_CONTIGUOUS(tensor) || CCV_TENSOR_GET_MEMORY(tensor->info.type) != CCV_TENSOR_GPU_MEMORY)
			return CCV_NNC_EXEC_INVALID;
	}
	const ccv_nnc_tensor_t* const queries = inputs[0];
	const ccv_nnc_tensor_t* const keys = inputs[1];
	const ccv_nnc_tensor_t* const values = inputs[2];
	const ccv_nnc_tensor_t* const decay = inputs[3];
	const ccv_nnc_tensor_t* const beta = inputs[4];
	const ccv_nnc_tensor_t* const initial = inputs[5];
	ccv_nnc_tensor_t* const output = outputs[0];
	ccv_nnc_tensor_t* const final_state = outputs[1];
	if (ccv_nnc_tensor_nd(queries->info.dim) == 0)
	{
		for (int index = 1; index < 8; index++)
			if (ccv_nnc_tensor_nd((index < 6 ? inputs[index] : outputs[index - 6])->info.dim) != 0) return CCV_NNC_EXEC_INVALID;
		return CCV_NNC_EXEC_SUCCESS;
	}
	if (ccv_nnc_tensor_nd(queries->info.dim) != 4 || ccv_nnc_tensor_nd(keys->info.dim) != 4 ||
		ccv_nnc_tensor_nd(values->info.dim) != 4 || ccv_nnc_tensor_nd(initial->info.dim) != 4 ||
		ccv_nnc_tensor_nd(output->info.dim) != 4 || ccv_nnc_tensor_nd(final_state->info.dim) != 4 ||
		ccv_nnc_tensor_nd(beta->info.dim) != 3 ||
		!_ccv_nnc_gated_delta_type(queries->info.datatype) || !_ccv_nnc_gated_delta_type(keys->info.datatype) ||
		!_ccv_nnc_gated_delta_type(values->info.datatype) || !_ccv_nnc_gated_delta_type(beta->info.datatype) ||
		decay->info.datatype != CCV_32F || initial->info.datatype != CCV_32F || final_state->info.datatype != CCV_32F ||
		output->info.datatype != values->info.datatype) return CCV_NNC_EXEC_INVALID;
	const int batch = queries->info.dim[0], sequence = queries->info.dim[1];
	const int key_heads = queries->info.dim[2], key_dim = queries->info.dim[3];
	const int value_heads = values->info.dim[2], value_dim = values->info.dim[3];
	const int checkpoints = cmd.info.gated_delta.state_checkpoint_count;
	const int decay_rank = ccv_nnc_tensor_nd(decay->info.dim);
	if (batch < 1 || sequence < 1 || key_heads < 1 || value_heads < 1 || key_dim < 1 || value_dim < 1 ||
		key_dim > 1024 || value_heads % key_heads || checkpoints < 0 || checkpoints >= sequence ||
		(decay_rank != 3 && decay_rank != 4) || (decay_rank == 4 && decay->info.dim[3] != key_dim) ||
		(initial->data.u8 == final_state->data.u8 && checkpoints != 0)) return CCV_NNC_EXEC_INVALID;
	for (int axis = 0; axis < 4; axis++)
		if (queries->info.dim[axis] != keys->info.dim[axis] || values->info.dim[axis] != output->info.dim[axis])
			return CCV_NNC_EXEC_INVALID;
	if (values->info.dim[0] != batch || values->info.dim[1] != sequence ||
		beta->info.dim[0] != batch || beta->info.dim[1] != sequence || beta->info.dim[2] != value_heads ||
		decay->info.dim[0] != batch || decay->info.dim[1] != sequence || decay->info.dim[2] != value_heads ||
		initial->info.dim[0] != batch || initial->info.dim[1] != value_heads || initial->info.dim[2] != value_dim || initial->info.dim[3] != key_dim ||
		final_state->info.dim[0] != batch || final_state->info.dim[1] != value_heads * (checkpoints + 1) ||
		final_state->info.dim[2] != value_dim || final_state->info.dim[3] != key_dim) return CCV_NNC_EXEC_INVALID;
	const size_t rows = (size_t)batch * value_heads * value_dim;
	if (rows > 0x7fffffffU) return CCV_NNC_EXEC_INVALID;
	int threads = 32;
	while (threads < key_dim) threads <<= 1;
	_ccv_nnc_gated_delta_kernel<<<rows, threads, 0, ccv_nnc_stream_context_get_stream(stream_context)>>>(
		sequence, key_heads, value_heads, key_dim, value_dim, cmd.info.gated_delta.log_decay,
		decay_rank == 4, checkpoints, queries->data.u8, queries->info.datatype,
		keys->data.u8, keys->info.datatype, values->data.u8, values->info.datatype,
		decay->data.f32, beta->data.u8, beta->info.datatype, initial->data.f32,
		output->data.u8, final_state->data.f32);
	return cudaPeekAtLastError() == cudaSuccess ? CCV_NNC_EXEC_SUCCESS : CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gated_delta_forw;
}
#endif