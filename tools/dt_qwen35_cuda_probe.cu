#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "nnc/cmd/rmsnorm_gated/gpu/ccv_nnc_rmsnorm_gated_gpu_ref.cu"
#include "nnc/cmd/gated_delta/gpu/ccv_nnc_gated_delta_gpu_ref.cu"

extern "C" cudaStream_t ccv_nnc_stream_context_get_stream(const ccv_nnc_stream_context_t* context)
{
	(void)context;
	return 0;
}

static ccv_nnc_tensor_t dt_tensor(void* data, const int rows, const int columns, const int datatype)
{
	ccv_nnc_tensor_t result = {};
	result.data.u8 = (unsigned char*)data;
	result.info.type = CCV_TENSOR_GPU_MEMORY;
	result.info.format = CCV_TENSOR_FORMAT_NHWC;
	result.info.datatype = datatype;
	result.info.dim[0] = rows;
	result.info.dim[1] = columns;
	return result;
}

struct DTBuffer {
	void* device = 0;
	std::vector<unsigned char> host;
	DTBuffer(const float* values, const size_t count, const int datatype)
	{
		host.resize(count * (datatype == CCV_32F ? 4 : 2));
		for (size_t index = 0; index < count; index++)
			if (datatype == CCV_16F) ((__half*)host.data())[index] = __float2half_rn(values[index]);
			else if (datatype == CCV_16BF) ((__nv_bfloat16*)host.data())[index] = __float2bfloat16_rn(values[index]);
			else ((float*)host.data())[index] = values[index];
		if (cudaMalloc(&device, host.size()) == cudaSuccess)
			if (cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice) != cudaSuccess)
			{
				cudaFree(device);
				device = 0;
			}
	}
	~DTBuffer() { if (device) cudaFree(device); }
	bool read(float* values, const int datatype)
	{
		if (cudaMemcpy(host.data(), device, host.size(), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
		for (size_t index = 0; index < host.size() / (datatype == CCV_32F ? 4 : 2); index++)
			values[index] = datatype == CCV_16F ? __half2float(((__half*)host.data())[index]) :
				datatype == CCV_16BF ? __bfloat162float(((__nv_bfloat16*)host.data())[index]) : ((float*)host.data())[index];
		return true;
	}
};

extern "C" int dt_cuda_rmsnorm(const int rows, const int columns, const int datatype,
	const int gate_datatype, const int scale_datatype, const int affine, const int inplace,
	const float* values, const float* gate, const float* scale, float* output)
{
	DTBuffer value_buffer(values, (size_t)rows * columns, datatype);
	DTBuffer gate_buffer(gate, (size_t)rows * columns, gate_datatype);
	DTBuffer scale_buffer(scale, columns, scale_datatype);
	DTBuffer output_buffer(values, (size_t)rows * columns, datatype);
	if (!value_buffer.device || !gate_buffer.device || !scale_buffer.device || !output_buffer.device) return -10;
	ccv_nnc_tensor_t tensors[] = {
		dt_tensor(value_buffer.device, rows, columns, datatype), dt_tensor(gate_buffer.device, rows, columns, gate_datatype),
		dt_tensor(scale_buffer.device, 1, columns, scale_datatype),
		dt_tensor(inplace ? value_buffer.device : output_buffer.device, rows, columns, datatype)
	};
	ccv_nnc_tensor_t* inputs[] = {tensors, tensors + 1, tensors + 2};
	ccv_nnc_tensor_t* outputs[] = {tensors + 3};
	ccv_nnc_cmd_t command = {};
	command.info.rmsnorm_gated.epsilon = 1e-6;
	command.info.rmsnorm_gated.elementwise_affine = affine;
	command.info.rmsnorm_gated.count = 1;
	command.info.rmsnorm_gated.axis[0] = 1;
	const ccv_nnc_hint_t hint = {};
	const int status = _ccv_nnc_rmsnorm_gated_forw(command, hint, 0, inputs, affine ? 3 : 2, outputs, 1, 0);
	if (status != CCV_NNC_EXEC_SUCCESS) return status;
	if (cudaDeviceSynchronize() != cudaSuccess) return -11;
	return (inplace ? value_buffer : output_buffer).read(output, datatype) ? 0 : -12;
}

extern "C" int dt_cuda_rmsnorm_contract(const int mode)
{
	ccv_nnc_tensor_t tensors[] = {
		dt_tensor(0, mode ? 2 : 0, mode ? 3 : 0, CCV_32F),
		dt_tensor(0, mode ? 2 : 0, mode ? 3 : 0, CCV_32F),
		dt_tensor(0, mode ? 2 : 0, mode ? 3 : 0, CCV_32F)
	};
	ccv_nnc_tensor_t* inputs[] = {tensors, tensors + 1};
	ccv_nnc_tensor_t* outputs[] = {tensors + 2};
	ccv_nnc_cmd_t command = {};
	command.info.rmsnorm_gated.epsilon = 1e-6;
	command.info.rmsnorm_gated.count = 1;
	command.info.rmsnorm_gated.axis[0] = mode == 1 ? 0 : 1;
	if (mode == 2) tensors[1].info.dim[1] = 4;
	if (mode == 3) tensors[0].info.datatype = CCV_64F;
	if (mode == 4) tensors[0].info.type = CCV_TENSOR_CPU_MEMORY;
	const ccv_nnc_hint_t hint = {};
	return _ccv_nnc_rmsnorm_gated_forw(command, hint, 0, inputs, 2, outputs, 1, 0);
}

extern "C" int dt_cuda_gated_delta(const int* dimensions, const int datatype,
	const int log_decay, const int channel_decay, const int checkpoints, const int inplace,
	const float* queries, const float* keys, const float* values, const float* decay,
	const float* beta, const float* initial, float* output, float* final_state)
{
	const int batch = dimensions[0], sequence = dimensions[1], key_heads = dimensions[2];
	const int value_heads = dimensions[3], key_dim = dimensions[4], value_dim = dimensions[5];
	const size_t query_count = (size_t)batch * sequence * key_heads * key_dim;
	const size_t value_count = (size_t)batch * sequence * value_heads * value_dim;
	const size_t gate_count = (size_t)batch * sequence * value_heads;
	const size_t state_count = (size_t)batch * value_heads * value_dim * key_dim;
	std::vector<float> empty_state(state_count * (checkpoints + 1), 0);
	DTBuffer query_buffer(queries, query_count, datatype), key_buffer(keys, query_count, datatype);
	DTBuffer value_buffer(values, value_count, datatype), decay_buffer(decay, gate_count * (channel_decay ? key_dim : 1), CCV_32F);
	DTBuffer beta_buffer(beta, gate_count, datatype), initial_buffer(initial, state_count, CCV_32F);
	DTBuffer output_buffer(values, value_count, datatype), state_buffer(empty_state.data(), empty_state.size(), CCV_32F);
	if (!query_buffer.device || !key_buffer.device || !value_buffer.device || !decay_buffer.device ||
		!beta_buffer.device || !initial_buffer.device || !output_buffer.device || !state_buffer.device) return -10;
	ccv_nnc_tensor_t tensors[] = {
		dt_tensor(query_buffer.device, 0, 0, datatype), dt_tensor(key_buffer.device, 0, 0, datatype),
		dt_tensor(value_buffer.device, 0, 0, datatype), dt_tensor(decay_buffer.device, 0, 0, CCV_32F),
		dt_tensor(beta_buffer.device, 0, 0, datatype), dt_tensor(initial_buffer.device, 0, 0, CCV_32F),
		dt_tensor(output_buffer.device, 0, 0, datatype), dt_tensor(inplace ? initial_buffer.device : state_buffer.device, 0, 0, CCV_32F)
	};
	const int shapes[8][4] = {
		{batch, sequence, key_heads, key_dim}, {batch, sequence, key_heads, key_dim},
		{batch, sequence, value_heads, value_dim}, {batch, sequence, value_heads, channel_decay ? key_dim : 0},
		{batch, sequence, value_heads, 0}, {batch, value_heads, value_dim, key_dim},
		{batch, sequence, value_heads, value_dim}, {batch, value_heads * (checkpoints + 1), value_dim, key_dim}
	};
	for (int index = 0; index < 8; index++) memcpy(tensors[index].info.dim, shapes[index], sizeof(shapes[index]));
	ccv_nnc_tensor_t* inputs[] = {tensors, tensors + 1, tensors + 2, tensors + 3, tensors + 4, tensors + 5};
	ccv_nnc_tensor_t* outputs[] = {tensors + 6, tensors + 7};
	ccv_nnc_cmd_t command = {};
	command.info.gated_delta.log_decay = log_decay;
	command.info.gated_delta.state_checkpoint_count = checkpoints;
	const ccv_nnc_hint_t hint = {};
	const int status = _ccv_nnc_gated_delta_forw(command, hint, 0, inputs, 6, outputs, 2, 0);
	if (status != CCV_NNC_EXEC_SUCCESS) return status;
	if (cudaDeviceSynchronize() != cudaSuccess) return -11;
	if (!output_buffer.read(output, datatype)) return -12;
	return (inplace ? initial_buffer : state_buffer).read(final_state, CCV_32F) ? 0 : -13;
}

extern "C" int dt_cuda_gated_delta_contract(const int mode)
{
	ccv_nnc_tensor_t tensors[8];
	const int shapes[8][4] = {
		{1, 3, 2, 4}, {1, 3, 2, 4}, {1, 3, 4, 5}, {1, 3, 4, 0},
		{1, 3, 4, 0}, {1, 4, 5, 4}, {1, 3, 4, 5}, {1, 4, 5, 4}
	};
	for (int index = 0; index < 8; index++)
	{
		tensors[index] = dt_tensor(0, 0, 0, CCV_32F);
		if (mode) memcpy(tensors[index].info.dim, shapes[index], sizeof(shapes[index]));
	}
	ccv_nnc_cmd_t command = {};
	if (mode == 1) tensors[5].info.dim[3] = 3;
	if (mode == 2) tensors[2].info.dim[2] = 3;
	if (mode == 3) tensors[5].info.datatype = CCV_16F;
	if (mode == 4) tensors[0].info.type = CCV_TENSOR_CPU_MEMORY;
	if (mode == 5) command.info.gated_delta.state_checkpoint_count = 3;
	ccv_nnc_tensor_t* inputs[] = {tensors, tensors + 1, tensors + 2, tensors + 3, tensors + 4, tensors + 5};
	ccv_nnc_tensor_t* outputs[] = {tensors + 6, tensors + 7};
	const ccv_nnc_hint_t hint = {};
	return _ccv_nnc_gated_delta_forw(command, hint, 0, inputs, 6, outputs, 2, 0);
}