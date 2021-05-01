from io import UnsupportedOperation
from layers.id_gen import IdGenerator
from layers.input import Input1D, InputImage
import fire
import tflite
from layers.conv2d import Conv2d
from layers.conv2d_dw import Conv2d_DW
from layers.avgpool_2d import AvgPool2D
from layers.layer import Layer
from layers.reshape import Reshape
from layers.fc import FC
from layers.softmax import Softmax
from layers.add import Add
from typing import Tuple
import numpy as np

def clean_data_int_op(buffer_data, is_32):
    num_bytes_skip = 4 if is_32 else 1 
    actual_res = []
    for i in range(0, len(buffer_data), num_bytes_skip):
        actual_res.append(int.from_bytes(buffer_data[i:i+num_bytes_skip], "little", signed=True))
    return np.array(actual_res)

def handle_conv2d_parsing(op: tflite.Operator, builtin_code: tflite.BuiltinOperator, graph: tflite.SubGraph, model: tflite.Model) -> Tuple[str, Layer]:
    op_opt = op.BuiltinOptions()

    opt = tflite.Conv2DOptions()
    opt.Init(op_opt.Bytes, op_opt.Pos)


    stride = opt.StrideW()

    input_ids = op.InputsAsNumpy()
    output_ids = op.OutputsAsNumpy()
    assert len(input_ids) == 3
    input_tensor_id, filter_tensor_id, bias_tensor_id = list(input_ids)

    input_tensor = graph.Tensors(input_tensor_id)
    input_shape = input_tensor.ShapeAsNumpy()
    assert len(input_shape) == 4
    _, input_width, input_height, input_depth = list(input_shape)

    filter_tensor = graph.Tensors(filter_tensor_id)
    filter_shape = filter_tensor.ShapeAsNumpy()
    assert len(filter_shape) == 4
    if builtin_code == tflite.BuiltinOperator.CONV_2D:
        filter_count, filter_width, filter_height, filter_in_channels = list(filter_shape)
    else:
        filter_in_channels, filter_width, filter_height, filter_count = list(filter_shape)
    filter_quantization = filter_tensor.Quantization()
    filter_scales = filter_quantization.ScaleAsNumpy()

    filter_data = clean_data_int_op(model.Buffers(filter_tensor.Buffer()).DataAsNumpy(), is_32=False).reshape(filter_shape)

    padding = opt.Padding()
    padding_size = 0
    if padding == tflite.Padding.SAME:
        padding_size = int(filter_width) // 2
    else:
        # PADDING should be valid in this case
        raise Exception(f"Unexpected padding type: {padding}")

    bias_tensor = graph.Tensors(bias_tensor_id)
    bias_data = clean_data_int_op(model.Buffers(bias_tensor.Buffer()).DataAsNumpy(), is_32=True)

    assert len(output_ids) == 1
    output_id = output_ids[0]
    output_tensor = graph.Tensors(output_id)
    output_shape = output_tensor.ShapeAsNumpy()
    output_batch_size, output_width, output_height, output_channels = list(output_shape)
    assert output_channels == filter_count

    output_quantization = output_tensor.Quantization().ScaleAsNumpy()
    assert len(output_quantization) == 1
    output_scale = output_quantization[0]

    activation_type = opt.FusedActivationFunction()
    followed_by_relu = activation_type == tflite.ActivationFunctionType.RELU

    output_name = output_tensor.Name()
    input_name = input_tensor.Name()

    if builtin_code == tflite.BuiltinOperator.CONV_2D:
        return output_name, Conv2d(
            input=None,
            filter_width=filter_width, filter_height=filter_height, filter_in_channels=filter_in_channels, filter_count=filter_count, filter_data=filter_data, bias_data=bias_data,
            input_width=input_width, input_height=input_height, stride=stride, padding=padding_size,
            followed_by_relu=followed_by_relu, output_scale=output_scale, filter_scales=filter_scales,
            output_width=output_width, output_height=output_height
        ), input_name
    else:
        return output_name, Conv2d_DW(
            input=None,
            filter_width=filter_width, filter_height=filter_height, filter_in_channels=filter_in_channels, filter_count=filter_count, filter_data=filter_data, bias_data=bias_data,
            input_width=input_width, input_height=input_height, stride=stride, padding=padding_size,
            followed_by_relu=followed_by_relu, output_scale=output_scale, filter_scales=filter_scales,
            output_width=output_width, output_height=output_height
        ), input_name

def handle_pool2d_parsing(op: tflite.Operator, builtin_code: tflite.BuiltinOperator, graph: tflite.SubGraph, model: tflite.Model) -> Tuple[str, Layer]:
    op_opt = op.BuiltinOptions()

    input_ids = op.InputsAsNumpy()
    assert len(input_ids) == 1
    input_tensor = graph.Tensors(input_ids[0])
    input_shape = input_tensor.ShapeAsNumpy()
    assert len(input_shape) == 4
    input_batch_size, input_width, input_height, input_depth = list(input_shape)

    output_ids = op.OutputsAsNumpy()
    assert len(output_ids) == 1
    output_tensor = graph.Tensors(output_ids[0])
    output_shape = output_tensor.ShapeAsNumpy()
    assert len(output_shape) == 4
    output_batch_size, output_width, output_height, output_depth = list(output_shape)
   
    input_name = input_tensor.Name()
    output_name = output_tensor.Name()
    layer = AvgPool2D(
        input=None,
        input_width=input_width,
        input_height=input_height,
        input_depth=input_depth,
        output_width=output_width,
        output_height=output_height,
        output_depth=output_depth,
    )

    return output_name, layer, input_name

def handle_reshape_parsing(op: tflite.Operator, builtin_code: tflite.BuiltinOperator, graph: tflite.SubGraph, model: tflite.Model) -> Tuple[str, Layer]:
    op_opt = op.BuiltinOptions()

    input_ids = op.InputsAsNumpy()
    assert len(input_ids) == 2
    input_tensor = graph.Tensors(input_ids[0])
    input_shape = input_tensor.ShapeAsNumpy()
    assert len(input_shape) == 4
    input_batch_size, input_width, input_height, input_depth = list(input_shape)

    output_ids = op.OutputsAsNumpy()
    assert len(output_ids) == 1
    output_tensor = graph.Tensors(output_ids[0])
    output_shape = output_tensor.ShapeAsNumpy()
    assert len(output_shape) == 2
    _, output_size = list(output_shape)
    
    input_name = input_tensor.Name()
    output_name = output_tensor.Name()
    layer = Reshape(input=None, output_size=output_size)
    return output_name, layer, input_name

def handle_fc_parsing(op: tflite.Operator, builtin_code: tflite.BuiltinOperator, graph: tflite.SubGraph, model: tflite.Model) -> Tuple[str, Layer]:
    op_opt = op.BuiltinOptions()
    opt = tflite.FullyConnectedOptions()
    opt.Init(op_opt.Bytes, op_opt.Pos)

    input_ids = op.InputsAsNumpy()
    output_ids = op.OutputsAsNumpy()
    assert len(input_ids) == 3
    input_tensor_id, weight_tensor_id, bias_tensor_id = list(input_ids)
    input_tensor = graph.Tensors(input_tensor_id)
    input_shape = input_tensor.ShapeAsNumpy()
    assert len(input_shape) == 2
    input_batch_size, input_size = list(input_shape)

    weight_tensor = graph.Tensors(weight_tensor_id)
    weight_shape = weight_tensor.ShapeAsNumpy()
    assert len(weight_shape) == 2
    weight_width, weight_height = list(weight_shape)
    weight_quantization = weight_tensor.Quantization()
    weight_quantization_scales = weight_quantization.ScaleAsNumpy()
    assert len(weight_quantization_scales) == 1
    weight_scale = weight_quantization_scales[0]
    weight_zero_points = weight_quantization.ZeroPointAsNumpy()
    assert len(weight_zero_points) == 1
    weight_zero_point = weight_zero_points[0]

    weight_data = clean_data_int_op(model.Buffers(weight_tensor.Buffer()).DataAsNumpy(), is_32=False).reshape(weight_shape)

    bias_tensor = graph.Tensors(bias_tensor_id)
    bias_data = clean_data_int_op(model.Buffers(bias_tensor.Buffer()).DataAsNumpy(), is_32=True)

    assert len(output_ids) == 1
    output_id = output_ids[0]
    output_tensor = graph.Tensors(output_id)
    output_shape = output_tensor.ShapeAsNumpy()
    assert len(output_shape) == 2
    output_batch_size, output_size = list(output_shape)
    output_quantization = output_tensor.Quantization()
 
    output_quantization_scales = output_quantization.ScaleAsNumpy()
    assert len(output_quantization_scales) == 1
    output_scale = output_quantization_scales[0]
   
    output_zero_points = output_quantization.ZeroPointAsNumpy()
    assert len(output_zero_points) == 1
    output_zero_point = output_zero_points[0]


    activation_type = opt.FusedActivationFunction()
    followed_by_relu = activation_type == tflite.ActivationFunctionType.RELU
    
    input_name = input_tensor.Name()
    output_name = output_tensor.Name()
    layer = FC(
        input=None, output_size=output_size, input_size=input_size,
        weight_data=weight_data, weight_scale=weight_scale, bias_data=bias_data,
        output_scale=output_scale, followed_by_relu=followed_by_relu,
        weight_zero_point=weight_zero_point,
        output_zero_point=output_zero_point
    )
    return output_name, layer, input_name

def handle_softmax_parsing(op: tflite.Operator, builtin_code: tflite.BuiltinOperator, graph: tflite.SubGraph, model: tflite.Model) -> Tuple[str, Layer]:
    op_opt = op.BuiltinOptions()
    opt = tflite.SoftmaxOptions()
    opt.Init(op_opt.Bytes, op_opt.Pos)

    input_ids = op.InputsAsNumpy()
    assert len(input_ids) == 1
    input_tensor = graph.Tensors(input_ids[0])

    output_ids = op.OutputsAsNumpy()
    assert len(output_ids) == 1
    output_id = output_ids[0]
    output_tensor = graph.Tensors(output_id)
    output_shape = output_tensor.ShapeAsNumpy()
    assert len(output_shape) == 2
    output_batch_size, output_size = list(output_shape)

    output_quantization = output_tensor.Quantization()
    output_quantization_scales = output_quantization.ScaleAsNumpy()
    assert len(output_quantization_scales) == 1
    out_scale = output_quantization_scales[0]
   
    output_zero_points = output_quantization.ZeroPointAsNumpy()
    assert len(output_zero_points) == 1
    output_zero_point = output_zero_points[0]


    input_name = input_tensor.Name()
    output_name = output_tensor.Name()
    layer = Softmax(
        input=None,
        output_size=output_size,
        out_scale=out_scale,
        output_zero_point=output_zero_point
    )
    return output_name, layer, input_name

def handle_add_parsing(op: tflite.Operator, builtin_code: tflite.BuiltinOperator, graph: tflite.SubGraph, model: tflite.Model) -> Tuple[str, Layer]:
    op_opt = op.BuiltinOptions()
    opt = tflite.AddOptions()
    opt.Init(op_opt.Bytes, op_opt.Pos)
    activation_type = opt.FusedActivationFunction()
    followed_by_relu = activation_type == tflite.ActivationFunctionType.RELU
    pot_scale_int16 = opt.PotScaleInt16()
    input_ids = op.InputsAsNumpy()
    assert len(input_ids) == 2
    input1_tensor_id, input2_tensor_id = list(input_ids)

    input1_tensor = graph.Tensors(input1_tensor_id)
    input1_shape = input1_tensor.ShapeAsNumpy()

    input2_tensor = graph.Tensors(input2_tensor_id)
    input2_shape = input2_tensor.ShapeAsNumpy()
    input_batch_size, input_width, input_height, input_depth = list(input2_shape)
    assert np.array_equal(input1_shape,input2_shape) # shape for both should be the same

    output_ids = op.OutputsAsNumpy()
    assert len(output_ids) == 1
    output_id = output_ids[0]
    output_tensor = graph.Tensors(output_id)
    output_shape = output_tensor.ShapeAsNumpy()
    assert np.array_equal(input1_shape,output_shape) # shape for both should be the same

    output_quantization = output_tensor.Quantization()
    output_quantization_scales = output_quantization.ScaleAsNumpy()
    assert len(output_quantization_scales) == 1
    output_scale = output_quantization_scales[0]

   
    output_zero_points = output_quantization.ZeroPointAsNumpy()
    assert len(output_zero_points) == 1
    output_zero_point = output_zero_points[0]


    output_name = output_tensor.Name()
    input_names = [input1_tensor.Name(),  input2_tensor.Name()]
    layer = Add(
        inputs = None,
        output_width=input_width,
        output_height=input_height,
        output_depth=input_depth,
        followed_by_relu=followed_by_relu,
        output_scale=output_scale,
        pot_scale_int16=pot_scale_int16
    )
    return output_name, layer, input_names

def parse(file_name): 
    with open(file_name, 'rb') as f:
        buf = f.read()
        model = tflite.Model.GetRootAsModel(buf, 0)

    assert model.SubgraphsLength() == 1
    graph = model.Subgraphs(0)
    first_input_tensor = graph.Tensors(graph.Inputs(0))
    if first_input_tensor.ShapeLength() == 4:
        image_batch_size, image_width, image_height, image_channels = list(first_input_tensor.ShapeAsNumpy())
        input_tensor_scales = first_input_tensor.Quantization().ScaleAsNumpy()
        assert(len(input_tensor_scales) == 1)
        output_image_scale = input_tensor_scales[0]
        first_layer = InputImage(input_width=image_width, input_height=image_height, input_depth=image_channels, output_scale=output_image_scale)
    elif first_input_tensor.ShapeLength() == 2:
        data_batch_size, input_size = list(first_input_tensor.ShapeAsNumpy())
        # input_tensor_scales = first_input_tensor.Quantization().ScaleAsNumpy()
        # assert(len(input_tensor_scales) == 1)
        # output_scale = input_tensor_scales[0]
        output_scale = 1 # TODO input quantization not showing up for 1d data. 
        first_layer = Input1D(input_size=input_size, output_scale=output_scale)
    else:
        raise UnsupportedOperation(f"Unsupported input size {first_input_tensor.ShapeLength()}")

    input_name = first_input_tensor.Name()
    name_to_layer = {}
    name_to_layer[input_name] = first_layer, None
    output_name = graph.Tensors(graph.Outputs(0)).Name()

    for i in range(graph.OperatorsLength()):
        op = graph.Operators(i)
        op_code = model.OperatorCodes(op.OpcodeIndex())
        builtin_code = op_code.BuiltinCode()
        if tflite.BuiltinOperator.CONV_2D == builtin_code or tflite.BuiltinOperator.DEPTHWISE_CONV_2D == builtin_code:
            output_name, layer, in_name = handle_conv2d_parsing(op, builtin_code, graph, model)
        
        elif tflite.BuiltinOperator.AVERAGE_POOL_2D == builtin_code:
            output_name, layer, in_name = handle_pool2d_parsing(op, builtin_code, graph, model)

        elif tflite.BuiltinOperator.RESHAPE == builtin_code:
            output_name, layer, in_name = handle_reshape_parsing(op, builtin_code, graph, model)

        elif tflite.BuiltinOperator.FULLY_CONNECTED == builtin_code:
            output_name, layer, in_name = handle_fc_parsing(op, builtin_code, graph, model)

        elif tflite.BuiltinOperator.SOFTMAX == builtin_code:
            output_name, layer, in_name = handle_softmax_parsing(op, builtin_code, graph, model)
        elif tflite.BuiltinOperator.ADD == builtin_code:
            output_name, layer, in_name = handle_add_parsing(op, builtin_code, graph, model)
        else:
            name = tflite.opcode2name(builtin_code)
            raise UnsupportedOperation(f"Unsupported operator {name}")

        name_to_layer[output_name] = (layer, in_name)
    
    for out_name in name_to_layer.keys():
        layer, input_names = name_to_layer[out_name]
        if input_names:
            if isinstance(input_names, list) and isinstance(layer, Add):
                layer.inputs = [name_to_layer[name][0] for name in input_names]
            else:
                layer.input = name_to_layer[input_names][0]
            
    output_layer, _ = name_to_layer[output_name]
    return first_layer, output_layer, name_to_layer

def generate(file_name, data_file, runtime_file):
    _, final_layer, _ = parse(file_name)
    data_buffer = ["#include \"include/gemmini_nn_mini.h\""]
    runtime_buffer = [f"#include \"{data_file}\"", "void run() {"]
  
    output_name = final_layer.generate_gemmini_cached(data_buffer, runtime_buffer, IdGenerator(), {})
    runtime_buffer.append("}")
    data_buffer.append(f"#define final_output_matrix {output_name}")

    with open(data_file, "w") as data:
        data.write("\n".join(data_buffer))
    with open(runtime_file, "w") as runtime:
        runtime.write("\n".join(runtime_buffer))

def simulate(file_name, input):
    input_layer, output_layer, name_to_layer = parse(file_name)
    runtime_order = []
    cache = { id(input_layer): input }
    output_name = output_layer.simulate_python_cached(runtime_order, cache)
    return runtime_order, cache, name_to_layer

if __name__ == '__main__': 
    fire.Fire(generate)
