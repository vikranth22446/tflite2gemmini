import sys
import json
import textwrap
import numpy as np

netron_json = sys.argv[1]
netron_json_new_format = sys.argv[2]

def generate_input_loader(output_id, input_node, runtime_lines):
  _, input_width, input_height, input_depth = input_node["_arguments"][0]["_type"]["_shape"]["_dimensions"]
  runtime_lines.append(f"#define output_{output_id}_dim {input_height}*{input_width}*{input_depth}")
  runtime_lines.append(f"elem_t output_{output_id}[{input_height * input_width}][{input_depth}];")

def get_attribute_by_name(node, name):
  for attr in node["_attributes"]:
    if attr["_name"] == name:
      return attr["_value"]
  raise Exception("not found")

def get_input_by_name(node, input_name):
  for _input in node["_inputs"]:
    if _input["_name"] == input_name:
      return _input
  raise Exception("not found")

def get_data_from_dict(dic, is_32):
  res = []
  for i in range(len(dic.keys())):
    res.append(dic[str(i)])
  
  if is_32:
    actual_res = []
    for i in range(0, len(res), 4):
      actual_res.append(int.from_bytes(res[i:i+4], "little", signed=True))
    return actual_res
  else:
    actual_res = []
    for i in range(0, len(res), 1):
      actual_res.append(int.from_bytes(res[i:i+1], "little", signed=True))
    return actual_res

def indexed_dict_to_list(dictionary):
  res = []
  for i in range(len(dictionary.keys())):
    res.append(dictionary[str(i)])
  return res

def np2carray(arr):
  if np.isscalar(arr):
    return str(arr)
  else:
    return "{ " + ", ".join(map(np2carray, arr)) + " }"

def generate_conv2d(output_id, node, runtime_lines, data_lines, compute_order_renaming, quantization_tensor_mappings):
  followed_by_relu = "_chain" in node and len(node["_chain"]) == 1 and node["_chain"][0]["_type"]["name"] == "Relu"
  if followed_by_relu:
    node["_chain"] = [] # don't use the default chain logic
  input_attribute = get_input_by_name(node, "input")
  filter_attribute = get_input_by_name(node, "filter")
  bias_attribute = get_input_by_name(node, "bias")
  assert len(node["_outputs"]) == 1
  output_attribute = node["_outputs"][0]
  assert len(output_attribute["_arguments"]) == 1
  output_quantization = output_attribute["_arguments"][0]["_quantization"]
  output_quant_value = output_quantization.split("≤")[1].split("*")[0].strip() # 0 ≤ 0.04099825397133827 * (q - -128) ≤ 10.454554557800293

  input_quantization = input_attribute["_arguments"][0]["_quantization"]
  input_quant_value = input_quantization.split("≤")[1].split("*")[0].strip() # 0 ≤ 0.04099825397133827 * (q - -128) ≤ 10.454554557800293

  assert len(input_attribute["_arguments"]) == 1
  assert len(filter_attribute["_arguments"]) == 1
  assert len(bias_attribute["_arguments"]) == 1

  quantization_for_filter = quantization_tensor_mappings[filter_attribute["_arguments"][0]["_name"]]["quantization"]
  assert quantization_for_filter["quantized_dimension"] == 0 # output channels each have their own scale
  quantization_scales = indexed_dict_to_list(quantization_for_filter["scale"])
  output_multipliers = [ (float(input_quant_value) * axis_scale) / float(output_quant_value) for axis_scale in quantization_scales ]

  input_id = compute_order_renaming[input_attribute["_arguments"][0]["_name"]]
  input_variable_name = f"output_{input_id}"
  _, input_width, input_height, input_depth = input_attribute["_arguments"][0]["_type"]["_shape"]["_dimensions"]
  assert input_width == input_height

  _, output_width, output_height, output_depth = output_attribute["_arguments"][0]["_type"]["_shape"]["_dimensions"]
  assert output_width == output_height
  
  filter_dimensions = filter_attribute["_arguments"][0]["_initializer"]["_type"]["_shape"]["_dimensions"]
  filter_data  = get_data_from_dict(filter_attribute["_arguments"][0]["_initializer"]["_data"], False)

  filter_count, filter_width, filter_height, filter_in_channels = filter_dimensions
  assert filter_width == filter_height # Verify Square filters
  assert filter_count == output_depth
  assert output_depth == len(quantization_scales)
  
  bias_dimensions = bias_attribute["_arguments"][0]["_initializer"]["_type"]["_shape"]["_dimensions"]
  bias_data = get_data_from_dict(bias_attribute["_arguments"][0]["_initializer"]["_data"], True)
  assert output_depth == len(bias_data)

  padding = get_attribute_by_name(node, "padding")
  padding_size = 0
  if padding == "SAME":
    padding_size = filter_width // 2
  else:
    raise Exception(f"Unexpected padding type: {padding}")
  stride_w = get_attribute_by_name(node, "stride_w")
  stride_h = get_attribute_by_name(node, "stride_h")
  assert stride_w == stride_h

  fused_activation_function = get_attribute_by_name(node, "fused_activation_function")
  dilation_w_factor = get_attribute_by_name(node, "dilation_w_factor")
  dilation_h_factor = get_attribute_by_name(node, "dilation_h_factor")
  assert dilation_w_factor == 1
  assert dilation_h_factor == 1

  total_filter_size = filter_width * filter_height * filter_count * filter_in_channels
  assert total_filter_size == len(filter_data)

  # input has order height, width, depth
  filter_data = np.array(filter_data).reshape((filter_count, filter_height, filter_width, filter_in_channels))
  filter_data_output_last = filter_data.transpose((1, 2, 3, 0))

  data_lines.append(f"const acc_t output_{output_id}_bias[{len(bias_data)}] = {np2carray(bias_data)};")
  data_lines.append(f"const float output_{output_id}_multipliers[{output_depth}] = {np2carray(output_multipliers)};")
  data_lines.append(f"elem_t output_{output_id}[{output_height} * {output_width}][{output_depth}];")

  if filter_width != 1 or True:
    filter_data_orig_input_flattened = filter_data.reshape((filter_count, filter_height * filter_width * filter_in_channels))
    data_lines.append(f"const elem_t output_{output_id}_weight_filterFirst[{filter_count}][{filter_width * filter_height * filter_in_channels}] = {np2carray(filter_data_orig_input_flattened)};")

    runtime_lines.append(textwrap.dedent(f"""\
      conv_auto_multiscale(
        1, {input_width}, {input_depth},
        {output_depth}, {output_width},
        {stride_w}, {padding_size}, {filter_width},
        
        {input_variable_name}, output_{output_id}_weight_filterFirst, output_{output_id}_bias, output_{output_id},

        {"RELU" if followed_by_relu else "NO_ACTIVATION"}, output_{output_id}_multipliers, 0,
        1, 1, 0,
        WS
      );"""
    ))
  else:
    # 1x1 convs can be implemented as matmul
    assert filter_height == 1
    assert filter_width == 1
    filter_data_output_last_with_input_flattened = filter_data_output_last.reshape((filter_height * filter_width * filter_in_channels, filter_count))
    data_lines.append(f"const elem_t output_{output_id}_weight_filterLast[{filter_width * filter_height * filter_in_channels}][{filter_count}] = {np2carray(filter_data_output_last_with_input_flattened)};")
    runtime_lines.append(textwrap.dedent(f"""\
      tiled_matmul_nn_auto_multiscale(
        {input_width * input_height}, {output_depth}, {filter_width * filter_height * filter_in_channels},
        {input_variable_name}, output_{output_id}_weight_filterLast, output_{output_id}_bias, output_{output_id},
        {"RELU" if followed_by_relu else "NO_ACTIVATION"}, output_{output_id}_multipliers, 0, true,
        WS, false, "conv_{output_id}"
      );"""
    ))
  runtime_lines.append(textwrap.dedent(f"""\
    printf("output_{output_id} = ");
    display_im({output_height}, {output_width}, {output_depth}, output_{output_id});"""
  ))

def generate_dw_conv2d(output_id, node, runtime_lines, data_lines, compute_order_renaming, quantization_tensor_mappings):
  followed_by_relu = "_chain" in node and len(node["_chain"]) == 1 and node["_chain"][0]["_type"]["name"] == "Relu"
  if followed_by_relu:
    node["_chain"] = [] # don't use the default chain logic

  input_attribute = get_input_by_name(node, "input")
  weights_attribute = get_input_by_name(node, "weights")
  bias_attribute = get_input_by_name(node, "bias")
  assert len(node["_outputs"]) == 1
  output_attribute = node["_outputs"][0]
  assert len(output_attribute["_arguments"]) == 1
  output_quantization = output_attribute["_arguments"][0]["_quantization"]
  output_quant_value = float(output_quantization.split("≤")[1].split("*")[0].strip()) # 0 ≤ 0.04099825397133827 * (q - -128) ≤ 10.454554557800293

  input_quantization = input_attribute["_arguments"][0]["_quantization"]
  input_quant_value = float(input_quantization.split("≤")[1].split("*")[0].strip()) # 0 ≤ 0.04099825397133827 * (q - -128) ≤ 10.454554557800293

  quantization_for_filter = quantization_tensor_mappings[weights_attribute["_arguments"][0]["_name"]]["quantization"]
  assert quantization_for_filter["quantized_dimension"] == 3 # output channels each have their own scale
  quantization_scales = indexed_dict_to_list(quantization_for_filter["scale"])
  output_multipliers = [ (float(input_quant_value) * axis_scale) / float(output_quant_value) for axis_scale in quantization_scales ]

  assert len(input_attribute["_arguments"]) == 1
  assert len(weights_attribute["_arguments"]) == 1
  assert len(bias_attribute["_arguments"]) == 1

  input_id = compute_order_renaming[input_attribute["_arguments"][0]["_name"]]
  input_variable_name = f"output_{input_id}"
  _, input_width, input_height, input_depth = input_attribute["_arguments"][0]["_type"]["_shape"]["_dimensions"]
  assert input_width == input_height

  _, output_width, output_height, output_depth = output_attribute["_arguments"][0]["_type"]["_shape"]["_dimensions"]
  assert output_width == output_height
  
  weights_dimensions = weights_attribute["_arguments"][0]["_initializer"]["_type"]["_shape"]["_dimensions"]
  weights_data  = get_data_from_dict(weights_attribute["_arguments"][0]["_initializer"]["_data"], False)

  weights_in_channels, weights_width, weights_height, weights_count = weights_dimensions
  assert weights_width == weights_height # Verify Square filters
  assert weights_in_channels == 1
  assert input_depth == output_depth
  assert weights_count == output_depth

  total_weights_size = weights_width * weights_height * weights_count * weights_in_channels
  assert total_weights_size == len(weights_data)

  weights_data = np.array(weights_data).reshape((weights_in_channels, weights_height, weights_width, weights_count))
  weights_data_count_first = weights_data.transpose((3, 0, 1, 2)).reshape((weights_count, weights_height * weights_width))
  
  bias_dimensions = bias_attribute["_arguments"][0]["_initializer"]["_type"]["_shape"]["_dimensions"]
  bias_data = get_data_from_dict(bias_attribute["_arguments"][0]["_initializer"]["_data"], True)
  assert output_depth == len(bias_data)

  padding = get_attribute_by_name(node, "padding")
  padding_size = 0
  if padding == "SAME":
    padding_size = weights_width // 2
  else:
    raise Exception(f"Unexpected padding type: {padding}")

  stride_w = get_attribute_by_name(node, "stride_w")
  stride_h = get_attribute_by_name(node, "stride_h")
  assert stride_w == stride_h

  fused_activation_function = get_attribute_by_name(node, "fused_activation_function")
  dilation_w_factor = get_attribute_by_name(node, "dilation_w_factor")
  dilation_h_factor = get_attribute_by_name(node, "dilation_h_factor")
  assert dilation_w_factor == 1
  assert dilation_h_factor == 1

  data_lines.append(f"elem_t output_{output_id}_weight[{weights_count}][{weights_height * weights_width}] = {np2carray(weights_data_count_first)};")
  data_lines.append(f"acc_t output_{output_id}_bias[{len(bias_data)}] = {np2carray(bias_data)};")
  data_lines.append(f"const float output_{output_id}_multipliers[{output_depth}] = {np2carray(output_multipliers)};")

  data_lines.append(f"elem_t output_{output_id}[{output_height} * {output_width}][{output_depth}];")

  runtime_lines.append(textwrap.dedent(f"""\
    conv_auto_dw_multiscale(
      1, {input_width}, {input_depth},
      {output_depth}, {output_width},
      {stride_w}, {padding_size}, {weights_width},

      {input_variable_name},  output_{output_id}_weight, output_{output_id}_bias, output_{output_id},

      {"RELU" if followed_by_relu else "NO_ACTIVATION"}, output_{output_id}_multipliers, 0,
      1, 0, 0,
      WS
    );"""
  ))
  runtime_lines.append(textwrap.dedent(f"""\
    printf("output_{output_id} = ");
    display_im({output_height}, {output_width}, {output_depth}, output_{output_id});"""
  ))

def generate_avgPool2d(output_id, node, runtime_lines, data_lines, compute_order_renaming, quantization_tensor_mappings):
  input_attribute = get_input_by_name(node, "input")
  input_id = compute_order_renaming[input_attribute["_arguments"][0]["_name"]]
  input_variable_name = f"output_{input_id}"

  assert len(node["_outputs"]) == 1
  output_attribute = node["_outputs"][0]
  assert len(output_attribute["_arguments"]) == 1

  _, input_width, input_height, input_depth = input_attribute["_arguments"][0]["_type"]["_shape"]["_dimensions"]
  _, output_width, output_height, output_depth = output_attribute["_arguments"][0]["_type"]["_shape"]["_dimensions"]
  assert input_depth == output_depth
  assert output_height == 1
  assert output_width == 1

  data_lines.append(f"elem_t output_{output_id}[{output_height} * {output_width}][{output_depth}];")
  runtime_lines.append(textwrap.dedent(f"""\
    compute_average_pooling(
      {input_width}, {input_depth},
      {input_variable_name},
      output_{output_id}
    );"""
  ))
  runtime_lines.append(textwrap.dedent(f"""\
    printf("output_{output_id} = ");
    display_im({output_height}, {output_width}, {output_depth}, output_{output_id});"""
  ))

def generate_reshape(output_id, node, runtime_lines, data_lines, compute_order_renaming, quantization_tensor_mappings):
  input_attribute = get_input_by_name(node, "data")
  input_id = compute_order_renaming[input_attribute["_arguments"][0]["_name"]]
  input_variable_name = f"output_{input_id}"

  assert len(node["_outputs"]) == 1
  output_attribute = node["_outputs"][0]
  assert len(output_attribute["_arguments"]) == 1
  output_dims = output_attribute["_arguments"][0]["_type"]["_shape"]["_dimensions"][1:]
  assert len(output_dims) == 1

  runtime_lines.append(f"elem_t (*output_{output_id})[{output_dims[0]}] = {input_variable_name};")

def generate_fc(output_id, node, runtime_lines, data_lines, compute_order_renaming, quantization_tensor_mappings):
  followed_by_relu = "_chain" in node and len(node["_chain"]) == 1 and node["_chain"][0]["_type"]["name"] == "Relu"
  assert not followed_by_relu

  input_attribute = get_input_by_name(node, "input")
  weights_attribute = get_input_by_name(node, "weights")
  bias_attribute = get_input_by_name(node, "bias")
  assert len(node["_outputs"]) == 1
  output_attribute = node["_outputs"][0]
  assert len(output_attribute["_arguments"]) == 1
  
  weights_quantization = weights_attribute["_arguments"][0]["_quantization"]
  weights_quant_value = float(weights_quantization.split("≤")[1].split("*")[0].strip()) # 0 ≤ 0.04099825397133827 * (q - -128) ≤ 10.454554557800293

  output_quantization = output_attribute["_arguments"][0]["_quantization"]
  output_quant_value = float(output_quantization.split("≤")[1].split("*")[0].strip()) # 0 ≤ 0.04099825397133827 * (q - -128) ≤ 10.454554557800293

  input_quantization = input_attribute["_arguments"][0]["_quantization"]
  input_quant_value = float(input_quantization.split("≤")[1].split("*")[0].strip()) # 0 ≤ 0.04099825397133827 * (q - -128) ≤ 10.454554557800293

  output_scale = (input_quant_value * weights_quant_value)/output_quant_value

  assert len(input_attribute["_arguments"]) == 1
  assert len(weights_attribute["_arguments"]) == 1
  assert len(bias_attribute["_arguments"]) == 1

  input_id = compute_order_renaming[input_attribute["_arguments"][0]["_name"]]
  input_variable_name = f"output_{input_id}"
  _, input_size = input_attribute["_arguments"][0]["_type"]["_shape"]["_dimensions"]

  _, output_size = output_attribute["_arguments"][0]["_type"]["_shape"]["_dimensions"]
  
  weight_dimensions = weights_attribute["_arguments"][0]["_initializer"]["_type"]["_shape"]["_dimensions"]
  assert len(weight_dimensions) == 2
  weights_data  = get_data_from_dict(weights_attribute["_arguments"][0]["_initializer"]["_data"], False)
  weights_data = np.array(weights_data).reshape((weight_dimensions[0], weight_dimensions[1]))
  
  bias_dimensions = bias_attribute["_arguments"][0]["_initializer"]["_type"]["_shape"]["_dimensions"]
  bias_data = get_data_from_dict(bias_attribute["_arguments"][0]["_initializer"]["_data"], True)
  assert output_size == len(bias_data)
  
  data_lines.append(f"elem_t output_{output_id}_weight[{weight_dimensions[0]}][{weight_dimensions[1]}] = {np2carray(weights_data)};")
  data_lines.append(f"acc_t output_{output_id}_bias[{len(bias_data)}] = {np2carray(bias_data)};")

  data_lines.append(f"elem_t output_{output_id}[{output_size}][1];")

  runtime_lines.append(textwrap.dedent(f"""\
    tiled_matmul_nn_auto(
      {output_size}, 1, {input_size},
      output_{output_id}_weight, {input_variable_name}, output_{output_id}_bias, output_{output_id},
      {"RELU" if followed_by_relu else "NO_ACTIVATION"}, {output_scale}, 0, false,
      WS, false, "fc_{output_id}"
    );"""
  ))

def generate_softmax(output_id, node, runtime_lines, data_lines, compute_order_renaming, quantization_tensor_mappings):
  input_attribute = get_input_by_name(node, "input")
  input_id = compute_order_renaming[input_attribute["_arguments"][0]["_name"]]
  input_variable_name = f"output_{input_id}"
  _, input_size = input_attribute["_arguments"][0]["_type"]["_shape"]["_dimensions"]

  assert len(node["_outputs"]) == 1
  output_attribute = node["_outputs"][0]
  assert len(output_attribute["_arguments"]) == 1
  _, output_size = output_attribute["_arguments"][0]["_type"]["_shape"]["_dimensions"]
  assert input_size == output_size

  data_lines.append(f"float output_{output_id}[{output_size}];")
  runtime_lines.append(textwrap.dedent(f"""\
    softmax({output_size}, {input_variable_name}, output_{output_id});"""
  ))

generate_operations = {
  "Conv2D": generate_conv2d,
  "DepthwiseConv2D": generate_dw_conv2d,
  "AveragePool2D": generate_avgPool2d,
  "Reshape": generate_reshape,
  "FullyConnected": generate_fc,
  "Softmax": generate_softmax
}

def generate_operation(output_id, node, runtime_lines, data_lines, compute_order_renaming, quantization_tensor_mappings):
  name = node["_type"]["name"]
  generate_operations[name](output_id, node, runtime_lines, data_lines, compute_order_renaming, quantization_tensor_mappings)
  if "_chain" in node:
    for chained_op in node["_chain"]:
      generate_operation(output_id, chained_op, runtime_lines, data_lines, compute_order_renaming, quantization_tensor_mappings)
    
def get_quantization_mappings(netron_file_new_format):
  name_tensor_mappings = {}
  for tensor in netron_file_new_format["subgraphs"][0]["tensors"]:
    name_tensor_mappings[tensor["name"]] = tensor
  return name_tensor_mappings
  
def generate_gemmini_code(netron_file, data_file, runtime_file, netron_file_new_format):
  assert len(netron_data["_inputs"]) == 1

  data_lines = ["#include \"include/gemmini_nn_mini.h\""]
  runtime_lines = ["#include \"data.h\""]
  
  input_node = netron_data["_inputs"][0]
  generate_input_loader("0", input_node, runtime_lines)
  quantization_tensor_mappings = get_quantization_mappings(netron_file_new_format)

  output_name_to_node = {}
  for node_defn in netron_file["_nodes"]:
    assert len(node_defn["_outputs"]) == 1
    assert len(node_defn["_outputs"][0]["_arguments"]) == 1
    output_name_to_node[node_defn["_outputs"][0]["_arguments"][0]["_name"]] = node_defn

  # construct a topological sort of the DAG
  remaining_nodes = set(output_name_to_node.keys())
  nodes_already_done = {input_node["_name"]}
  node_compute_order = []

  # terrible O(N^2) solution but whatever
  while len(remaining_nodes) > 0:
    found_next_node = False
    for potential in remaining_nodes:
      potential_node = output_name_to_node[potential]
      assert potential_node["_inputs"][0]["_name"] == "input" or (potential_node["_inputs"][0]["_name"] == "data" and potential_node["_type"]["name"] == "Reshape")
      dependencies = potential_node["_inputs"][0]["_arguments"]
      all_dependencies_satisfied = True
      for dependency in dependencies:
        if not (dependency["_name"] in nodes_already_done):
          all_dependencies_satisfied = False
          break
      if all_dependencies_satisfied:
        node_compute_order.append(potential)
        nodes_already_done.add(potential)
        remaining_nodes.remove(potential)
        found_next_node = True
        break
    assert found_next_node
  
  compute_order_renaming = {
    input_node["_name"]: "0"
  }
  next_id = 1
  for node_name in node_compute_order:
    compute_order_renaming[node_name] = str(next_id)
    next_id += 1

  runtime_lines.append("void run() {")
  for node_name in node_compute_order:
    output_id = compute_order_renaming[node_name]
    generate_operation(output_id, output_name_to_node[node_name], runtime_lines, data_lines, compute_order_renaming, quantization_tensor_mappings)
  runtime_lines.append("}")

  runtime_lines.append(f"#define final_output_matrix output_{compute_order_renaming[node_compute_order[-1]]}")
  
  data_file.write("\n".join(data_lines))
  runtime_file.write("\n".join(runtime_lines))

with open(netron_json, "r") as netron_file:
  netron_data = json.load(netron_file)
  with open(netron_json_new_format, "r") as netron_file_new:
    netron_data_new = json.load(netron_file_new)
    with open(sys.argv[3], "w") as data_file:
      with open(sys.argv[4], "w") as runtime_file:
        generate_gemmini_code(netron_data, data_file, runtime_file, netron_data_new)



