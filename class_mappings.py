import textwrap
from abc import ABC, abstractmethod
import numpy as np
class Layer(ABC):
    @abstractmethod
    def generate_data_gemmini(self):
        pass
    
    @abstractmethod
    def generate_runtime_gemmini(self):
        pass

    def np2carray(self, arr):
        if np.isscalar(arr):
            return str(arr)
        else:
            return "{ " + ", ".join(map(self.np2carray, arr)) + " }"

class Input(Layer):
    def __init__(self) -> None:
        self.output_id = None
        self.input_width = None
        self.input_height = None
        self.input_depth = None
        # TODO simplify python data generation for input layer
    
    def generate_data_gemmini(self):
        return textwrap.dedent(f"""\
            #define output_{self.output_id}_dim {self.input_height}*{self.input_width}*{self.input_depth}
            elem_t output_{self.output_id}[{self.input_height * self.input_width}][{self.input_depth}];
        """)
    
class Conv2d(Layer):
    def __init__(self) -> None:
        self.output_id = None
        self.filter_width = None
        self.filter_height = None
        self.filter_in_channels = None
        self.filter_count = None
        self.filter_data = None
        self.bias_data = None

        self.input_width = None
        self.input_depth = None
        
        self.stride_w = None
        self.padding_w = None
        self.followed_by_relu = None

        self.quantization = None
        self.output_scale = None
        self.output_multiplies = None
        self.input_variable_name = None

        self.output_width = None
        self.output_height = None
        self.output_depth = None
    
    def generate_data_gemmini(self):
        filter_data = np.array(self.filter_data).reshape((self.filter_in_channels, self.filter_height, self.filter_width, self.filter_count))
        weights_data_count_first = filter_data.transpose((3, 0, 1, 2)).reshape((self.filter_count, self.filter_height * self.filter_width))
        filter_data_orig_input_flattened = filter_data.reshape((self.filter_count, self.filter_height * self.filter_width * self.filter_in_channels))

        return textwrap.dedent(f"""\
            elem_t output_{self.output_id}_weight[{self.filter_count}][{self.filter_height * self.filter_width}] = {self.np2carray(weights_data_count_first)};
            acc_t output_{self.output_id}_bias[{len(self.bias_data)}] = {self.np2carray(self.bias_data)};
            const float output_{self.output_id}_multipliers[{self.output_depth}] = {self.np2carray(self.output_multiplies)};
            elem_t output_{self.output_id}[{self.output_height} * {self.output_width}][{self.output_depth}];
            const elem_t output_{self.output_id}_weight_filterFirst[{self.filter_count}][{self.filter_width * self.filter_height * self.filter_in_channels}] = {self.np2carray(filter_data_orig_input_flattened)};
        """)
    
    def generate_data_python(self):
        filter_data = np.array(self.filter_data).reshape((self.filter_in_channels, self.filter_height, self.filter_width, self.filter_count))
        weights_data_count_first = filter_data.transpose((3, 0, 1, 2)).reshape((self.filter_count, self.filter_height * self.filter_width))
        filter_data_orig_input_flattened = filter_data.reshape((self.filter_count, self.filter_height * self.filter_width * self.filter_in_channels))

        return textwrap.dedent(f"""\
            output_{self.output_id}_weight = {weights_data_count_first}
            output_{self.output_id}_bias = {self.bias_data}
            output_{self.output_id}_multipliers = {self.output_multiplies}
            output_{self.output_id}_weight_filterFirst = {filter_data_orig_input_flattened}
            """)

    def generate_runtime_gemmini(self):
        return textwrap.dedent(f"""\
            conv_auto_multiscale(
            1, {self.input_width}, {self.input_depth},
            {self.output_depth}, {self.output_width},
            {self.stride_w}, {self.padding_w}, {self.filter_width},
            
            {self.input_variable_name}, output_{self.output_id}_weight_filterFirst, output_{self.output_id}_bias, output_{self.output_id},

            {"RELU" if self.followed_by_relu else "NO_ACTIVATION"}, output_{self.output_id}_multipliers, 0,
            1, 1, 0,
            WS
        );
        """)

    def generate_runtime_python(self):
        return textwrap.dedent(f"""\
            output_{self.output_id} = conv2d({self.input_variable_name}, output_{self.output_id}_weight_filterFirst, output_{self.output_id}_bias, output_{self.output_id}_multipliers, stride={self.stride_w}, padding={self.padding_w})
        """)
    
    def generate_runtime_logging(self):
        return textwrap.dedent(f"""\
            printf("output_{self.output_id} = ");
            display_im({self.output_height}, {self.output_width}, {self.output_depth}, output_{self.output_id});"""
        )
class Conv2dDw(Conv2d):

    def generate_data_python(self):
        filter_data = np.array(self.filter_data).reshape((self.filter_in_channels, self.filter_height, self.filter_width, self.filter_count))
        weights_data_count_first = filter_data.transpose((3, 0, 1, 2)).reshape((self.filter_count, self.filter_height * self.filter_width))

        return textwrap.dedent(f"""\
            output_{self.output_id}_weight = {self.np2carray(weights_data_count_first)}
            output_{self.output_id}_bias = {self.np2carray(self.bias_data)}
            output_{self.output_id}_multipliers = {self.np2carray(self.output_multiplies)}
            """)

    def generate_data_gemmini(self):
        filter_data = np.array(self.filter_data).reshape((self.filter_in_channels, self.filter_height, self.filter_width, self.filter_count))
        weights_data_count_first = filter_data.transpose((3, 0, 1, 2)).reshape((self.filter_count, self.filter_height * self.filter_width))

        return textwrap.dedent(f"""\
            elem_t output_{self.output_id}_weight[{self.filter_count}][{self.filter_height * self.filter_width}] = {self.np2carray(weights_data_count_first)};
            acc_t output_{self.output_id}_bias[{len(self.bias_data)}] = {self.np2carray(self.bias_data)};
            const float output_{self.output_id}_multipliers[{self.output_depth}] = {self.np2carray(self.output_multiplies)};
            elem_t output_{self.output_id}[{self.output_height} * {self.output_width}][{self.output_depth}];
        """)

    def generate_runtime_gemmini(self):
        return textwrap.dedent(f"""\
            conv_auto_dw_multiscale(
            1, {self.input_width}, {self.input_depth},
            {self.output_depth}, {self.output_width},
            {self.stride_w}, {self.padding_w}, {self.filter_width},

            {self.input_variable_name},  output_{self.output_id}_weight, output_{self.output_id}_bias, output_{self.output_id},

            {"RELU" if self.followed_by_relu else "NO_ACTIVATION"}, output_{self.output_id}_multipliers, 0,
            1, 0, 0,
            WS
            );"""
        )

    def generate_runtime_python(self):
        return textwrap.dedent(f"""\
            output_{self.output_id} = conv2d_dw({self.input_variable_name}, output_{self.output_id}_weight, output_{self.output_id}_bias, output_{self.output_id}_multipliers)
        """)
class AveragePool2d(Layer):
    def __init__(self) -> None:
        self.output_id = None

        self.output_width = None
        self.output_height = None
        self.output_depth = None

        self.input_depth = None
        self.input_variable_name = None
    
    def generate_data_gemmini(self):
        return textwrap.dedent(f"""
            elem_t output_{self.output_id}[{self.output_height} * {self.output_width}][{self.output_depth}];
        """)

    def generate_runtime_gemmini(self):
        return textwrap.dedent(f"""\
            compute_average_pooling(
            {self.input_width}, {self.input_depth},
            {self.input_variable_name},
            output_{self.output_id}
            );"""
        )
    
    def generate_runtime_logging(self):
        return textwrap.dedent(f"""\
            printf("output_{self.output_id} = ");
            display_im({self.output_height}, {self.output_width}, {self.output_depth}, output_{self.output_id});"""
        )
    def generate_runtime_python(self):
        return textwrap.dedent(f"""\
            output_{self.output_id} = avg_pool2d({self.input_variable_name})
        """)
    
class Reshape(Layer):
    def __init__(self) -> None:
        self.output_id = None
        self.output_size = None

        self.input_variable_name = None
    
    def generate_data_gemmini(self):
        return ""
    
    def generate_runtime_gemmini(self):
        return textwrap.dedent(f"""\
            elem_t (*output_{self.output_id})[{self.output_size}] = {self.input_variable_name};"""
        )
   
    def generate_runtime_python(self):
        return textwrap.dedent(f"""\
            output_{self.output_id} = {self.input_variable_name}.flatten()
        """)
    
class Fc(Layer):
    def __init__(self) -> None:
        self.output_id = None
        
        self.output_size = None
        self.input_size = None

        self.weight_data = None
        self.weight_width = None
        self.weight_height = None

        self.bias_data = None

        self.quantization = None
        self.output_scale = None
        self.followed_by_relu = None

        self.input_variable_name = None
    
    def generate_data_python(self):
        return textwrap.dedent(f"""\
            output_{self.output_id}_weight = {self.weight_data};
            output_{self.output_id}_bias = {self.bias_data};
        """)

    def generate_data_gemmini(self):
        return ""
    
    def generate_data_gemmini(self):
        return textwrap.dedent(f"""\
            elem_t output_{self.output_id}_weight[{self.weight_width}][{self.weight_height}] = {self.np2carray(self.weight_data)};
            acc_t output_{self.output_id}_bias[{len(self.bias_data)}] = {self.np2carray(self.bias_data)};
            elem_t output_{self.output_id}[{self.output_size}][1];
        """)
     
    def generate_runtime_gemmini(self):
        return textwrap.dedent(f"""\
            tiled_matmul_nn_auto(
                {self.output_size}, 1, {self.input_size},
                output_{self.output_id}_weight, {self.input_variable_name}, output_{self.output_id}_bias, output_{self.output_id},
                {"RELU" if self.followed_by_relu else "NO_ACTIVATION"}, {self.output_scale}, 0, false,
                WS, false, "fc_{self.output_id}"
            );"""
        )
    def generate_runtime_python(self):
        # TODO handle zero point
        return textwrap.dedent(f"""\
            output_{self.output_id} = fc({self.input_variable_name}, output_{self.output_id}_weight, output_{self.output_id}_bias, out_zero_point=-5)
        """)
        
class Softmax(Layer):
    def __init__(self) -> None:
        self.output_id = None
        self.output_size = None
        self.input_variable_name = None
        self.out_scale = None

    def generate_data_gemmini(self):
        return ""
    
    def generate_runtime_gemmini(self):
        return textwrap.dedent(f"""\
                softmax({self.output_size}, {self.input_variable_name}, output_{self.output_id});
        """)
    
    def generate_runtime_python(self):
        # TODO handle in and out zero point
        return textwrap.dedent(f"""\
            output_{self.output_id} = softmax_layer(sim_output_30, in_zero_point=-5, out_zero_point=-128, out_scale={self.out_scale})
        """)
        