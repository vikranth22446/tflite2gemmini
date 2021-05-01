#include "include/gemmini.h"

#include <stdio.h>

void tiled_matmul_nn_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t A[dim_I][dim_K], const elem_t B[dim_K][dim_J],
        const void * D, elem_t C[dim_I][dim_J],
        int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
        enum tiled_matmul_type_t tiled_matmul_type,
        bool check, char * layer_name)
{
    tiled_matmul_auto(dim_I, dim_J, dim_K,
        (elem_t*)A, (elem_t*)B, D, (elem_t*)C, 
        dim_K, dim_J, dim_J, dim_J,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        act, scale, relu6_shift, repeating_bias,
        false, false,
        false, false,
        tiled_matmul_type);
}

void tiled_matmul_nn_auto_multiscale(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t A[dim_I][dim_K], const elem_t B[dim_K][dim_J],
        const void * D, elem_t C[dim_I][dim_J],
        int act, const acc_scale_t scales[dim_J], size_t relu6_shift, bool repeating_bias,
        enum tiled_matmul_type_t tiled_matmul_type,
        bool check, char * layer_name)
{
  for (int j = 0; j < dim_J; j++) {
    tiled_matmul_auto(dim_I, 1, dim_K,
        (elem_t*)A, (elem_t*)B + j, D + j, (elem_t*)C + j, 
        dim_K, dim_J, dim_J, dim_J,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        act, scales[j], relu6_shift, repeating_bias,
        false, false,
        false, false,
        tiled_matmul_type);
  }
}

void conv_auto_multiscale(
  int batch_size, int in_dim, int in_channels,
  int out_channels, int out_dim,
  int stride, int padding, int kernel_dim,

  const elem_t input[in_dim * in_dim][in_channels],
  const elem_t weights_transposed[out_channels][kernel_dim * kernel_dim * in_channels],
  const acc_t bias[out_channels],
  elem_t output[out_dim * out_dim][out_channels],

  int act, const acc_scale_t scales[out_channels], size_t relu6_shift,
  int pool_size, int pool_stride, int pool_padding,

  enum tiled_matmul_type_t tiled_conv_type
) {
  elem_t temp_output[out_dim * out_dim];
  for (int out_channel = 0; out_channel < out_channels; out_channel++) {
    tiled_conv_A_stride_auto(
      1, in_dim, in_channels,
      1, out_dim,
      stride, 1, padding, kernel_dim,

      (elem_t*) input,
      weights_transposed[out_channel],
      &bias[out_channel],
      temp_output,

      act, scales[out_channel], relu6_shift,
      pool_size, pool_stride, pool_padding,
      tiled_conv_type
    );

    // transpose back
    // TODO(shadaj): we can avoid this and tell the next conv that the input is transposed
    for (int out_row = 0; out_row < out_dim; out_row++) {
      for (int out_col = 0; out_col < out_dim; out_col++) {
        output[out_row * out_dim + out_col][out_channel] = temp_output[out_row * out_dim + out_col];
      }
    }
  }
}

void conv_auto_dw_multiscale(
  int batch_size, int in_dim, int in_channels,
  int out_channels, int out_dim,
  int stride, int padding, int kernel_dim,

  const elem_t input[in_dim * in_dim][in_channels],
  const elem_t weights_transposed[out_channels][kernel_dim * kernel_dim],
  const acc_t bias[out_channels],
  elem_t output[out_dim * out_dim][out_channels],

  int act, const acc_scale_t scales[out_channels], size_t relu6_shift,
  int pool_size, int pool_stride, int pool_padding,

  enum tiled_matmul_type_t tiled_conv_type
) {
  elem_t temp_output[out_dim * out_dim];
  elem_t input_transposed[in_channels][in_dim*in_dim];
  for(int in_channel = 0; in_channel < in_channels; in_channel++) {
    for(int row = 0; row<in_dim; row++) {
      for(int col = 0; col < in_dim; col++) {
        input_transposed[in_channel][row*in_dim + col] = input[row*in_dim + col][in_channel];
      }
    }
  }

  for (int out_channel = 0; out_channel < out_channels; out_channel++) {
    tiled_conv_A_stride_auto(
      1, in_dim, 1,
      1, out_dim,
      stride, 1, padding, kernel_dim,

      (elem_t*) &input_transposed[out_channel],
      weights_transposed[out_channel],
      &bias[out_channel],
      temp_output,

      act, scales[out_channel], relu6_shift,
      pool_size, pool_stride, pool_padding,
      tiled_conv_type
    );

    // transpose back
    // TODO(shadaj): we can avoid this and tell the next conv that the input is transposed
    for (int out_row = 0; out_row < out_dim; out_row++) {
      for (int out_col = 0; out_col < out_dim; out_col++) {
        output[out_row * out_dim + out_col][out_channel] = temp_output[out_row * out_dim + out_col];
      }
    }
  }
}

static void softmax(size_t input_len, elem_t input[input_len][1], float output[input_len]) {
  // elem_t max = -128; // used instead of -inf bc of elem_t
//   for (size_t i = 0; i < input_len; i++) {
//     if (input[i] > max) {
//       max = input[i];
//     }
//   }

//   float sum = 0.0;
//   for (size_t i = 0; i < input_len; i++) {
//     sum += expf(input[i] - max);
//   }

  // float offset = m + logf(sum);
  // for (size_t i = 0; i < input_len; i++) {
  //   // output[i] = expf(input[i] - offset);
  //   output[i] = (float) input[i][0];
  // }
  if (input[0][0] > input[1][0]) {
    output[0] = 1.0;
    output[1] = 0.0;
  } else {
    output[0] = 0.0;
    output[1] = 1.0;
  }
}

void compute_average_pooling(size_t in_dim, size_t channels,
                             elem_t input[in_dim * in_dim][channels],
                             elem_t average[1 * 1][channels]){
  for (int channel = 0; channel < channels; channel++) {
    int sum = 0;
    for (int row = 0; row < in_dim; row++) {
      for (int col = 0; col < in_dim; col++) {
        size_t r = row * in_dim + col;
        sum += input[r][channel];
      }
    }
    const int count = in_dim * in_dim;
    // some rounding up thing
    average[0][channel] = (sum + count/2) / count;
  }
}

void display_im(int row_count, int col_count, int channel_count, elem_t im[row_count * col_count][channel_count]) {
  printf("np.array([\n");
  for (int row = 0; row < row_count; row++) {
    printf("[");
    for (int col = 0; col < col_count; col++) {
      printf("[");
      for (int out_channel = 0; out_channel < channel_count; out_channel++) {
        printf("%d, ", im[row * col_count + col][out_channel]);
      }
      printf("], ");
    }
    printf("],\n");
  }
  printf("])\n");
}
