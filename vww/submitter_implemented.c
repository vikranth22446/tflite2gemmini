#include "submitter_implemented_for_c.h"

// #include <cstdarg>
// #include <cstdio>
// #include <cstdlib>
// #include <cstring>
// #include <ctime>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "api/internally_implemented_for_c.h"
// #include "include/gemmini.h"
#include "runtime.h"

#define kNumCols 96
#define kNumRows 96
#define kNumChannels 3

// #define kVwwInputSize kNumCols * kNumRows * kNumChannels

#define kCategoryCount 2
// #define kPersonIndex 1
// #define kNotAPersonIndex 0


// elem_t random_matrix[kVwwInputSize][kCategoryCount];
// elem_t random_bias[kCategoryCount];
// elem_t output_matrix[kCategoryCount];

// elem_t input[kVwwInputSize];

// #define DEBUG_GEMMINI_TESTS

// Implement this method to prepare for inference and preprocess inputs.

void th_load_tensor() {
  size_t bytes = ee_get_buffer((uint8_t*) output_0,
                               output_0_dim * sizeof(int8_t));
  if (bytes / sizeof(int8_t) != output_0_dim) {
    th_printf("Input db has %d elements, expected %d\n",
              bytes / sizeof(int8_t), output_0_dim);
    return;
  }

  for (int row = 0; row < kNumRows; row++) {
    for (int col = 0; col < kNumCols; col++) {
      for (int channel = 0; channel < kNumChannels; channel++) {
        output_0[row * kNumCols + col][channel] -= 128;
      }
    }
  }

  printf("import numpy as np\n");
  printf("input_image = ");
  display_im(kNumRows, kNumCols, kNumChannels, output_0);
}

// Add to this method to return real inference results.
void th_results() {
  const int nresults = 3;
  /**
   * The results need to be printed back in exactly this format; if easier
   * to just modify this loop than copy to results[] above, do that.
   */
  th_printf("m-results-[");
  for (size_t i = 0; i < kCategoryCount; i++) {
    float converted = final_output_matrix[i];

    // Some platforms don't implement floating point formatting.
    th_printf("%0.3f", converted);
    if (i < (nresults - 1)) {
      th_printf(",");
    }
  }
  th_printf("]\r\n");
}

// Implement this method with the logic to perform one inference cycle.
void th_infer() {
  run();
}

const elem_t test_A[1][3] = {
  {1, 2, 3}
};

const elem_t test_B[3][3] = {
  {1, 2, 3},
  {4, 5, 6},
  {7, 8, 9}
};

const acc_t test_D[1][3] = {
  {0, 0, 0}
};

const float test_scales[3] = { 1.0, 2.0, 3.0 };

elem_t test_C[1][3];

#define conv_test_in_dim 4
// nhwc
const elem_t test_conv_image[conv_test_in_dim * conv_test_in_dim][1] = {
  {1}, {2}, {3}, {4},
  {5}, {6}, {7}, {8},
  {9}, {10}, {11}, {12},
  {13}, {14}, {15}, {16}
};

#define conv_test_kernel_dim 2
#define conv_test_kernel_count 2

const elem_t test_conv_image_for_dw[conv_test_in_dim * conv_test_in_dim][conv_test_kernel_count] = {
  {1, 13}, {2, 14}, {3, 15}, {4, 16},
  {5, 9}, {6, 10}, {7, 13}, {8, 12},
  {9, 5}, {10, 6}, {11, 7}, {12, 8},
  {13, 1}, {14, 2}, {15, 3}, {16, 4}
};

// in_channels, in_h, in_w, filter_count
const elem_t row_align(1) test_conv_filter[1 * conv_test_kernel_dim * conv_test_kernel_dim][conv_test_kernel_count] = {
  {1, 2}, {1, 2},
  {1, 2}, {1, 2}
};

const elem_t row_align(1) test_conv_filter_transposed[conv_test_kernel_count][1 * conv_test_kernel_dim * conv_test_kernel_dim] = {
  {
    1, 1,
    1, 1
  },
  {
    2, 2,
    2, 2
  }
};

#define conv_test_stride 1
#define conv_test_padding 0
#define conv_test_dilation 1 // default on gemmini

const acc_t test_conv_bias[conv_test_kernel_count] = { 0, 0 };

const float test_conv_scales[conv_test_kernel_count] = { 0.5, 2.0 };

#define conv_test_out_dim 3
elem_t test_conv_out[conv_test_out_dim * conv_test_out_dim][conv_test_kernel_count];

/// \brief optional API.
void th_final_initialize(void) {
  gemmini_flush(0);
  #ifdef DEBUG_GEMMINI_TESTS
    // test 1x1 conv axis scaling
    tiled_matmul_nn_auto(1, 3, 3,
          test_A, test_B,
          test_D, test_C,
          NO_ACTIVATION, 1.0, 0, true,
          WS,
          false, "test_matmul");

    for (int i = 0; i < 3; i++) {
      printf("output %d is %d\n", i, test_C[0][i]);
    }

    printf("now with multiscale\n");

    tiled_matmul_nn_auto_multiscale(1, 3, 3,
          test_A, test_B,
          test_D, test_C,
          NO_ACTIVATION, test_scales, 0, true,
          WS,
          false, "test_matmul");

    for (int i = 0; i < 3; i++) {
      printf("output %d is %d\n", i, test_C[0][i]);
    }

    if (conv_test_out_dim != (conv_test_in_dim + 2*conv_test_padding - conv_test_dilation * (conv_test_kernel_dim - 1) - 1) / conv_test_stride + 1) {
        printf("conv out_dim is not correct\n");
    }

    // test 2x2 conv axis-scaling
    tiled_conv_A_stride_auto(
      1, conv_test_in_dim, 1,
      conv_test_kernel_count, conv_test_out_dim,
      conv_test_stride, conv_test_dilation, conv_test_padding, conv_test_kernel_dim,

      test_conv_image,
      test_conv_filter,
      test_conv_bias,
      test_conv_out,

      NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 0, 0, 0,
      WS
    );

    printf("orig out:\n");
    for (int row = 0; row < conv_test_out_dim; row++) {
      for (int col = 0; col < conv_test_out_dim; col++) {
        printf("{ ");
        for (int out_channel = 0; out_channel < conv_test_kernel_count; out_channel++) {
          printf("%d ", test_conv_out[row * conv_test_out_dim + col][out_channel]);
          test_conv_out[row * conv_test_out_dim + col][out_channel] = 0;
        }
        printf("} ");
      }
      printf("\n");
    }

    conv_auto_multiscale(
      1, conv_test_in_dim, 1,
      conv_test_kernel_count, conv_test_out_dim,
      conv_test_stride, conv_test_padding, conv_test_kernel_dim,

      test_conv_image,
      test_conv_filter_transposed,
      test_conv_bias,
      test_conv_out,

      NO_ACTIVATION, test_conv_scales, 0, 0, 0, 0,
      WS
    );

    printf("conv with multiscale:\n");
    for (int row = 0; row < conv_test_out_dim; row++) {
      for (int col = 0; col < conv_test_out_dim; col++) {
        printf("{ ");
        for (int out_channel = 0; out_channel < conv_test_kernel_count; out_channel++) {
          printf("%d ", test_conv_out[row * conv_test_out_dim + col][out_channel]);
          test_conv_out[row * conv_test_out_dim + col][out_channel] = 0;
        }
        printf("} ");
      }
      printf("\n");
    }

    /*
    should be:
    { 7 56 } { 9 72 } { 11 88 } 
    { 15 120 } { 17 127 } { 19 127 } 
    { 23 127 } { 25 127 } { 27 127 } 
    */

    printf("now dw multiscale\n");
    conv_auto_dw_multiscale(
      1, conv_test_in_dim, conv_test_kernel_count,
      conv_test_kernel_count, conv_test_out_dim,
      conv_test_stride, conv_test_padding, conv_test_kernel_dim,

      test_conv_image_for_dw,
      test_conv_filter_transposed,
      test_conv_bias,
      test_conv_out,

      NO_ACTIVATION, test_conv_scales, 0, 0, 0, 0,
      WS
    );

    printf("out:\n");
    for (int row = 0; row < conv_test_out_dim; row++) {
      for (int col = 0; col < conv_test_out_dim; col++) {
        printf("{ ");
        for (int out_channel = 0; out_channel < conv_test_kernel_count; out_channel++) {
          printf("%d ", test_conv_out[row * conv_test_out_dim + col][out_channel]);
          test_conv_out[row * conv_test_out_dim + col][out_channel] = 0;
        }
        printf("} ");
      }
      printf("\n");
    }

    /*
    should be:
    { 7 127 } { 9 127 } { 11 127 } 
    { 15 120 } { 17 127 } { 19 127 } 
    { 23 56 } { 25 72 } { 27 88 } 
    */
  #endif
}

void th_pre() {}
void th_post() {}

void th_command_ready(char volatile *p_command) {
  p_command = p_command;
  ee_serial_command_parser_callback((char *)p_command);
}

// th_libc implementations.
int th_strncmp(const char *str1, const char *str2, size_t n) {
  return strncmp(str1, str2, n);
}

char *th_strncpy(char *dest, const char *src, size_t n) {
  return strncpy(dest, src, n);
}

size_t th_strnlen(const char *str, size_t maxlen) {
  // return strnlen(str, maxlen);
  return strlen(str);
}

char *th_strcat(char *dest, const char *src) { return strcat(dest, src); }

char *th_strtok(char *str1, const char *sep) { return strtok(str1, sep); }

int th_atoi(const char *str) { return atoi(str); }

void *th_memset(void *b, int c, size_t len) { return memset(b, c, len); }

void *th_memcpy(void *dst, const void *src, size_t n) {
  return memcpy(dst, src, n);
}

char th_getchar() { 
  char oneByte;
  int r = read(0, &oneByte, 1);
  return oneByte;
}

void th_serialport_initialize(void) { }

void th_timestamp(void) {
  unsigned long microSeconds = 0ul;
  /* USER CODE 2 BEGIN */
  microSeconds = (unsigned long)((uint64_t)clock() * 1000000 / CLOCKS_PER_SEC);
  /* USER CODE 2 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP, microSeconds);
}

void th_timestamp_initialize(void) {
  /* USER CODE 1 BEGIN */
  // Setting up BOTH perf and energy here
  /* USER CODE 1 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP_MODE);
  /* Always call the timestamp on initialize so that the open-drain output
     is set to "1" (so that we catch a falling edge) */
  th_timestamp();
}
