#include <stddef.h>
#include <stdint.h>
#include "merged_layers.h"

#define MAX_FEATURE_SIZE 153600
#define MAX_IM2COL_SIZE 13824
#define MAX_BUFFERB_SIZE 13824

static uint8_t  buffer0[MAX_FEATURE_SIZE];
static uint8_t  buffer1[MAX_FEATURE_SIZE];
static int16_t  bufferA[MAX_IM2COL_SIZE];
static uint8_t  bufferB[MAX_BUFFERB_SIZE];

void invoke_layers(void) {
  // Layer 0 convolution
  arm_convolve_HWC_int4_u4_int4(
    buffer0, 160, 3, WEIGHT_0, 24, 3, 1, 1, 1, 1, 2, BIAS_0, buffer1, 80, 0, Z_W_0, 0, M_ZERO_0, N_ZERO_0, bufferA, bufferB);

  // Layer 1 depthwise convolution
  arm_depthwise_separable_conv_HWC_u4_u4_u4(
    buffer1, 80, 24, WEIGHT_1, 24, 3, 1, 1, 1, 1, 1, BIAS_1, buffer0, 80, 0, Z_W_1, 0, M_ZERO_1, N_ZERO_1, bufferA, bufferB);

  // Layer 2 convolution
  arm_convolve_HWC_int4_u4_int4(
    buffer0, 80, 24, WEIGHT_2, 48, 1, 0, 0, 0, 0, 1, BIAS_2, buffer1, 80, 0, Z_W_2, 0, M_ZERO_2, N_ZERO_2, bufferA, bufferB);

  // Layer 3 depthwise convolution
  arm_depthwise_separable_conv_HWC_u4_u4_u4(
    buffer1, 80, 48, WEIGHT_3, 48, 3, 1, 1, 1, 1, 2, BIAS_3, buffer0, 40, 0, Z_W_3, 0, M_ZERO_3, N_ZERO_3, bufferA, bufferB);

  // Layer 4 convolution
  arm_convolve_HWC_int4_u4_int4(
    buffer0, 40, 48, WEIGHT_4, 96, 1, 0, 0, 0, 0, 1, BIAS_4, buffer1, 40, 0, Z_W_4, 0, M_ZERO_4, N_ZERO_4, bufferA, bufferB);

  // Layer 5 depthwise convolution
  arm_depthwise_separable_conv_HWC_u4_u4_u4(
    buffer1, 40, 96, WEIGHT_5, 96, 3, 1, 1, 1, 1, 1, BIAS_5, buffer0, 40, 0, Z_W_5, 0, M_ZERO_5, N_ZERO_5, bufferA, bufferB);

  // Layer 6 convolution
  arm_convolve_HWC_int4_u4_int4(
    buffer0, 40, 96, WEIGHT_6, 96, 1, 0, 0, 0, 0, 1, BIAS_6, buffer1, 40, 0, Z_W_6, 0, M_ZERO_6, N_ZERO_6, bufferA, bufferB);

  // Layer 7 depthwise convolution
  arm_depthwise_separable_conv_HWC_u4_u4_u4(
    buffer1, 40, 96, WEIGHT_7, 96, 3, 1, 1, 1, 1, 2, BIAS_7, buffer0, 20, 0, Z_W_7, 0, M_ZERO_7, N_ZERO_7, bufferA, bufferB);

  // Layer 8 convolution
  arm_convolve_HWC_int4_u4_int4(
    buffer0, 20, 96, WEIGHT_8, 192, 1, 0, 0, 0, 0, 1, BIAS_8, buffer1, 20, 0, Z_W_8, 0, M_ZERO_8, N_ZERO_8, bufferA, bufferB);

  // Layer 9 depthwise convolution
  arm_depthwise_separable_conv_HWC_u4_u4_u4(
    buffer1, 20, 192, WEIGHT_9, 192, 3, 1, 1, 1, 1, 1, BIAS_9, buffer0, 20, 0, Z_W_9, 0, M_ZERO_9, N_ZERO_9, bufferA, bufferB);

  // Layer 10 convolution
  arm_convolve_HWC_int4_u4_int4(
    buffer0, 20, 192, WEIGHT_10, 192, 1, 0, 0, 0, 0, 1, BIAS_10, buffer1, 20, 0, Z_W_10, 0, M_ZERO_10, N_ZERO_10, bufferA, bufferB);

  // Layer 11 depthwise convolution
  arm_depthwise_separable_conv_HWC_u4_u4_u4(
    buffer1, 20, 192, WEIGHT_11, 192, 3, 1, 1, 1, 1, 2, BIAS_11, buffer0, 10, 0, Z_W_11, 0, M_ZERO_11, N_ZERO_11, bufferA, bufferB);

  // Layer 12 convolution
  arm_convolve_HWC_int4_u4_int4(
    buffer0, 10, 192, WEIGHT_12, 384, 1, 0, 0, 0, 0, 1, BIAS_12, buffer1, 10, 0, Z_W_12, 0, M_ZERO_12, N_ZERO_12, bufferA, bufferB);

  // Layer 13 depthwise convolution
  arm_depthwise_separable_conv_HWC_u4_u4_u4(
    buffer1, 10, 384, WEIGHT_13, 384, 3, 1, 1, 1, 1, 1, BIAS_13, buffer0, 10, 0, Z_W_13, 0, M_ZERO_13, N_ZERO_13, bufferA, bufferB);

  // Layer 14 convolution
  arm_convolve_HWC_int4_u4_int4(
    buffer0, 10, 384, WEIGHT_14, 384, 1, 0, 0, 0, 0, 1, BIAS_14, buffer1, 10, 0, Z_W_14, 0, M_ZERO_14, N_ZERO_14, bufferA, bufferB);

  // Layer 15 depthwise convolution
  arm_depthwise_separable_conv_HWC_u4_u4_u4(
    buffer1, 10, 384, WEIGHT_15, 384, 3, 1, 1, 1, 1, 1, BIAS_15, buffer0, 10, 0, Z_W_15, 0, M_ZERO_15, N_ZERO_15, bufferA, bufferB);

  // Layer 16 convolution
  arm_convolve_HWC_int4_u4_int4(
    buffer0, 10, 384, WEIGHT_16, 384, 1, 0, 0, 0, 0, 1, BIAS_16, buffer1, 10, 0, Z_W_16, 0, M_ZERO_16, N_ZERO_16, bufferA, bufferB);

  // Layer 17 depthwise convolution
  arm_depthwise_separable_conv_HWC_u4_u4_u4(
    buffer1, 10, 384, WEIGHT_17, 384, 3, 1, 1, 1, 1, 1, BIAS_17, buffer0, 10, 0, Z_W_17, 0, M_ZERO_17, N_ZERO_17, bufferA, bufferB);

  // Layer 18 convolution
  arm_convolve_HWC_int4_u4_int4(
    buffer0, 10, 384, WEIGHT_18, 384, 1, 0, 0, 0, 0, 1, BIAS_18, buffer1, 10, 0, Z_W_18, 0, M_ZERO_18, N_ZERO_18, bufferA, bufferB);

  // Layer 19 depthwise convolution
  arm_depthwise_separable_conv_HWC_u4_u4_u4(
    buffer1, 10, 384, WEIGHT_19, 384, 3, 1, 1, 1, 1, 1, BIAS_19, buffer0, 10, 0, Z_W_19, 0, M_ZERO_19, N_ZERO_19, bufferA, bufferB);

  // Layer 20 convolution
  arm_convolve_HWC_int4_u4_int4(
    buffer0, 10, 384, WEIGHT_20, 384, 1, 0, 0, 0, 0, 1, BIAS_20, buffer1, 10, 0, Z_W_20, 0, M_ZERO_20, N_ZERO_20, bufferA, bufferB);

  // Layer 21 depthwise convolution
  arm_depthwise_separable_conv_HWC_u4_u4_u4(
    buffer1, 10, 384, WEIGHT_21, 384, 3, 1, 1, 1, 1, 1, BIAS_21, buffer0, 10, 0, Z_W_21, 0, M_ZERO_21, N_ZERO_21, bufferA, bufferB);

  // Layer 22 convolution
  arm_convolve_HWC_int4_u4_int4(
    buffer0, 10, 384, WEIGHT_22, 384, 1, 0, 0, 0, 0, 1, BIAS_22, buffer1, 10, 0, Z_W_22, 0, M_ZERO_22, N_ZERO_22, bufferA, bufferB);

  // Layer 23 depthwise convolution
  arm_depthwise_separable_conv_HWC_u4_u4_u4(
    buffer1, 10, 384, WEIGHT_23, 384, 3, 1, 1, 1, 1, 2, BIAS_23, buffer0, 5, 0, Z_W_23, 0, M_ZERO_23, N_ZERO_23, bufferA, bufferB);

  // Layer 24 convolution
  arm_convolve_HWC_int4_u4_int4(
    buffer0, 5, 384, WEIGHT_24, 768, 1, 0, 0, 0, 0, 1, BIAS_24, buffer1, 5, 0, Z_W_24, 0, M_ZERO_24, N_ZERO_24, bufferA, bufferB);

  // Layer 25 depthwise convolution
  arm_depthwise_separable_conv_HWC_u4_u4_u4(
    buffer1, 5, 768, WEIGHT_25, 768, 3, 1, 1, 1, 1, 1, BIAS_25, buffer0, 5, 0, Z_W_25, 0, M_ZERO_25, N_ZERO_25, bufferA, bufferB);

  // Layer 26 convolution
  arm_convolve_HWC_int4_u4_int4(
    buffer0, 5, 768, WEIGHT_26, 768, 1, 0, 0, 0, 0, 1, BIAS_26, buffer1, 5, 0, Z_W_26, 0, M_ZERO_26, N_ZERO_26, bufferA, bufferB);

}
