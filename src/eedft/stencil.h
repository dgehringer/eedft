#ifndef EEDFT_STENCIL_H
#define EEDFT_STENCIL_H

#include "wf_grid.h"
#include <stdint.h>

#define FD_MIN_ORDER 2
#define FD_MAX_ORDER 16
#define FD_COEFFS_MAX_P 8
#define FD_COEFFS_MAX_COEFF 17
#define NUM_ELEMENTS(x) (sizeof(x) / sizeof(x[0]))

struct fd_coeffs {
  SCALAR *coeffs;
  uint8_t p;
};

struct fd_stencil {
  uint8_t p;
  SCALAR coeffs[3][FD_COEFFS_MAX_COEFF];
};

static SCALAR fd_coeffs_2[2][3] = {{-1.0 / 2.0, 0.0, 1.0 / 2.0},
                                   {1.0 / 1.0, -2.0 / 1.0, 1.0 / 1.0}};

static SCALAR fd_coeffs_4[2][5] = {
    {1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0},
    {-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0}};

static SCALAR fd_coeffs_6[2][7] = {{-1.0 / 60.0, 3.0 / 20.0, -3.0 / 4.0, 0.0,
                                    3.0 / 4.0, -3.0 / 20, 1.0 / 60.0},
                                   {1.0 / 90.0, -3.0 / 20.0, 3.0 / 2.0,
                                    -49.0 / 18.0, 3.0 / 2.0, -3.0 / 20.0,
                                    1.0 / 90.0}};

static SCALAR fd_coeffs_8[2][9] = {
    {1.0 / 280.0, -4.0 / 105.0, 1.0 / 5.0, -4.0 / 5.0, 0.0, 4.0 / 5.0,
     -1.0 / 5.0, 4.0 / 105.0, -1.0 / 280.0},
    {-1.0 / 560.0, 8.0 / 315.0, -1.0 / 5.0, 8.0 / 5.0, -205.0 / 72.0, 8.0 / 5.0,
     -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0}};

static SCALAR fd_coeffs_10[2][11] = {
    {-1.0 / 1260.0, 5.0 / 504.0, -5.0 / 84.0, 5.0 / 21.0, -5.0 / 6.0, 0.0,
     5.0 / 6.0, -5.0 / 21.0, 5.0 / 84.0, -5.0 / 504.0, 1.0 / 1260.0},
    {1.0 / 3150.0, -5.0 / 1008.0, 5.0 / 126.0, -5.0 / 21.0, 5.0 / 3.0,
     -5269.0 / 1800.0, 5.0 / 3.0, -5.0 / 21.0, 5.0 / 126.0, -5.0 / 1008.0,
     1.0 / 3150.0}};

static SCALAR fd_coeffs_12[2][13] = {
    {1.0 / 5544.0, -1.0 / 385.0, 1.0 / 56.0, -5.0 / 63.0, 15.0 / 56.0,
     -6.0 / 7.0, 0.0, 6.0 / 7.0, -15.0 / 56.0, 5.0 / 63.0, -1.0 / 56.0,
     1.0 / 385.0, -1.0 / 5544.0},
    {-1.0 / 16632.0, 2.0 / 1925.0, -1.0 / 112.0, 10 / 189.0, -15.0 / 56.0,
     12.0 / 7.0, -5369.0 / 1800, 12.0 / 7.0, -15.0 / 56.0, 10 / 189.0,
     -1.0 / 112.0, 2.0 / 1925.0, -1.0 / 16632.0}};

static SCALAR fd_coeffs_14[2][15] = {
    {-1.0 / 24024.0, 7.0 / 10296.0, -7.0 / 1320, 7.0 / 264.0, -7.0 / 72.0,
     7.0 / 24.0, -7.0 / 8.0, 0.0, 7.0 / 8.0, -7.0 / 24.0, 7.0 / 72.0,
     -7.0 / 264.0, 7.0 / 1320, -7.0 / 10296.0, 1.0 / 24024.0},
    {1.0 / 84084.0, -7.0 / 30888.0, 7.0 / 3300, -7.0 / 528.0, 7.0 / 108.0,
     -7.0 / 24.0, 7.0 / 4.0, -266681.0 / 88200, 7.0 / 4.0, -7.0 / 24.0,
     7.0 / 108.0, -7.0 / 528.0, 7.0 / 3300, -7.0 / 30888.0, 1.0 / 84084.0}};

static SCALAR fd_coeffs_16[2][17] = {
    {1.0 / 102960.0, -8.0 / 45045.0, 2.0 / 1287.0, -56.0 / 6435.0, 7.0 / 198.0,
     -56.0 / 495.0, 14.0 / 45.0, -8.0 / 9.0, 0.0, 8.0 / 9.0, -14.0 / 45.0,
     56.0 / 495.0, -7.0 / 198.0, 56.0 / 6435.0, -2.0 / 1287.0, 8.0 / 45045.0,
     -1.0 / 102960.0},

    {-1.0 / 411840.0, 16.0 / 315315.0, -2.0 / 3861.0, 112.0 / 32175.0,
     -7.0 / 396.0, 112.0 / 1485.0, -14.0 / 45.0, 16.0 / 9.0,
     -1077749.0 / 352800.0, 16.0 / 9.0, -14.0 / 45.0, 112.0 / 1485.0,
     -7.0 / 396.0, 112.0 / 32175.0, -2.0 / 3861.0, 16.0 / 315315.0,
     -1.0 / 411840.0}};

static struct fd_coeffs _gradients[] = {
    {fd_coeffs_2[0], NUM_ELEMENTS(fd_coeffs_2[0]) / 2},
    {fd_coeffs_4[0], NUM_ELEMENTS(fd_coeffs_4[0]) / 2},
    {fd_coeffs_6[0], NUM_ELEMENTS(fd_coeffs_6[0]) / 2},
    {fd_coeffs_8[0], NUM_ELEMENTS(fd_coeffs_8[0]) / 2},
    {fd_coeffs_10[0], NUM_ELEMENTS(fd_coeffs_10[0]) / 2},
    {fd_coeffs_12[0], NUM_ELEMENTS(fd_coeffs_12[0]) / 2},
    {fd_coeffs_14[0], NUM_ELEMENTS(fd_coeffs_14[0]) / 2},
    {fd_coeffs_16[0], NUM_ELEMENTS(fd_coeffs_16[0]) / 2},
};

static struct fd_coeffs _laplacians[] = {
    {fd_coeffs_2[1], NUM_ELEMENTS(fd_coeffs_2[1]) / 2},
    {fd_coeffs_4[1], NUM_ELEMENTS(fd_coeffs_4[1]) / 2},
    {fd_coeffs_6[1], NUM_ELEMENTS(fd_coeffs_6[1]) / 2},
    {fd_coeffs_8[1], NUM_ELEMENTS(fd_coeffs_8[1]) / 2},
    {fd_coeffs_10[1], NUM_ELEMENTS(fd_coeffs_10[1]) / 2},
    {fd_coeffs_12[1], NUM_ELEMENTS(fd_coeffs_12[1]) / 2},
    {fd_coeffs_14[1], NUM_ELEMENTS(fd_coeffs_14[1]) / 2},
    {fd_coeffs_16[1], NUM_ELEMENTS(fd_coeffs_16[1]) / 2},
};

struct fd_stencil *make_gradient(uint8_t order, struct wf_grid *grid,
                                 struct fd_stencil *stencil);
struct fd_stencil *make_laplacian(uint8_t order, struct wf_grid *gri,
                                  struct fd_stencil *stencil);

#endif // EEDFT_STENCIL_H
