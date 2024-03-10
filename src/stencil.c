
#include "eedft/stencil.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

struct fd_stencil *make_stencil(struct fd_coeffs all_coeffs[], SCALAR m,
                                uint8_t order, struct wf_grid *grid,
                                struct fd_stencil *stencil) {
  if (order < 2 || order > FD_MAX_ORDER || order % 2 == 1) {
    return NULL;
  }
  struct fd_coeffs cgrad = all_coeffs[order / 2 - 1];
  uint8_t ncoeffs = (2 * cgrad.p) + 1;
  assert(ncoeffs == order + 1);
  for (int i = 0; i < FD_COEFFS_MAX_COEFF; i++) {
    stencil->coeffs[0][i] =
        (i < ncoeffs) ? cgrad.coeffs[i] / pow(grid->hi, m) : 0.0;
    stencil->coeffs[1][i] =
        (i < ncoeffs) ? cgrad.coeffs[i] / pow(grid->hj, m) : 0.0;
    stencil->coeffs[2][i] =
        (i < ncoeffs) ? cgrad.coeffs[i] / pow(grid->hk, m) : 0.0;
  }
  stencil->p = cgrad.p;
  return stencil;
}

struct fd_stencil *make_gradient(uint8_t order, struct wf_grid *grid,
                                 struct fd_stencil *stencil) {
  return make_stencil(_gradients, 1.0, order, grid, stencil);
}

struct fd_stencil *make_laplacian(uint8_t order, struct wf_grid *grid,
                                  struct fd_stencil *stencil) {
  return make_stencil(_laplacians, 2.0, order, grid, stencil);
}
