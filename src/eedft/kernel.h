#ifndef EEDFT_KERNEL_H
#define EEDFT_KERNEL_H

#include "wf_grid.h"

#ifdef EEDFT_TESTING
void laplacian_and_gradient(const double * f,
                                   struct wf_grid *grid, int i, int j, int k,
                                   bool handle_boundary,
                                   struct fd_stencil *slap,
                                   struct fd_stencil *sgrad, double *lap,
                                   double *fi, double *fj, double *fk);
#endif // DEBUG

#endif // !EEDFT_KERNEL_H
