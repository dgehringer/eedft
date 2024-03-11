
#include "eedft/stencil.h"
#include "eedft/wf_grid.h"
#include <assert.h>
#include <stdbool.h>

static inline int handle_boundary_periodic(int i, int ni) {
  assert(ni > 0);
  assert(i < 2 * ni);
  assert(i >= -ni);
  if (i < 0) {
    return ni + i;
  } else if (i >= ni) {
    return i - ni;
  } else {
    return i;
  }
}

void laplacian_and_gradient(const SCALAR *restrict f, struct wf_grid *grid,
                            int i, int j, int k, bool handle_boundary,
                            struct fd_stencil *slap, struct fd_stencil *sgrad,
                            SCALAR *lap, SCALAR *fi, SCALAR *fj, SCALAR *fk) {
  int ni = grid->ni, nj = grid->nj, nk = grid->nk;
  int jstride = nk;
  int istride = nj * nk;
  int p = (int)slap->p;
  assert(slap->p == sgrad->p);
  SCALAR vlap = 0.0, vgi = 0.0, vgj = 0.0, vgk = 0.0;
  SCALAR v = f[i * istride + j * jstride + k];
  for (int ci = 0, off = -p; off < (p + 1); ci++, off++) {
    SCALAR vi, vj, vk;
    if (handle_boundary) {
      vi = f[handle_boundary_periodic(i + off, ni) * istride + j * jstride + k];
      vj = f[i * istride + handle_boundary_periodic(j + off, nj) * jstride + k];
      vk = f[i * istride + j * jstride + handle_boundary_periodic(k + off, nk)];
    } else {
      vi = f[(i + off) * istride + j * jstride + k];
      vj = f[i * istride + (j + off) * jstride + k];
      vk = f[i * istride + j * jstride + (k + off)];
    }
    vlap += vi * slap->coeffs[0][ci] + vj * slap->coeffs[1][ci] +
            vk * slap->coeffs[2][ci];
    vgi += (vi - v) * sgrad->coeffs[0][ci];
    vgj += (vj - v) * sgrad->coeffs[1][ci];
    vgk += (vk - v) * sgrad->coeffs[2][ci];
  }
  *lap = vlap;
  *fi = vgi;
  *fj = vgj;
  *fk = vgk;
}
