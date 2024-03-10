#ifndef EEDFT_WF_GRID_H
#define EEDFT_WF_GRID_H

#define SCALAR double

struct wf_grid {
  SCALAR *rho12;
  SCALAR *si12;
  SCALAR *sj12;
  SCALAR *sk12;
  int ni;
  int nj;
  int nk;
  SCALAR hi;
  SCALAR hj;
  SCALAR hk;
};

#endif // EEDFT_WF_GRID_H
