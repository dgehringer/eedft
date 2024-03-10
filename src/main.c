
#include "eedft/stencil.h"


int main() {

  struct fd_stencil stencil;
  struct wf_grid grid = { .hi = 1.0, .hj = 1.0, .hk = 1.0 };
 
  for (uint8_t order = 0; order < 7; order++) {
    make_gradient(order, &grid, &stencil);
    //if (order < FD_MIN_ORDER || order > FD_MAX_ORDER) ASSERT_EQ(result, nullptr);  
  }
  return 0;
}
