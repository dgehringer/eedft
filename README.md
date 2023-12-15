# eedft
Trying to implement Extended Electron Model (EE) by Hofer et al. in Rust

## main ideas
  - use FD (Finite differences) to compute apply the Hamiltonian
    - scales well on dirstributed system (future)
    - easy to accelerate with e.g GPGPUs (future)
    - easy to understand / easy to improve
    - well established solvers for solving matrix free equations
