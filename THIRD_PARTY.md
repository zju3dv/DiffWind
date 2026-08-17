# Third-party components

DiffWind includes or adapts the following research code. Retain the original
copyright notices and license files when redistributing this repository.

- The MPM implementation is adapted from
  [PAC-NeRF](https://github.com/xuan-li/PAC-NeRF), particularly its MPM
  simulator.
- The LBM implementation is adapted from
  [Computational-Biomimetics-of-Winged-Seeds](https://github.com/leqiqin/Computational-Biomimetics-of-Winged-Seeds).
- The Gaussian model, cameras, and renderer are based on
  [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).
- `third_party/diff-gaussian-rasterization` is the differentiable Gaussian-flow
  rasterizer from
  [Zerg-Overmind/diff-gaussian-rasterization](https://github.com/Zerg-Overmind/diff-gaussian-rasterization),
  revision `2917773c24af4decc298e511c3bf65c5397689fd`. Its own license is retained
  in that directory.
