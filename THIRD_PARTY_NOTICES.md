# Third-Party Notices

EigenPrim builds CUDA fatbins from Eigen headers.

## Eigen

- Project: Eigen
- Source: https://gitlab.com/libeigen/eigen
- Exact source used by the default build:
  https://gitlab.com/libeigen/eigen/-/tree/bc3b39870ecb690a623a3f49149a358b95c5781d
- Version: 5.0.1
- Commit: bc3b39870ecb690a623a3f49149a358b95c5781d
- Local modifications: none
- License: primarily Mozilla Public License 2.0. Some Eigen files contain
  third-party code under BSD or other MPL-2.0-compatible licenses; see
  `LICENSES/eigen/COPYING.README` and the other
  `LICENSES/eigen/COPYING.*` files.

Recipients can obtain the Eigen source used by EigenPrim from the exact source
link above, or from the `thirdparty/eigen` submodule in the EigenPrim source
tree. If a distributor builds EigenPrim with a different Eigen checkout via
`EIGEN_INCLUDE_DIR` or a CMake `EIGENPRIM_EIGEN_INCLUDE_DIR` override,
that distributor should update this notice to identify the Eigen source
actually used.
