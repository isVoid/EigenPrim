cmake_minimum_required(VERSION 3.27)
project(eigenprim LANGUAGES CUDA)

find_path(
  EIGENPRIM_EIGEN_INCLUDE_DIR
  NAMES Eigen/Dense
  HINTS
    "$ENV{EIGEN_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/eigen"
    "$ENV{CONDA_PREFIX}/include/eigen3"
    "$ENV{PREFIX}/include/eigen3"
  PATH_SUFFIXES eigen3
  REQUIRED
)
message(STATUS "Using Eigen headers: ${EIGENPRIM_EIGEN_INCLUDE_DIR}")

set(EIGENPRIM_CUDA_ARCHITECTURES "75;80;90" CACHE STRING "CUDA architectures to include in eigenprim fatbins")
set(_eigenprim_include_dir "${CMAKE_CURRENT_SOURCE_DIR}/eigenprim/include")

function(eigenprim_add_fatbin name source)
  add_library("${name}_fatbin" OBJECT "${_eigenprim_include_dir}/${source}")
  target_include_directories(
    "${name}_fatbin"
    PRIVATE
      "${EIGENPRIM_EIGEN_INCLUDE_DIR}"
      "${_eigenprim_include_dir}"
  )
  target_compile_options("${name}_fatbin" PRIVATE --expt-relaxed-constexpr)
  set_target_properties(
    "${name}_fatbin"
    PROPERTIES
      CUDA_ARCHITECTURES "${EIGENPRIM_CUDA_ARCHITECTURES}"
      CUDA_FATBIN_COMPILATION ON
      CUDA_SEPARABLE_COMPILATION ON
  )
  install(FILES "$<TARGET_OBJECTS:${name}_fatbin>" DESTINATION "eigenprim/fatbin")
endfunction()

eigenprim_add_fatbin(generic generic.cu)
eigenprim_add_fatbin(matrix matrix.cu)
