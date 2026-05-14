cmake_minimum_required(VERSION 3.15)
project(eigenprim LANGUAGES NONE)

install(
  FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/_eigenprim_nvidia_path.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/eigenprim_cuda_activate.pth"
  DESTINATION "."
)
