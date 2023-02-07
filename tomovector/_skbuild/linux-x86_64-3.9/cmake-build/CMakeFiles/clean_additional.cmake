# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Release")
  file(REMOVE_RECURSE
  "src/cuda/CMakeFiles/radonusfft.dir/radonusfftPYTHON_wrap.cxx"
  "src/cuda/radonusfft.py"
  )
endif()
