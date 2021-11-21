add_library(_ops MODULE)

file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${CUDAPlayground_SOURCE_DIR}/include/**/*.h")
file(GLOB CPP_SRC CONFIGURE_DEPENDS "**/*.cpp" "*.cpp")
file(GLOB CUDA_SRC CONFIGURE_DEPENDS "**/*.cu" "*.cu")

target_sources(_ops
  PRIVATE
    ${CPP_SRC}
    ${CUDA_SRC}
  PUBLIC
    ${HEADER_LIST}
)

target_include_directories(_ops PRIVATE
  ${TORCH_INCLUDE_DIRS}
  ${MAGMA_INCLUDE_DIRS}
  ../include
)

target_link_libraries(_ops
  ${TORCH_LIBRARIES}
  ${TORCH_PYTHON_LIBRARY}
  ${MAGMA_LIBRARY}
  CUDA::cublas
)

target_compile_options(_ops PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)
set_property(TARGET _ops PROPERTY CXX_STANDARD 17)

python_extension_module(_ops)
install(TARGETS _ops DESTINATION cuda_playground/ops)
