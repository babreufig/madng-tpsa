cmake_minimum_required(VERSION 3.18)
project(xgtpsa LANGUAGES C)

if(NOT CMAKE_SYSTEM_NAME MATCHES "Linux|Darwin")
  message(FATAL_ERROR "xgtpsa only supports Linux and macOS")
endif()

set(MADNG_SRC "${CMAKE_CURRENT_SOURCE_DIR}/madng/src" CACHE PATH "Path to MAD-NG src directory")
if(NOT EXISTS "${MADNG_SRC}/mad_tpsa.c")
  message(FATAL_ERROR "MADNG_SRC must point to a MAD-NG src directory with mad_tpsa.c")
endif()

set(GTPSA_BUILD_SRC "${CMAKE_CURRENT_BINARY_DIR}/madng_tpsa_src")
file(MAKE_DIRECTORY "${GTPSA_BUILD_SRC}")
file(GLOB MAD_HEADERS CONFIGURE_DEPENDS
  "${MADNG_SRC}/mad_*.h"
  "${MADNG_SRC}/mad_*.hpp"
)
file(GLOB LIBGTPSA_HEADERS CONFIGURE_DEPENDS "${MADNG_SRC}/libgtpsa/*.h")
file(GLOB SSE_SOURCES CONFIGURE_DEPENDS "${MADNG_SRC}/sse/*")
file(COPY ${MAD_HEADERS} ${LIBGTPSA_HEADERS} DESTINATION "${GTPSA_BUILD_SRC}")
if(EXISTS "${MADNG_SRC}/sse")
  file(COPY "${MADNG_SRC}/sse" DESTINATION "${GTPSA_BUILD_SRC}")
endif()

if(CMAKE_C_COMPILER_ID MATCHES "Clang")
  file(READ "${GTPSA_BUILD_SRC}/mad_mem.h" MAD_MEM_H)
  string(REPLACE "malloc(mad_free,1)," "" MAD_MEM_H "${MAD_MEM_H}")
  string(REPLACE ",malloc(mad_free,1)" "" MAD_MEM_H "${MAD_MEM_H}")
  file(WRITE "${GTPSA_BUILD_SRC}/mad_mem.h" "${MAD_MEM_H}")
endif()

set(MADNG_TPSA_SOURCE_NAMES
  mad_tpsa.c
  mad_tpsa_ops.c
  mad_tpsa_div.c
  mad_tpsa_fun.c
  mad_tpsa_io.c
  mad_tpsa_mops.c
  mad_tpsa_comp.c
  mad_tpsa_minv.c
  mad_desc.c
  mad_mono.c
  mad_bit.c
  mad_log.c
  mad_num.c
  mad_mat.c
  mad_vec.c
  mad_str.c
  mad_cst.c
  mad_mem.c
)

set(MADNG_TPSA_SOURCES)
foreach(source_name IN LISTS MADNG_TPSA_SOURCE_NAMES)
  if(EXISTS "${MADNG_SRC}/${source_name}")
    set(source_path "${MADNG_SRC}/${source_name}")
  elseif(EXISTS "${MADNG_SRC}/libgtpsa/${source_name}")
    set(source_path "${MADNG_SRC}/libgtpsa/${source_name}")
  else()
    message(FATAL_ERROR "Could not find ${source_name} in ${MADNG_SRC} or ${MADNG_SRC}/libgtpsa")
  endif()
  configure_file("${source_path}" "${GTPSA_BUILD_SRC}/${source_name}" COPYONLY)
  list(APPEND MADNG_TPSA_SOURCES "${GTPSA_BUILD_SRC}/${source_name}")
endforeach()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/gtpsa_stubs.c"
  "${GTPSA_BUILD_SRC}/gtpsa_stubs.c"
  COPYONLY
)
list(APPEND MADNG_TPSA_SOURCES "${GTPSA_BUILD_SRC}/gtpsa_stubs.c")

add_library(madng_tpsa SHARED ${MADNG_TPSA_SOURCES})
set_target_properties(madng_tpsa PROPERTIES
  PREFIX ""
  OUTPUT_NAME "madng_tpsa"
  C_STANDARD 99
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

set_source_files_properties("${GTPSA_BUILD_SRC}/mad_mem.c" PROPERTIES
  COMPILE_OPTIONS "-std=gnu11"
)

target_include_directories(madng_tpsa PRIVATE "${GTPSA_BUILD_SRC}")
target_compile_options(madng_tpsa PRIVATE
  -include stdlib.h
  -W
  -Wall
  -Wextra
  -O3
  -ffast-math
  -ftree-vectorize
  -Wno-misleading-indentation
  -Wno-empty-body
)

if(CMAKE_C_COMPILER_ID MATCHES "Clang")
  target_compile_options(madng_tpsa PRIVATE
    -Wno-array-parameter
    -Wno-unknown-attributes
  )
else()
  target_compile_options(madng_tpsa PRIVATE
    -Wno-vla-parameter
  )
endif()

if(APPLE)
  target_link_options(madng_tpsa PRIVATE "-Wl,-undefined,error")
  target_link_libraries(madng_tpsa PRIVATE "-framework Accelerate" m)
else()
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  target_link_options(madng_tpsa PRIVATE "-Wl,--no-undefined")
  target_link_libraries(madng_tpsa PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} m)
endif()

set(XGTPSA_SOURCE_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/xgtpsa/lib")
add_custom_command(TARGET madng_tpsa POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${XGTPSA_SOURCE_LIB_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:madng_tpsa>" "${XGTPSA_SOURCE_LIB_DIR}/$<TARGET_FILE_NAME:madng_tpsa>"
  COMMENT "Copying madng_tpsa shared library into src/xgtpsa/lib"
)

install(TARGETS madng_tpsa
  LIBRARY DESTINATION xgtpsa/lib
  RUNTIME DESTINATION xgtpsa/lib
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/xgtpsa/include/madng_tpsa.h"
  DESTINATION xgtpsa/include
)
