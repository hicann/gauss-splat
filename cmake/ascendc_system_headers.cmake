# CANN 9.2 or later is recommended. Use its normal header search by default.
# This opt-in workaround addresses missing host headers reported with CANN 9.1
# on Debian/aarch64; it is not a requirement for every CANN installation.
option(ASCENDC_ENABLE_HOST_HEADERS
    "Enable the host C++ header workaround for Ascend C" OFF)
set(ASCENDC_SYSTEM_INCLUDE_DIRS "" CACHE STRING
    "Explicit host C++ header directories (semicolon-separated; enables workaround)")

function(ascendc_system_header_options output)
  set(header_dirs ${ASCENDC_SYSTEM_INCLUDE_DIRS})
  # The environment switch also works with build.sh/pip and an existing cache.
  # Explicit directories are themselves an opt-in, retaining manual overrides.
  if(NOT header_dirs AND NOT ASCENDC_ENABLE_HOST_HEADERS
      AND NOT "$ENV{ASCENDC_ENABLE_HOST_HEADERS}")
    message(STATUS "Ascend C host header workaround disabled; CANN 9.2 or later is recommended")
    set(${output} "" PARENT_SCOPE)
    return()
  endif()
  if(NOT header_dirs)
    foreach(include_dir IN LISTS CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
      if(include_dir MATCHES "c\\+\\+")
        list(APPEND header_dirs "${include_dir}")
      endif()
    endforeach()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=include
        OUTPUT_VARIABLE gcc_include_dir OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE gcc_result)
      if(gcc_result EQUAL 0 AND IS_DIRECTORY "${gcc_include_dir}")
        list(APPEND header_dirs "${gcc_include_dir}")
      endif()
    endif()
    # Preserve manually supplied paths as explicit flags too.
    if(DEFINED ENV{CPLUS_INCLUDE_PATH})
      string(REPLACE ":" ";" extra_dirs "$ENV{CPLUS_INCLUDE_PATH}")
      list(APPEND header_dirs ${extra_dirs})
    endif()
  endif()
  list(REMOVE_DUPLICATES header_dirs)
  foreach(header IN ITEMS cstdint cfloat type_traits)
    set(found FALSE)
    foreach(include_dir IN LISTS header_dirs)
      if(EXISTS "${include_dir}/${header}")
        set(found TRUE)
      endif()
    endforeach()
    if(NOT found)
      message(FATAL_ERROR
        "Ascend C host header workaround cannot locate ${header}. "
        "CANN 9.2 or later is recommended. Install the matching g++/libstdc++ "
        "development headers, select the compiler with CXX, or set "
        "ASCENDC_SYSTEM_INCLUDE_DIRS in CMake (semicolon-separated paths). "
        "For Debian aarch64 GCC 12: /usr/include/c++/12;"
        "/usr/include/aarch64-linux-gnu/c++/12;/usr/lib/gcc/aarch64-linux-gnu/12/include")
    endif()
  endforeach()
  set(options)
  foreach(include_dir IN LISTS header_dirs)
    if(NOT IS_DIRECTORY "${include_dir}")
      message(FATAL_ERROR "Ascend C header directory does not exist: ${include_dir}")
    endif()
    list(APPEND options "-I${include_dir}")
  endforeach()
  message(STATUS "Ascend C system header directories: ${header_dirs}")
  set(${output} ${options} PARENT_SCOPE)
endfunction()
