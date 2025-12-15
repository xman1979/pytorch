if(NOT __NCCL_INCLUDED)
  set(__NCCL_INCLUDED TRUE)

  if(USE_SYSTEM_NCCL)
    # NCCL_ROOT, NCCL_LIB_DIR, NCCL_INCLUDE_DIR will be accounted in the following line.
    find_package(NCCL REQUIRED)
    if(NCCL_FOUND)
      add_library(__caffe2_nccl INTERFACE)
      target_link_libraries(__caffe2_nccl INTERFACE ${NCCL_LIBRARIES})
      target_include_directories(__caffe2_nccl INTERFACE ${NCCL_INCLUDE_DIRS})
    endif()
  else()
    torch_cuda_get_nvcc_gencode_flag(NVCC_GENCODE)
    string(REPLACE "-gencode;" "-gencode=" NVCC_GENCODE "${NVCC_GENCODE}")
    # this second replacement is needed when there are multiple archs
    string(REPLACE ";-gencode" " -gencode" NVCC_GENCODE "${NVCC_GENCODE}")

    if(DEFINED ENV{MAX_JOBS})
      set(MAX_JOBS "$ENV{MAX_JOBS}")
    else()
      include(ProcessorCount)
      ProcessorCount(NUM_HARDWARE_THREADS)
      # Assume 2 hardware threads per cpu core
      math(EXPR MAX_JOBS "${NUM_HARDWARE_THREADS} / 2")
      # ProcessorCount might return 0, set to a positive number
      if(MAX_JOBS LESS 2)
        set(MAX_JOBS 2)
      endif()
    endif()

    if("${CMAKE_GENERATOR}" MATCHES "Make")
      # Recursive make with jobserver for parallelism, and also put a load limit
      # here to avoid flaky OOM, https://www.gnu.org/software/make/manual/html_node/Parallel.html
      set(MAKE_COMMAND "$(MAKE)" "-l${MAX_JOBS}")
    else()
      # Parallel build with CPU load limit to avoid oversubscription
      set(MAKE_COMMAND "make" "-j${MAX_JOBS}" "-l${MAX_JOBS}")
    endif()

    set(__NCCL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/nccl")
    ExternalProject_Add(nccl_external
      SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/nccl
      BUILD_IN_SOURCE 1
      CONFIGURE_COMMAND ""
      BUILD_COMMAND
        ${MAKE_COMMAND}
        "CXX=${CMAKE_CXX_COMPILER}"
        "CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}"
        "NVCC=${CUDA_NVCC_EXECUTABLE}"
        "NVCC_GENCODE=${NVCC_GENCODE}"
        "BUILDDIR=${__NCCL_BUILD_DIR}"
        "VERBOSE=0"
        "DEBUG=0"
      BUILD_BYPRODUCTS "${__NCCL_BUILD_DIR}/lib/libnccl_static.a"
      INSTALL_COMMAND ""
      )

    set(__NCCL_LIBRARY_DEP nccl_external)
    set(NCCL_LIBRARIES ${__NCCL_BUILD_DIR}/lib/libnccl_static.a)

    set(NCCL_FOUND TRUE)
    add_library(__caffe2_nccl INTERFACE)
    # The following old-style variables are set so that other libs, such as Gloo,
    # can still use it.
    set(NCCL_INCLUDE_DIRS ${__NCCL_BUILD_DIR}/include)
    add_dependencies(__caffe2_nccl ${__NCCL_LIBRARY_DEP})
    target_link_libraries(__caffe2_nccl INTERFACE ${NCCL_LIBRARIES})
    target_include_directories(__caffe2_nccl INTERFACE ${NCCL_INCLUDE_DIRS})
    # nccl includes calls to shm_open/shm_close and therefore must depend on librt on Linux
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      target_link_libraries(__caffe2_nccl INTERFACE rt)
    endif()
    # Export third-party dependencies required by libnccl_static.a
    set(CONDA_LIB_DIR /checkpoint/fairinfra/conda_envs/fair_efficiency/lib)

    # Helper macro to conditionally add static libraries if they exist
    set(__NCCL_THIRD_PARTY_LIBS "")
    macro(add_nccl_dep_if_exists lib_name)
      if(EXISTS "${CONDA_LIB_DIR}/${lib_name}.a")
        list(APPEND __NCCL_THIRD_PARTY_LIBS "${CONDA_LIB_DIR}/${lib_name}.a")
      endif()
    endmacro()

    # Add static libraries from conda
    add_nccl_dep_if_exists(libfolly)
    add_nccl_dep_if_exists(libfolly_exception_tracer)
    add_nccl_dep_if_exists(libfolly_exception_tracer_base)
    add_nccl_dep_if_exists(libglog)
    add_nccl_dep_if_exists(libgflags)
    add_nccl_dep_if_exists(libfmt)
    add_nccl_dep_if_exists(libboost_context)
    add_nccl_dep_if_exists(libboost_program_options)
    add_nccl_dep_if_exists(libboost_filesystem)
    add_nccl_dep_if_exists(libboost_regex)
    add_nccl_dep_if_exists(libboost_system)
    add_nccl_dep_if_exists(libboost_thread)
    add_nccl_dep_if_exists(libboost_atomic)
    add_nccl_dep_if_exists(libdouble-conversion)
    add_nccl_dep_if_exists(libevent)
    add_nccl_dep_if_exists(libsnappy)
    add_nccl_dep_if_exists(libsodium)
    add_nccl_dep_if_exists(libopentelemetry_exporter_otlp_http_log)
    add_nccl_dep_if_exists(libopentelemetry_otlp_recordable)
    add_nccl_dep_if_exists(libopentelemetry_exporter_otlp_http_client)
    add_nccl_dep_if_exists(libopentelemetry_http_client_curl)
    add_nccl_dep_if_exists(libopentelemetry_proto)
    add_nccl_dep_if_exists(libopentelemetry_logs)
    add_nccl_dep_if_exists(libopentelemetry_trace)
    add_nccl_dep_if_exists(libopentelemetry_metrics)
    add_nccl_dep_if_exists(libopentelemetry_resources)
    add_nccl_dep_if_exists(libopentelemetry_common)
    add_nccl_dep_if_exists(libprotobuf)
    add_nccl_dep_if_exists(libcurl)
    add_nccl_dep_if_exists(libzstd)
    add_nccl_dep_if_exists(libpsl)
    add_nccl_dep_if_exists(libz)
    add_nccl_dep_if_exists(libabsl_log_internal_check_op)
    add_nccl_dep_if_exists(libabsl_die_if_null)
    add_nccl_dep_if_exists(libabsl_log_internal_conditions)
    add_nccl_dep_if_exists(libabsl_log_internal_message)
    add_nccl_dep_if_exists(libabsl_log_internal_nullguard)
    add_nccl_dep_if_exists(libabsl_examine_stack)
    add_nccl_dep_if_exists(libabsl_log_internal_format)
    add_nccl_dep_if_exists(libabsl_log_internal_structured_proto)
    add_nccl_dep_if_exists(libabsl_log_internal_log_sink_set)
    add_nccl_dep_if_exists(libabsl_log_sink)
    add_nccl_dep_if_exists(libabsl_log_entry)
    add_nccl_dep_if_exists(libabsl_log_internal_proto)
    add_nccl_dep_if_exists(libabsl_flags_internal)
    add_nccl_dep_if_exists(libabsl_flags_marshalling)
    add_nccl_dep_if_exists(libabsl_flags_reflection)
    add_nccl_dep_if_exists(libabsl_flags_config)
    add_nccl_dep_if_exists(libabsl_flags_program_name)
    add_nccl_dep_if_exists(libabsl_flags_private_handle_accessor)
    add_nccl_dep_if_exists(libabsl_flags_commandlineflag)
    add_nccl_dep_if_exists(libabsl_flags_commandlineflag_internal)
    add_nccl_dep_if_exists(libabsl_log_initialize)
    add_nccl_dep_if_exists(libabsl_log_internal_globals)
    add_nccl_dep_if_exists(libabsl_log_globals)
    add_nccl_dep_if_exists(libabsl_vlog_config_internal)
    add_nccl_dep_if_exists(libabsl_log_internal_fnmatch)
    add_nccl_dep_if_exists(libabsl_raw_hash_set)
    add_nccl_dep_if_exists(libabsl_hashtablez_sampler)
    add_nccl_dep_if_exists(libabsl_random_distributions)
    add_nccl_dep_if_exists(libabsl_random_seed_sequences)
    add_nccl_dep_if_exists(libabsl_random_internal_entropy_pool)
    add_nccl_dep_if_exists(libabsl_random_internal_randen)
    add_nccl_dep_if_exists(libabsl_random_internal_randen_hwaes)
    add_nccl_dep_if_exists(libabsl_random_internal_randen_hwaes_impl)
    add_nccl_dep_if_exists(libabsl_random_internal_randen_slow)
    add_nccl_dep_if_exists(libabsl_random_internal_platform)
    add_nccl_dep_if_exists(libabsl_random_internal_seed_material)
    add_nccl_dep_if_exists(libabsl_random_seed_gen_exception)
    add_nccl_dep_if_exists(libabsl_statusor)
    add_nccl_dep_if_exists(libabsl_status)
    add_nccl_dep_if_exists(libabsl_cord)
    add_nccl_dep_if_exists(libabsl_cordz_info)
    add_nccl_dep_if_exists(libabsl_cord_internal)
    add_nccl_dep_if_exists(libabsl_hash)
    add_nccl_dep_if_exists(libabsl_city)
    add_nccl_dep_if_exists(libabsl_cordz_functions)
    add_nccl_dep_if_exists(libabsl_exponential_biased)
    add_nccl_dep_if_exists(libabsl_cordz_handle)
    add_nccl_dep_if_exists(libabsl_crc_cord_state)
    add_nccl_dep_if_exists(libabsl_crc32c)
    add_nccl_dep_if_exists(libabsl_crc_internal)
    add_nccl_dep_if_exists(libabsl_crc_cpu_detect)
    add_nccl_dep_if_exists(libabsl_leak_check)
    add_nccl_dep_if_exists(libabsl_strerror)
    add_nccl_dep_if_exists(libabsl_str_format_internal)
    add_nccl_dep_if_exists(libabsl_synchronization)
    add_nccl_dep_if_exists(libabsl_stacktrace)
    add_nccl_dep_if_exists(libabsl_symbolize)
    add_nccl_dep_if_exists(libabsl_debugging_internal)
    add_nccl_dep_if_exists(libabsl_demangle_internal)
    add_nccl_dep_if_exists(libabsl_demangle_rust)
    add_nccl_dep_if_exists(libabsl_decode_rust_punycode)
    add_nccl_dep_if_exists(libabsl_utf8_for_code_point)
    add_nccl_dep_if_exists(libabsl_graphcycles_internal)
    add_nccl_dep_if_exists(libabsl_kernel_timeout_internal)
    add_nccl_dep_if_exists(libabsl_malloc_internal)
    add_nccl_dep_if_exists(libabsl_tracing_internal)
    add_nccl_dep_if_exists(libabsl_time)
    add_nccl_dep_if_exists(libabsl_civil_time)
    add_nccl_dep_if_exists(libabsl_time_zone)
    add_nccl_dep_if_exists(libutf8_validity)
    add_nccl_dep_if_exists(libabsl_strings)
    add_nccl_dep_if_exists(libabsl_int128)
    add_nccl_dep_if_exists(libabsl_strings_internal)
    add_nccl_dep_if_exists(libabsl_string_view)
    add_nccl_dep_if_exists(libabsl_base)
    add_nccl_dep_if_exists(libabsl_spinlock_wait)
    add_nccl_dep_if_exists(libabsl_throw_delegate)
    add_nccl_dep_if_exists(libabsl_raw_logging_internal)
    add_nccl_dep_if_exists(libabsl_log_severity)
    add_nccl_dep_if_exists(libssh2)
    add_nccl_dep_if_exists(libnghttp2)
    add_nccl_dep_if_exists(libiconv)

    # Link all collected static libraries
    target_link_libraries(__caffe2_nccl INTERFACE ${__NCCL_THIRD_PARTY_LIBS})

    # Add link directory for shared libraries and system libraries
    target_link_directories(__caffe2_nccl INTERFACE ${CONDA_LIB_DIR})

    # Add shared libraries that don't have static versions (ssl, crypto)
    target_link_libraries(__caffe2_nccl INTERFACE
      ssl
      crypto
      idn2
      unistring
      pthread
      m
      dl
    )
  endif()
endif()
