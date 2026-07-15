cmake_minimum_required(VERSION 3.16)
project(qwen3_tts_ncnn LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(QWEN3_TTS_BUILD_TOOLS "Build internal validation/debug tools" OFF)
option(QWEN3_TTS_USE_NCNN_PACKAGE "Use an installed ncnn CMake package" ON)

if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

set(QWEN3_TTS_NCNN_ROOT "/workspace/Tencent/ncnn" CACHE PATH "Path to local ncnn source tree for legacy builds")
set(QWEN3_TTS_NCNN_BUILD_DIR "${QWEN3_TTS_NCNN_ROOT}/build-linux" CACHE PATH "Path to local ncnn build tree for legacy builds")

find_package(Threads REQUIRED)
find_package(OpenMP)

if(QWEN3_TTS_USE_NCNN_PACKAGE)
    find_package(ncnn CONFIG REQUIRED)
    set(QWEN3_TTS_NCNN_LIBS ncnn Threads::Threads)
else()
    if(NOT EXISTS "${QWEN3_TTS_NCNN_BUILD_DIR}/src/${CMAKE_STATIC_LIBRARY_PREFIX}ncnn${CMAKE_STATIC_LIBRARY_SUFFIX}")
        message(FATAL_ERROR "Legacy ncnn library not found under QWEN3_TTS_NCNN_BUILD_DIR=${QWEN3_TTS_NCNN_BUILD_DIR}")
    endif()
    set(QWEN3_TTS_NCNN_LIBS
        "${QWEN3_TTS_NCNN_BUILD_DIR}/src/${CMAKE_STATIC_LIBRARY_PREFIX}ncnn${CMAKE_STATIC_LIBRARY_SUFFIX}"
        Threads::Threads
    )
    if(EXISTS "${QWEN3_TTS_NCNN_BUILD_DIR}/glslang/glslang/${CMAKE_STATIC_LIBRARY_PREFIX}glslang${CMAKE_STATIC_LIBRARY_SUFFIX}")
        list(APPEND QWEN3_TTS_NCNN_LIBS
            "${QWEN3_TTS_NCNN_BUILD_DIR}/glslang/glslang/${CMAKE_STATIC_LIBRARY_PREFIX}glslang${CMAKE_STATIC_LIBRARY_SUFFIX}"
            "${QWEN3_TTS_NCNN_BUILD_DIR}/glslang/SPIRV/${CMAKE_STATIC_LIBRARY_PREFIX}SPIRV${CMAKE_STATIC_LIBRARY_SUFFIX}"
        )
        if(CMAKE_DL_LIBS)
            list(APPEND QWEN3_TTS_NCNN_LIBS ${CMAKE_DL_LIBS})
        endif()
    endif()
    if(ANDROID)
        list(APPEND QWEN3_TTS_NCNN_LIBS log android)
    endif()
endif()

add_library(qwen3_tts_runtime STATIC
    src/qwen3_tts_runtime.cpp
    src/qwen3_tts_frontend.cpp
    src/qwen_bpe.cpp
)
target_include_directories(qwen3_tts_runtime PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
if(NOT QWEN3_TTS_USE_NCNN_PACKAGE)
    target_include_directories(qwen3_tts_runtime PUBLIC
        "${QWEN3_TTS_NCNN_ROOT}/src"
        "${QWEN3_TTS_NCNN_BUILD_DIR}/src"
    )
endif()
target_link_libraries(qwen3_tts_runtime PUBLIC ${QWEN3_TTS_NCNN_LIBS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(qwen3_tts_runtime PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(qwen3_tts src/qwen3_tts_example.cpp)
target_link_libraries(qwen3_tts PRIVATE qwen3_tts_runtime)

add_executable(qwen3_tts_benchmark src/qwen3_tts_benchmark.cpp)
target_link_libraries(qwen3_tts_benchmark PRIVATE qwen3_tts_runtime)

function(qwen3_tts_add_internal_tool target source)
    add_executable(${target} ${source})
    if(NOT QWEN3_TTS_USE_NCNN_PACKAGE)
        target_include_directories(${target} PRIVATE
            "${QWEN3_TTS_NCNN_ROOT}/src"
            "${QWEN3_TTS_NCNN_BUILD_DIR}/src"
        )
    endif()
    target_link_libraries(${target} PRIVATE ${QWEN3_TTS_NCNN_LIBS})
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endfunction()

function(qwen3_tts_add_runtime_tool target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE qwen3_tts_runtime)
endfunction()

if(QWEN3_TTS_BUILD_TOOLS)
    qwen3_tts_add_internal_tool(qwen3_tts_decoder_check src/qwen3_tts_decoder_check.cpp)
    qwen3_tts_add_internal_tool(qwen3_tts_hidden_decoder_check src/qwen3_tts_hidden_decoder_check.cpp)
    qwen3_tts_add_internal_tool(qwen3_tts_front_check src/qwen3_tts_front_check.cpp)
    qwen3_tts_add_internal_tool(qwen3_tts_talker_prefill_check src/qwen3_tts_talker_prefill_check.cpp)
    qwen3_tts_add_internal_tool(qwen3_tts_code_predictor_loop_check src/qwen3_tts_code_predictor_loop_check.cpp)
    qwen3_tts_add_internal_tool(qwen3_tts_codepred_kv_check src/qwen3_tts_codepred_kv_check.cpp)
    qwen3_tts_add_internal_tool(qwen3_tts_talker_external_check src/qwen3_tts_talker_external_check.cpp)
    qwen3_tts_add_internal_tool(qwen3_tts_talker_kv_decode_check src/qwen3_tts_talker_kv_decode_check.cpp)
    qwen3_tts_add_internal_tool(qwen3_tts_talker_loop_check src/qwen3_tts_talker_loop_check.cpp)
    qwen3_tts_add_internal_tool(qwen3_tts_mat_check src/qwen3_tts_mat_check.cpp)
    qwen3_tts_add_internal_tool(qwen3_tts_decoder_chunk_check src/qwen3_tts_decoder_chunk_check.cpp)
    qwen3_tts_add_internal_tool(qwen3_tts_decoder_blob_diff src/qwen3_tts_decoder_blob_diff.cpp)
    qwen3_tts_add_internal_tool(qwen3_tts_codes_to_wav_check src/qwen3_tts_codes_to_wav_check.cpp)
    qwen3_tts_add_internal_tool(qwen3_tts_full_pipeline_check src/qwen3_tts_full_pipeline_check.cpp)

    qwen3_tts_add_runtime_tool(qwen3_tts_runtime_codes_to_wav_check src/qwen3_tts_runtime_codes_to_wav_check.cpp)
    qwen3_tts_add_runtime_tool(qwen3_tts_runtime_full_check src/qwen3_tts_runtime_full_check.cpp)
    qwen3_tts_add_runtime_tool(qwen3_tts_model_json_check src/qwen3_tts_model_json_check.cpp)
    qwen3_tts_add_runtime_tool(qwen3_tts_frontend_check src/qwen3_tts_frontend_check.cpp)
    qwen3_tts_add_runtime_tool(qwen_bpe_dump src/qwen_bpe_dump.cpp)
endif()

if(BUILD_TESTING)
    add_test(NAME qwen3_tts_help COMMAND qwen3_tts --help)
endif()
