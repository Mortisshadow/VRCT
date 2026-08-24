cmake_minimum_required(VERSION 3.20)
project(vrct_whisper_worker LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(WHISPER_CPP_TAG a8d002cfd879315632a579e73f0148d06959de36 CACHE STRING "Pinned whisper.cpp v1.7.6 commit")

include(FetchContent)
FetchContent_Declare(whisper_cpp
  GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp.git
  GIT_TAG        ${WHISPER_CPP_TAG}
  GIT_SHALLOW    FALSE
)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(whisper_cpp)

add_executable(vrct-whisper-worker main.cpp)
target_link_libraries(vrct-whisper-worker PRIVATE whisper)
if (WIN32)
  target_compile_definitions(vrct-whisper-worker PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()
install(TARGETS vrct-whisper-worker RUNTIME DESTINATION .)
