# VRCT whisper.cpp worker

This is a persistent, binary-protocol transcription worker. It fetches whisper.cpp v1.7.6 at pinned commit `a8d002cfd879315632a579e73f0148d06959de36` and enables GGML Vulkan. The Windows release helper places the executable in `src-tauri/bin/_internal/whisper_cpp` for PyInstaller collection.

Build on Windows with CMake 3.20+, Visual Studio 2022, and the Vulkan SDK:

```bat
cmake -S native\whisper_cpp_worker -B native\whisper_cpp_worker\build -A x64 -DGGML_VULKAN=ON
cmake --build native\whisper_cpp_worker\build --config Release
```
