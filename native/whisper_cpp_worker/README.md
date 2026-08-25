# VRCT whisper.cpp worker

This is a persistent, binary-protocol transcription worker. It fetches whisper.cpp v1.9.3 at pinned commit `371b5a7561823ab2bb32142d2751e35e7534727b` and enables GGML Vulkan. The Windows release helper places the executable in `src-tauri/bin/_internal/whisper_cpp` for PyInstaller collection. VRCT uses greedy decoding and explicitly disables flash attention for broad Windows Vulkan driver compatibility.

Build on Windows with CMake 3.20+, Visual Studio 2022, and the Vulkan SDK:

```bat
cmake -S native\whisper_cpp_worker -B native\whisper_cpp_worker\build -A x64 -DGGML_VULKAN=ON
cmake --build native\whisper_cpp_worker\build --config Release
```
