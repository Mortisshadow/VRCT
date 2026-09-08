#include <whisper.h>
#include <ggml-backend.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {
constexpr uint32_t MAGIC = 0x54524356u; // "VCRT" little-endian
constexpr uint32_t VERSION = 1;
template <typename T> bool read_one(T &v) { return static_cast<bool>(std::cin.read(reinterpret_cast<char *>(&v), sizeof(v))); }
bool read_bytes(std::string &s, uint32_t n) {
    s.resize(n);
    return n == 0 || static_cast<bool>(std::cin.read(s.data(), n));
}
template <typename T> void write_one(const T &v) { std::cout.write(reinterpret_cast<const char *>(&v), sizeof(v)); }
void response(int32_t status, float confidence, double elapsed, const std::string &lang,
              const std::string &text, const std::string &error) {
    write_one(MAGIC); write_one(status); write_one(confidence); write_one(elapsed);
    uint32_t ll = static_cast<uint32_t>(lang.size()), tl = static_cast<uint32_t>(text.size()), el = static_cast<uint32_t>(error.size());
    write_one(ll); write_one(tl); write_one(el);
    if (ll) std::cout.write(lang.data(), ll); if (tl) std::cout.write(text.data(), tl); if (el) std::cout.write(error.data(), el);
    std::cout.flush();
}
std::string escaped(std::string s) { for (char &c : s) if (c == '\t' || c == '\r' || c == '\n') c = ' '; return s; }
ggml_backend_dev_t gpu_device_at(int requested) {
    int gpu_index = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        if (gpu_index++ == requested) return dev;
    }
    return nullptr;
}
}

int main(int argc, char **argv) {
#ifdef _WIN32
    // Python and this worker exchange packed structs and raw float32 samples.
    // MSVC starts standard streams in text mode, where CR/LF translation and
    // the legacy Ctrl-Z EOF marker can corrupt or truncate arbitrary audio.
    // The ready line remains valid in binary mode and is still newline-delimited.
    if (_setmode(_fileno(stdin), _O_BINARY) == -1 ||
        _setmode(_fileno(stdout), _O_BINARY) == -1) {
        std::cerr << "failed to switch worker pipes to binary mode\n";
        return 4;
    }
#endif
    std::string model; int threads = 4; int device = 0; bool probe = false; bool pipe_probe = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::max(1, std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--device") && i + 1 < argc) device = std::max(0, std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--probe")) probe = true;
        else if (!std::strcmp(argv[i], "--pipe-probe")) pipe_probe = true;
    }
    if (pipe_probe) {
        char payload[4];
        if (!std::cin.read(payload, sizeof(payload))) return 8;
        std::cout.write(payload, sizeof(payload));
        std::cout.flush();
        return std::cout ? 0 : 9;
    }
    if (probe) {
        const char *raw = whisper_print_system_info();
        std::cout << "VRCT_PROBE\tGGML_VULKAN=1\tdevices=" << ggml_backend_dev_count()
                  << "\t" << escaped(raw ? raw : "unknown") << "\n";
        return 0;
    }
    if (model.empty()) { std::cerr << "--model is required\n"; return 2; }
    const auto load_started = std::chrono::steady_clock::now();
    auto lp = whisper_context_default_params(); lp.use_gpu = true; lp.flash_attn = false; lp.gpu_device = device;
    whisper_context *ctx = whisper_init_from_file_with_params(model.c_str(), lp);
    if (!ctx) { std::cerr << "failed to load whisper model\n"; return 3; }
    const char *raw_system_info = whisper_print_system_info();
    const auto selected_device = gpu_device_at(device);
    const bool gpu_active = selected_device != nullptr;
    const std::string device_name = selected_device
        ? escaped(std::string(ggml_backend_dev_name(selected_device)) + " (" + ggml_backend_dev_description(selected_device) + ")")
        : "no registered Vulkan GPU at index " + std::to_string(device);
    const std::string system_info = device_name + "; " + escaped(raw_system_info ? raw_system_info : "whisper.cpp");
    const double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - load_started).count();
    std::cerr << "whisper.cpp backend initialized; gpu_active=" << gpu_active
              << " threads=" << threads << " device=" << system_info << " load_ms=" << load_ms << "\n";
    std::cout << "VRCT_READY\t" << (gpu_active ? 1 : 0) << "\t" << system_info << "\t" << load_ms << "\n" << std::flush;
    std::vector<float> pcm;
    for (;;) {
        uint32_t magic, version, command, lang_len, samples; float avg_logprob, no_speech; int32_t ngram;
        if (!read_one(magic) || !read_one(version) || !read_one(command) || !read_one(lang_len) || !read_one(samples) ||
            !read_one(avg_logprob) || !read_one(no_speech) || !read_one(ngram)) {
            std::cerr << "worker stdin closed while waiting for a request header\n";
            whisper_free(ctx);
            return 10;
        }
        std::cerr << "request command=" << command << " samples=" << samples
                  << " language_bytes=" << lang_len << "\n";
        if (lang_len > 64 || samples > 16000u * 60u * 30u) {
            std::cerr << "request rejected due to invalid payload size\n";
            return 5;
        }
        std::string language;
        if (!read_bytes(language, lang_len)) {
            std::cerr << "worker stdin closed while reading request language\n";
            return 6;
        }
        pcm.resize(samples);
        if (samples && !std::cin.read(reinterpret_cast<char *>(pcm.data()), samples * sizeof(float))) {
            std::cerr << "worker stdin closed while reading float32 audio payload\n";
            return 7;
        }
        if (magic != MAGIC || version != VERSION) { response(-2, 0, 0, "", "", "invalid request header"); continue; }
        if (command == 2) {
            std::cerr << "worker received graceful shutdown command\n";
            break;
        }
        if (command != 1) { response(-3, 0, 0, "", "", "unknown command"); continue; }
        const bool silent = std::none_of(pcm.begin(), pcm.end(), [](float sample) { return std::fabs(sample) > 1.0e-6f; });
        if (pcm.size() < 1600 || silent) {
            response(0, 0, 0, language, "", "");
            continue;
        }
        auto started = std::chrono::steady_clock::now();
        // Greedy decoding is whisper.cpp's normal low-latency path. Beam search
        // multiplies Vulkan decoder work and has triggered driver crashes on
        // some AMD devices without materially helping short VRCT phrases.
        auto params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        params.n_threads = threads; params.no_timestamps = true; params.temperature = 0.0f;
        params.language = language.empty() ? "auto" : language.c_str(); params.translate = false;
        params.no_speech_thold = no_speech; params.logprob_thold = avg_logprob; params.greedy.best_of = 1;
        // Mic and speaker intentionally share the resident model/context. Do
        // not leak decoder text history between independent audio streams;
        // VRCT carries a small acoustic overlap and merges stable text itself.
        params.no_context = true; params.max_tokens = 96; params.suppress_nst = false;
        // Live chunks must have bounded latency. Retrying the same decode at
        // increasing temperatures can multiply inference time on weak/noisy
        // input; whisper.cpp's streaming example disables this fallback too.
        params.temperature_inc = -1.0f;
        params.single_segment = false; params.print_progress = false; params.print_realtime = false; params.print_timestamps = false;
        int rc = whisper_full(ctx, params, pcm.data(), static_cast<int>(pcm.size()));
        double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        if (rc != 0) { response(-4, 0, elapsed, "", "", "whisper inference failed"); continue; }
        std::string text; float confidence = 0.0f; int confidence_count = 0; int n = whisper_full_n_segments(ctx);
        for (int i = 0; i < n; ++i) {
            text += whisper_full_get_segment_text(ctx, i);
            const int nt = whisper_full_n_tokens(ctx, i);
            for (int t = 0; t < nt; ++t) { confidence += whisper_full_get_token_p(ctx, i, t); ++confidence_count; }
        }
        if (confidence_count > 0) confidence /= confidence_count;
        const char *detected = whisper_lang_str(whisper_full_lang_id(ctx));
        std::cerr << "transcription_ms=" << elapsed << " segments=" << n << "\n";
        response(0, confidence, elapsed, detected ? detected : "", text, "");
    }
    whisper_free(ctx); return 0;
}
