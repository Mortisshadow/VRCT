#include <whisper.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

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
}

int main(int argc, char **argv) {
    std::string model; int threads = 4; int device = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::max(1, std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--device") && i + 1 < argc) device = std::max(0, std::atoi(argv[++i]));
    }
    if (model.empty()) { std::cerr << "--model is required\n"; return 2; }
    const auto load_started = std::chrono::steady_clock::now();
    auto lp = whisper_context_default_params(); lp.use_gpu = true; lp.gpu_device = device;
    whisper_context *ctx = whisper_init_from_file_with_params(model.c_str(), lp);
    if (!ctx) { std::cerr << "failed to load whisper model\n"; return 3; }
    const char *raw_system_info = whisper_print_system_info();
    const std::string system_info = escaped(raw_system_info ? raw_system_info : "whisper.cpp Vulkan");
    const bool gpu_active = lp.use_gpu && system_info.find("VULKAN = 1") != std::string::npos;
    const double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - load_started).count();
    std::cerr << "whisper.cpp backend initialized; gpu_active=" << gpu_active
              << " threads=" << threads << " device=" << system_info << " load_ms=" << load_ms << "\n";
    std::cout << "VRCT_READY\t" << (gpu_active ? 1 : 0) << "\t" << system_info << "\t" << load_ms << "\n" << std::flush;
    for (;;) {
        uint32_t magic, version, command, lang_len, samples; float avg_logprob, no_speech; int32_t ngram;
        if (!read_one(magic) || !read_one(version) || !read_one(command) || !read_one(lang_len) || !read_one(samples) ||
            !read_one(avg_logprob) || !read_one(no_speech) || !read_one(ngram)) break;
        std::string language; if (!read_bytes(language, lang_len)) break;
        std::vector<float> pcm(samples); if (samples && !std::cin.read(reinterpret_cast<char *>(pcm.data()), samples * sizeof(float))) break;
        if (magic != MAGIC || version != VERSION) { response(-2, 0, 0, "", "", "invalid request header"); continue; }
        if (command == 2) break;
        if (command != 1) { response(-3, 0, 0, "", "", "unknown command"); continue; }
        auto started = std::chrono::steady_clock::now();
        auto params = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
        params.n_threads = threads; params.no_timestamps = true; params.temperature = 0.0f;
        params.language = language.empty() ? "auto" : language.c_str(); params.translate = false;
        params.no_speech_thold = no_speech; params.logprob_thold = avg_logprob; params.beam_search.beam_size = 5;
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
