/**
 * api_server.c — Local inference API server for wubuwizard
 *
 * OpenAI-compatible HTTP server for educational/research use.
 * Provides a REST API wrapping the infer_text_gpu binary.
 *
 * ============================================================================
 * DISCLAIMER
 * ============================================================================
 * This software is provided for EDUCATIONAL AND RESEARCH PURPOSES only.
 * It is openly licensed scaffolding — anyone may build, deploy, or sell
 * inference services built on this foundation.
 *
 * THE AUTHORS ASSUME NO LIABILITY for any use, including but not limited to:
 *   - Security vulnerabilities in deployment configurations
 *   - Data privacy violations by downstream operators
 *   - Regulatory compliance (GDPR, CCPA, EU AI Act, etc.)
 *
 * Operators are responsible for their own security posture, API key management,
 * rate limiting, content filtering, and legal compliance.
 * ============================================================================
 *
 * Protocol: HTTP/1.1 with optional TLS (OpenSSL)
 * Endpoints:
 *   POST /v1/chat/completions  — Chat completions (OpenAI-compatible)
 *   POST /v1/completions       — Text completions
 *   GET  /v1/models            — List available models
 *   GET  /health               — Health check
 *
 * Usage:
 *   ./api_server             # Port 8080, no auth, no TLS
 *   ./api_server --port 8443 --tls cert.pem key.pem
 *   ./api_server --sandbox   # Fake responses, isolated testing
 *   ./api_server --auth "sk-..."  # Require API key
 *
 * Environment:
 *   INFER_BIN=./infer_text_gpu  [default]
 *   MODEL_PATH=/path/to/model.gguf [default from config]
 *   API_PORT=8080
 *   API_AUTH_KEY=sk-...
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <math.h>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <io.h>
typedef int socklen_t;
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/wait.h>
#endif
#include "wubu_banner.h"   /* UX: consistent banner + stats formatting */
#include <fcntl.h>
#include <ctype.h>
#include "wubu_core_dumps.h"

/* OpenSSL for TLS support */
#if defined(__has_include)
#  if __has_include(<openssl/ssl.h>)
#    define WUBU_WITH_TLS 1
#  endif
#endif
#ifdef WUBU_WITH_TLS
#include <openssl/ssl.h>
#include <openssl/err.h>
#else
/* TLS unavailable: keep the same call sites compiling as no-ops. */
typedef void SSL;
typedef void SSL_CTX;
#define SSL_write(...) 0
#define SSL_read(...) 0
#define SSL_shutdown(...) 0
#define SSL_free(...) ((void)0)
#define SSL_new(...) NULL
#define SSL_set_fd(...) 0
#define SSL_CTX_new(...) NULL
#define SSL_CTX_free(...) ((void)0)
#define SSL_CTX_use_certificate_file(...) 0
#define SSL_CTX_use_PrivateKey_file(...) 0
#define SSL_load_error_strings() ((void)0)
#define OPENSSL_init_ssl(...) 0
#define TLS_server_method() NULL
#define SSL_FILETYPE_PEM 0
#endif

/* ================================================================
 *  Constants
 * ================================================================ */

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)

#define MAX_HEADERS 16384
#define MAX_BODY    1048576  /* 1 MB */
#define MAX_PATH    4096
#define MAX_TOKENS  4096
#define RATE_WINDOW 60       /* seconds */
#define RATE_LIMIT  60       /* requests per window */
#define DEFAULT_PORT 8080

/* ================================================================
 *  CLI flags / config
 * ================================================================ */

static int     g_port = DEFAULT_PORT;
static int     g_sandbox = 0;
static char    g_auth_key[256] = "";
static int     g_use_tls = 0;
static char    g_tls_cert[1024] = "";
static char    g_tls_key[1024] = "";
static char    g_infer_bin[1024] = "./infer_text_gpu";
static char    g_model_path[1024] = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
static volatile int g_running = 1;

/* Rate limiter state */
typedef struct { time_t timestamps[RATE_LIMIT]; int count; } rate_bucket_t;
static rate_bucket_t g_rate_bucket;
static int g_rate_count = 0;

/* ================================================================
 *  Signal handler
 * ================================================================ */

static void handle_signal(int sig) {
    (void)sig;
    g_running = 0;
    fprintf(stderr, "\n[server] Shutting down...\n");
}

/* ================================================================
 *  Helpers
 * ================================================================ */

static void print_banner(void) {
    wubu_print_banner("Local Inference API Server",
                      "OpenAI-compatible · Research-only");
    wubu_print_stat("Port", "%d", g_port);
    wubu_print_stat("TLS", "%s", g_use_tls ? "yes" : "no");
    wubu_print_stat("Sandbox", "%s", g_sandbox ? "yes" : "no");
    wubu_print_stat("Model", "%s", g_model_path);
    wubu_print_stat("Infer", "%s", g_infer_bin);
    wubu_print_stat("Auth", "%s", g_auth_key[0] ? "enabled" : "none (open)");
    printf("\n");
}

/* ================================================================
 *  JSON helpers (minimal, inline)
 * ================================================================ */

/* Escape a string for JSON */
static char *json_escape(const char *s) {
    if (!s) return strdup("null");
    size_t cap = strlen(s) * 2 + 3;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    size_t j = 0;
    out[j++] = '"';
    for (const char *p = s; *p; p++) {
        unsigned char c = (unsigned char)*p;
        switch (c) {
            case '"':  out[j++] = '\\'; out[j++] = '"';  break;
            case '\\': out[j++] = '\\'; out[j++] = '\\'; break;
            case '\n': out[j++] = '\\'; out[j++] = 'n';  break;
            case '\r': out[j++] = '\\'; out[j++] = 'r';  break;
            case '\t': out[j++] = '\\'; out[j++] = 't';  break;
            default:
                if (c < 0x20) {
                    if (j + 6 >= cap) { cap *= 2; out = (char *)realloc(out, cap); }
                    j += snprintf(out + j, cap - j, "\\u%04x", c);
                } else {
                    out[j++] = (char)c;
                }
                break;
        }
        if (j + 8 >= cap) { cap *= 2; out = (char *)realloc(out, cap); }
    }
    out[j++] = '"';
    out[j] = '\0';
    return out;
}

/* Build simple JSON response */
static char *make_json_response(const char *content) {
    /* This is simplified — real impl would use a JSON builder */
    size_t clen = content ? strlen(content) : 0;
    size_t bufsz = clen + 4096;
    char *buf = (char *)malloc(bufsz);
    if (!buf) return NULL;

    char *escaped = json_escape(content ? content : "");
    time_t now = time(NULL);
    char ts[64];
    snprintf(ts, sizeof(ts), "%ld", (long)now);

    snprintf(buf, bufsz,
        "{"
        "\"id\":\"chatcmpl-%s\","
        "\"object\":\"chat.completion\","
        "\"created\":%s,"
        "\"model\":\"wubuwizard-qwen3.6\","
        "\"choices\":[{"
          "\"index\":0,"
          "\"message\":{\"role\":\"assistant\",\"content\":%s},"
          "\"finish_reason\":\"stop\""
        "}],"
        "\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}"
        "}",
        ts, ts, escaped);

    free(escaped);
    return buf;
}

/* ================================================================
 *  Rate limiter
 * ================================================================ */

static int check_rate_limit(void) {
    time_t now = time(NULL);
    time_t cutoff = now - RATE_WINDOW;

    /* Prune old entries */
    int new_count = 0;
    for (int i = 0; i < g_rate_count; i++) {
        if (g_rate_bucket.timestamps[i] > cutoff) {
            g_rate_bucket.timestamps[new_count++] = g_rate_bucket.timestamps[i];
        }
    }
    g_rate_count = new_count;

    if (g_rate_count >= RATE_LIMIT) return 0; /* rate limited */
    g_rate_bucket.timestamps[g_rate_count++] = now;
    return 1; /* allowed */
}

/* ================================================================
 *  API key validation
 * ================================================================ */

static int check_auth(const char *auth_header) {
    if (!g_auth_key[0]) return 1; /* no auth required */
    if (!auth_header) return 0;

    /* Expect: "Bearer sk-..." */
    const char *p = auth_header;
    while (*p == ' ') p++;
    if (strncmp(p, "Bearer ", 7) != 0) return 0;
    p += 7;
    return strcmp(p, g_auth_key) == 0;
}

/* ================================================================
 *  Inference subprocess
 * ================================================================ */

static char *run_inference(const char *prompt, int max_tokens,
                            double temperature, int top_k, double top_p)
{
    /* Sandbox mode: return fake response */
    if (g_sandbox) {
        char *result = (char *)malloc(8192);
        if (!result) return NULL;

        char *escaped = json_escape("This is a sandbox test response for educational API verification purposes. In production, this would contain real model-generated text.");
        time_t now = time(NULL);
        snprintf(result, 8192,
            "{"
            "\"id\":\"chatcmpl-sandbox-%ld\","
            "\"object\":\"chat.completion\","
            "\"created\":%ld,"
            "\"model\":\"wubuwizard-qwen3.6-sandbox\","
            "\"choices\":[{"
              "\"index\":0,"
              "\"message\":{\"role\":\"assistant\",\"content\":%s},"
              "\"finish_reason\":\"stop\""
            "}],"
            "\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}"
            "}",
            (long)now, (long)now, escaped);
        free(escaped);
        return result;
    }

    /* Build command */
    char cmd[32768];
    snprintf(cmd, sizeof(cmd),
        "TEM=%.4f TOP_K=%d TOP_P=%.4f MOE=1 timeout %d %s %s %s %d 2>&1",
        temperature, top_k, top_p,
        max_tokens > 100 ? max_tokens + 30 : 120,
        g_infer_bin, g_model_path, prompt, max_tokens);

    /* Run and capture output */
    FILE *fp = popen(cmd, "r");
    if (!fp) return NULL;

    size_t cap = 65536, len = 0;
    char *output = (char *)malloc(cap);
    if (!output) { pclose(fp); return NULL; }
    output[0] = '\0';

    char line[4096];
    int in_generation = 0;
    char gen_text[65536] = "";
    size_t gen_len = 0;

    while (fgets(line, sizeof(line), fp)) {
        size_t llen = strlen(line);
        if (len + llen + 1 > cap) {
            cap *= 2;
            char *new_out = (char *)realloc(output, cap);
            if (!new_out) { free(output); pclose(fp); return NULL; }
            output = new_out;
        }
        memcpy(output + len, line, llen + 1);
        len += llen;

        /* Extract generated text: everything after "--- Decode ---" */
        if (strstr(line, "--- Decode") || strstr(line, "--- Chunked prefill")) {
            in_generation = 1;
            continue;
        }
        if (strstr(line, "=== Summary") || strstr(line, "=== PASS ===")) {
            in_generation = 0;
            continue;
        }
        if (in_generation && line[0] != '\n' && !strstr(line, "Prefill:")
            && !strstr(line, "T+") && !strstr(line, "=== PASS")) {
            size_t add = strlen(line);
            if (gen_len + add < sizeof(gen_text)) {
                memcpy(gen_text + gen_len, line, add);
                gen_len += add;
            }
        }
    }

    (void)pclose(fp);

    /* Build JSON response */
    if (gen_len == 0) {
        /* Fallback: use raw output */
        snprintf(gen_text, sizeof(gen_text), "%.3000s", output);
        gen_len = strlen(gen_text);
    }

    char *escaped = json_escape(gen_text);
    time_t now = time(NULL);
    char result[131072];
    snprintf(result, sizeof(result),
        "{"
        "\"id\":\"chatcmpl-%ld\","
        "\"object\":\"chat.completion\","
        "\"created\":%ld,"
        "\"model\":\"wubuwizard-qwen3.6\","
        "\"choices\":[{"
          "\"index\":0,"
          "\"message\":{\"role\":\"assistant\",\"content\":%s},"
          "\"finish_reason\":\"stop\""
        "}],"
        "\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}"
        "}",
        (long)now, (long)now, escaped);

    free(escaped);
    free(output);
    return strdup(result);
}

#ifdef _WIN32
#define sock_read(fd, buf, n)  recv((SOCKET)(fd), (char *)(buf), (int)(n), 0)
#define sock_write(fd, buf, n) send((SOCKET)(fd), (const char *)(buf), (int)(n), 0)
#define sock_close(fd)         closesocket((SOCKET)(fd))
#else
#define sock_read(fd, buf, n)  read((fd), (buf), (n))
#define sock_write(fd, buf, n) write((fd), (buf), (n))
#define sock_close(fd)         close(fd)
#endif

/* ================================================================
 *  HTTP response helpers
 * ================================================================ */

static void send_response(int fd, int status, const char *content_type,
                           const char *body, SSL *ssl)
{
    char header[4096];
    int len = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Server: wubuwizard-api/1.0 (educational research)\r\n"
        "X-Content-Type-Options: nosniff\r\n"
        "Connection: close\r\n"
        "\r\n",
        status,
        status == 200 ? "OK" :
        status == 400 ? "Bad Request" :
        status == 401 ? "Unauthorized" :
        status == 404 ? "Not Found" :
        status == 429 ? "Too Many Requests" :
        status == 500 ? "Internal Server Error" : "Unknown",
        content_type ? content_type : "application/json",
        body ? strlen(body) : 0);

    if (ssl) {
        SSL_write(ssl, header, len);
        if (body) SSL_write(ssl, body, strlen(body));
    } else {
        sock_write(fd, header, len);
        if (body) sock_write(fd, body, strlen(body));
    }
}

static void send_json(int fd, int status, const char *json, SSL *ssl) {
    send_response(fd, status, "application/json; charset=utf-8", json, ssl);
}

static void send_error(int fd, int status, const char *msg, SSL *ssl) {
    char err_body[4096];
    snprintf(err_body, sizeof(err_body),
        "{\"error\":{\"message\":%s,\"type\":\"error\",\"code\":%d}}",
        json_escape(msg), status);
    send_json(fd, status, err_body, ssl);
}

/* ================================================================
 * SSE token streaming (OpenAI-compatible chat.completion.chunk)
 * ================================================================ */

static void sse_write(int fd, SSL *ssl, const char *s)
{
    if (ssl) SSL_write(ssl, s, (int)strlen(s));
    else (void)!sock_write(fd, s, strlen(s));
}

/* Longest prefix of n bytes that ends on a complete UTF-8 codepoint.
 * llama.cpp buffers partial sequences the same way (server-common.cpp). */
static size_t utf8_complete_prefix(const char *s, size_t n)
{
    size_t i = 0;
    while (i < n) {
        unsigned char c = (unsigned char)s[i];
        int need = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 :
                   (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        if (i + (size_t)need > n) break;
        int valid = 1;
        for (int k = 1; k < need; k++)
            if (((unsigned char)s[i + k] & 0xC0) != 0x80) { valid = 0; break; }
        if (!valid) { need = 1; }
        i += (size_t)need;
    }
    return i;
}

static void sse_chunk(int fd, SSL *ssl, long id, time_t now,
                      const char *content, const char *finish_reason)
{
    char *escaped = content && content[0] ? json_escape(content) : strdup("\"\"");
    char buf[16384];
    int n = snprintf(buf, sizeof(buf),
        "data: {\"id\":\"chatcmpl-%ld\",\"object\":\"chat.completion.chunk\","
        "\"created\":%ld,\"model\":\"wubuwizard-stream\","
        "\"choices\":[{\"index\":0,\"delta\":{%s%s},\"finish_reason\":%s}]}\n\n",
        id, (long)now,
        content && content[0] ? "\"content\":" : "",
        escaped ? escaped : "\"\"",
        finish_reason ? finish_reason : "null");
    if (n > 0) sse_write(fd, ssl, buf);
    free(escaped);
}

static void send_sse_stream(int fd, const char *prompt, int max_tokens,
                            double temperature, int top_k, double top_p, SSL *ssl)
{
    static const char *hdr =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream; charset=utf-8\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "X-Accel-Buffering: no\r\n"
        "\r\n";
    sse_write(fd, ssl, hdr);
    long id = (long)getpid();
    time_t now = time(NULL);

    if (g_sandbox) {
        const char *words[] = { "This ", "is ", "a ", "streamed ", "sandbox ", "reply.\n" };
        for (size_t i = 0; i < sizeof(words)/sizeof(words[0]); i++) {
            sse_chunk(fd, ssl, id, now, words[i], NULL);
            usleep(20000);
        }
    } else {
        char cmd[32768];
        /* popen uses cmd.exe on Windows: VAR=value prefix is invalid there.
         * Set the engine env in this process so the child inherits it. */
        {
            char v[64];
#ifdef _WIN32
            _putenv("G4_STREAM_LINE=1");
            snprintf(v, sizeof(v), "TEM=%.4f", temperature); _putenv(v);
            snprintf(v, sizeof(v), "TOP_K=%d", top_k); _putenv(v);
            snprintf(v, sizeof(v), "TOP_P=%.4f", top_p); _putenv(v);
            _putenv("MOE=1");
#else
            setenv("G4_STREAM_LINE", "1", 1);
            snprintf(v, sizeof(v), "%.4f", temperature); setenv("TEM", v, 1);
            snprintf(v, sizeof(v), "%d", top_k); setenv("TOP_K", v, 1);
            snprintf(v, sizeof(v), "%.4f", top_p); setenv("TOP_P", v, 1);
            setenv("MOE", "1", 1);
#endif
        }
#ifdef _WIN32
        /* cmd.exe (used by MinGW popen) rejects a quoted exe path followed by
         * quoted args. Use a short (8.3) path so the exe token needs no quotes. */
        {
            char exe_short[MAX_PATH], model_short[MAX_PATH];
            DWORD ne = GetShortPathNameA(g_infer_bin, exe_short, MAX_PATH);
            DWORD nm = GetShortPathNameA(g_model_path, model_short, MAX_PATH);
            const char *exe = ne > 0 && ne < MAX_PATH ? exe_short : g_infer_bin;
            const char *model = nm > 0 && nm < MAX_PATH ? model_short : g_model_path;
            if (strchr(exe, ' ') || strchr(model, ' ')) {
                snprintf(cmd, sizeof(cmd), "call \"%s\" \"%s\" \"%s\" %d",
                         exe, model, prompt, max_tokens);
            } else {
                snprintf(cmd, sizeof(cmd), "%s \"%s\" \"%s\" %d",
                         exe, model, prompt, max_tokens);
            }
        }
#else
        snprintf(cmd, sizeof(cmd), "timeout %d %s %s %s %d",
                 max_tokens > 100 ? max_tokens + 30 : 120,
                 g_infer_bin, g_model_path, prompt, max_tokens);
#endif
        if (getenv("SSE_DEBUG")) fprintf(stderr, "[SSE] cmd: %s\n", cmd);
        FILE *fp = popen(cmd, "r");
        if (!fp) { sse_chunk(fd, ssl, id, now, "inference failed", "error"); goto done; }
        setvbuf(fp, NULL, _IONBF, 0); /* critical: emit each token line as it arrives */
        char line[4096];
        char pending[16];
        size_t npending = 0;
        int in_generation = 0;
        for (;;) {
#ifdef _WIN32
            /* llama.cpp sends ": ping" every 30 s of silence to keep proxies
             * from closing an idle SSE connection. */
            HANDLE ph = (HANDLE)_get_osfhandle(_fileno(fp));
            DWORD wr = WaitForSingleObject(ph, 30000);
            if (wr == WAIT_TIMEOUT) {
                sse_write(fd, ssl, ":\n\n");
                continue;
            }
            if (wr != WAIT_OBJECT_0) break;
#endif
            if (!fgets(line, sizeof(line), fp)) break;
            if (strstr(line, "<<<STREAM_START>>>")) { in_generation = 1; continue; }
            if (!in_generation) continue;
            if (line[0] == '\n' || line[0] == '\0') continue;
            if (strstr(line, "--- Stats") || strstr(line, "Total:") ||
                strstr(line, "Prefill:") || strstr(line, "Decode:")) break;
            char text[4096];
            snprintf(text, sizeof(text), "%s", line);
            size_t ntext = strlen(text);
            while (ntext && (text[ntext - 1] == '\n' || text[ntext - 1] == '\r'))
                text[--ntext] = '\0';
            if (!ntext) continue;
            char combined[4200], emit[4200];
            size_t total = npending + ntext;
            if (total >= sizeof(combined)) total = sizeof(combined) - 1;
            memcpy(combined, pending, npending);
            size_t copy = ntext;
            if (npending + copy > total) copy = total - npending;
            memcpy(combined + npending, text, copy);
            size_t complete = utf8_complete_prefix(combined, total);
            if (complete) {
                memcpy(emit, combined, complete);
                emit[complete] = '\0';
                sse_chunk(fd, ssl, id, now, emit, NULL);
            }
            npending = total - complete;
            memmove(pending, combined + complete, npending);
        }
        if (npending) {
            char emit[16];
            memcpy(emit, pending, npending);
            emit[npending] = '\0';
            sse_chunk(fd, ssl, id, now, emit, NULL);
        }
        (void)pclose(fp);
    }
done:
    sse_chunk(fd, ssl, id, now, NULL, "\"stop\"");
    sse_write(fd, ssl, "data: [DONE]\n\n");
}

/* In-process loopback test: exercises the real SSE writer over sockets. */
static int sse_selftest(void)
{
    const char *bin_env = getenv("INFER_BIN");
    const char *model_env = getenv("MODEL_PATH");
    if (getenv("SSE_REAL")) {
        if (bin_env) snprintf(g_infer_bin, sizeof(g_infer_bin), "%s", bin_env);
        if (model_env) snprintf(g_model_path, sizeof(g_model_path), "%s", model_env);
        g_sandbox = 0;
    } else {
        g_sandbox = 1;
    }
    int server, client;
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return 1;
    struct sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    server = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0 || bind(server, (struct sockaddr *)&a, sizeof(a)) < 0 ||
        listen(server, 1) < 0) return 1;
    int alen = sizeof(a);
    getsockname(server, (struct sockaddr *)&a, &alen);
    client = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (client < 0 || connect(client, (struct sockaddr *)&a, alen) < 0) return 1;
    server = (int)accept(server, NULL, NULL);
    if (server < 0) return 1;
#else
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) return 1;
    server = sv[0]; client = sv[1];
#endif
    send_sse_stream(server, "hi", 8, 1.0, 20, 0.95, NULL);
    sock_close(server);

    char buf[65536];
    size_t len = 0;
    ssize_t n;
    while (len + 1 < sizeof(buf) &&
           (n = sock_read(client, buf + len, sizeof(buf) - len - 1)) > 0) len += (size_t)n;
    buf[len] = '\0';
    sock_close(client);
#ifdef _WIN32
    WSACleanup();
#endif
    if (getenv("SSE_DUMP")) fwrite(buf, 1, len, stderr);
    int chunks = 0;
    for (const char *p = buf; (p = strstr(p, "data:")); p += 5) chunks++;
    int min_chunks = getenv("SSE_REAL") ? 3 : 6;
    int ok = strstr(buf, "text/event-stream") &&
             strstr(buf, "chat.completion.chunk") &&
             strstr(buf, "\"finish_reason\":\"stop\"") &&
             strstr(buf, "[DONE]") &&
             !strstr(buf, "inference failed") &&
             chunks >= min_chunks;
    if (!ok) {
        fprintf(stderr, "sse_selftest: FAIL chunks=%d len=%zu\n%.400s\n", chunks, len, buf);
        return 1;
    }
    printf("test_sse: PASS (%d stream events)\n", chunks);
    return 0;
}

/* ================================================================
 *  Request parsing
 * ================================================================ */

typedef struct {
    char method[16];
    char path[1024];
    char headers[MAX_HEADERS];
    char body[MAX_BODY];
    size_t body_len;
    int content_length;
} http_request_t;

static int parse_request(int fd, http_request_t *req, SSL *ssl) {
    memset(req, 0, sizeof(*req));

    /* Read request line + headers */
    char buf[MAX_HEADERS + MAX_BODY];
    size_t total = 0;

    while (total < sizeof(buf) - 1) {
        int n;
        if (ssl)
            n = SSL_read(ssl, buf + total, (int)(sizeof(buf) - total - 1));
        else
            n = (int)sock_read(fd, buf + total, sizeof(buf) - total - 1);

        if (n <= 0) break;
        total += (size_t)n;
        buf[total] = '\0';

        /* Check for end of headers */
        if (strstr(buf, "\r\n\r\n")) break;
    }

    if (total == 0) return 0;
    buf[total] = '\0';

    /* Parse first line */
    char *line_end = strchr(buf, '\r');
    if (!line_end) line_end = strchr(buf, '\n');
    if (!line_end) return 0;
    *line_end = '\0';

    char method[16], path[1024], version[16];
    if (sscanf(buf, "%15s %1023s %15s", method, path, version) < 2) return 0;
    strcpy(req->method, method);
    strcpy(req->path, path);

    /* Parse headers */
    char *hdr_start = line_end + 1;
    while (*hdr_start == '\n' || *hdr_start == '\r') hdr_start++;

    char *hdr_end = strstr(hdr_start, "\r\n\r\n");
    if (!hdr_end) hdr_end = strstr(hdr_start, "\n\n");
    if (!hdr_end) return 0;

    size_t hdr_len = (size_t)(hdr_end - hdr_start);
    if (hdr_len >= sizeof(req->headers)) hdr_len = sizeof(req->headers) - 1;
    memcpy(req->headers, hdr_start, hdr_len);
    req->headers[hdr_len] = '\0';

    /* Find Content-Length */
    char *cl = strstr(req->headers, "Content-Length:");
    if (!cl) cl = strstr(req->headers, "content-length:");
    if (!cl) cl = strstr(req->headers, "Content-length:");
    if (cl) {
        cl += 15; /* skip past "Content-Length:" */
        while (*cl == ' ') cl++;
        req->content_length = atoi(cl);
        if (req->content_length > MAX_BODY) req->content_length = MAX_BODY;

        /* Body is after headers */
        char *body_start = hdr_end;
        while (*body_start == '\r' || *body_start == '\n') body_start++;
        size_t body_avail = total - (size_t)(body_start - buf);
        req->body_len = (size_t)(req->content_length < (int)body_avail ?
                         req->content_length : (int)body_avail);
        if (req->body_len > 0) {
            memcpy(req->body, body_start, req->body_len);
            req->body[req->body_len] = '\0';
        }
    }

    return 1;
}

/* Find header value (case-insensitive) */
static const char *get_header(http_request_t *req, const char *name) {
    if (!req->headers[0]) return NULL;
    char *p = req->headers;
    while (*p) {
        if (strncasecmp(p, name, strlen(name)) == 0) {
            p += strlen(name);
            while (*p == ':' || *p == ' ') p++;
            return p;
        }
        /* Skip to next line */
        while (*p && *p != '\n') p++;
        if (*p == '\n') p++;
    }
    return NULL;
}

/* ================================================================
 *  Router
 * ================================================================ */

static void handle_request(int fd, http_request_t *req, SSL *ssl) {
    /* Health check */
    if (strcmp(req->path, "/health") == 0 && strcmp(req->method, "GET") == 0) {
        send_json(fd, 200, "{\"status\":\"ok\",\"service\":\"wubuwizard-inference-api\",\"version\":\"1.0.0\",\"educational\":true,\"notice\":\"For educational/research use only. No liability assumed.\"}", ssl);
        return;
    }

    /* List models */
    if (strcmp(req->path, "/v1/models") == 0 && strcmp(req->method, "GET") == 0) {
        char models[4096];
        char *name = json_escape("wubuwizard-qwen3.6");
        snprintf(models, sizeof(models),
            "{\"object\":\"list\",\"data\":[{\"id\":%s,\"object\":\"model\",\"created\":1715097600,\"owned_by\":\"wubuwizard\"}]}",
            name);
        free(name);
        send_json(fd, 200, models, ssl);
        return;
    }

    /* CORS preflight */
    if (strcmp(req->method, "OPTIONS") == 0) {
        send_response(fd, 204, "text/plain", "", ssl);
        return;
    }

    /* Auth check */
    const char *auth = get_header(req, "Authorization");
    if (!check_auth(auth)) {
        send_error(fd, 401, "Invalid or missing API key. Set --auth or provide Authorization: Bearer <key>", ssl);
        return;
    }

    /* Rate limit check */
    if (!check_rate_limit()) {
        send_error(fd, 429, "Rate limit exceeded. Max " STR(RATE_LIMIT) " requests per " STR(RATE_WINDOW) " seconds.", ssl);
        return;
    }

    /* Chat completions */
    if ((strcmp(req->path, "/v1/chat/completions") == 0 ||
         strcmp(req->path, "/v1/completions") == 0) &&
        strcmp(req->method, "POST") == 0)
    {
        if (req->body_len == 0) {
            send_error(fd, 400, "Empty request body", ssl);
            return;
        }

        /* Parse JSON body for prompt/messages */
        /* Simple extraction: find "content" field for chat, or "prompt" for completions */
        char prompt[65536] = "";
        int max_tokens = 256;
        double temperature = 1.0;
        int top_k = 20;
        double top_p = 0.95;

        const char *body = req->body;

        /* Extract prompt from messages array or prompt field */
        char *content_ptr = strstr(body, "\"content\"");
        char *prompt_ptr = strstr(body, "\"prompt\"");
        char *msgs_ptr = strstr(body, "\"messages\"");

        if (content_ptr && msgs_ptr && content_ptr > msgs_ptr) {
            /* Extract content value */
            char *val_start = strchr(content_ptr, ':');
            if (val_start) {
                val_start++;
                while (*val_start == ' ' || *val_start == '"') val_start++;
                char *val_end = strchr(val_start, '"');
                if (val_end) {
                    size_t plen = (size_t)(val_end - val_start);
                    if (plen > sizeof(prompt) - 1) plen = sizeof(prompt) - 1;
                    memcpy(prompt, val_start, plen);
                    prompt[plen] = '\0';
                }
            }
        } else if (prompt_ptr) {
            char *val_start = strchr(prompt_ptr, ':');
            if (val_start) {
                val_start++;
                while (*val_start == ' ' || *val_start == '"') val_start++;
                char *val_end = strchr(val_start, '"');
                if (val_end) {
                    size_t plen = (size_t)(val_end - val_start);
                    if (plen > sizeof(prompt) - 1) plen = sizeof(prompt) - 1;
                    memcpy(prompt, val_start, plen);
                    prompt[plen] = '\0';
                }
            }
        }

        /* Extract max_tokens, temperature, top_k, top_p */
        char *mt = strstr(body, "\"max_tokens\"");
        if (mt) { char *v = strchr(mt, ':'); if (v) max_tokens = atoi(v + 1); }
        char *tmp = strstr(body, "\"temperature\"");
        if (tmp) { char *v = strchr(tmp, ':'); if (v) temperature = atof(v + 1); }
        char *tk = strstr(body, "\"top_k\"");
        if (tk) { char *v = strchr(tk, ':'); if (v) top_k = atoi(v + 1); }
        char *tp = strstr(body, "\"top_p\"");
        if (tp) { char *v = strchr(tp, ':'); if (v) top_p = atof(v + 1); }

        if (prompt[0] == '\0') {
            /* Fallback: use whole body as debug info */
            snprintf(prompt, sizeof(prompt), "hello");
        }

        /* stream:true -> SSE token chunks instead of one buffered JSON response */
        char *stream_ptr = strstr(body, "\"stream\"");
        int stream = stream_ptr && strstr(stream_ptr, "true") != NULL;
        if (stream) {
            send_sse_stream(fd, prompt, max_tokens, temperature, top_k, top_p, ssl);
            return;
        }

        /* Run inference */
        char *result = run_inference(prompt, max_tokens, temperature, top_k, top_p);
        if (!result) {
            send_error(fd, 500, "Inference failed — check model path and infer binary", ssl);
            return;
        }

        send_json(fd, 200, result, ssl);
        free(result);
        return;
    }

    /* 404 for everything else */
    send_error(fd, 404, "Not found. Available endpoints: GET /health, GET /v1/models, POST /v1/chat/completions, POST /v1/completions", ssl);
}

/* ================================================================
 *  Client handler thread (simplified: fork per connection)
 * ================================================================ */

static void handle_client(int fd, SSL *ssl) {
    http_request_t req;
    if (parse_request(fd, &req, ssl)) {
        handle_request(fd, &req, ssl);
    }
    if (ssl) {
        SSL_shutdown(ssl);
        SSL_free(ssl);
    }
    sock_close(fd);
}

/* ================================================================
 *  Main
 * ================================================================ */

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "OpenAI-compatible API server for wubuwizard inference engine.\n");
    fprintf(stderr, "EDUCATIONAL / RESEARCH USE ONLY.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --port N        Port to listen on (default: %d)\n", DEFAULT_PORT);
    fprintf(stderr, "  --sandbox       Fake responses (no GPU needed)\n");
    fprintf(stderr, "  --auth KEY      Require API key (Authorization: Bearer <key>)\n");
    fprintf(stderr, "  --tls CERT KEY  Enable TLS with certificate + key\n");
    fprintf(stderr, "  --model PATH    Model GGUF path\n");
    fprintf(stderr, "  --bin PATH      Infer binary path\n");
    fprintf(stderr, "  --help          This message\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Environment:\n");
    fprintf(stderr, "  INFER_BIN=./infer_text_gpu\n");
    fprintf(stderr, "  MODEL_PATH=/path/to/model.gguf\n");
    fprintf(stderr, "  API_PORT=8080\n");
    fprintf(stderr, "  API_AUTH_KEY=sk-...\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Security Notice: This server is for local/trusted-network use.\n");
    fprintf(stderr, "Do not expose to the internet without proper auth, TLS, and firewall.\n");
    fprintf(stderr, "Operators assume all liability for deployment.\n");
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--sse-selftest") == 0)
        return sse_selftest();
#ifdef _WIN32
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
        fprintf(stderr, "WSAStartup failed\n");
        return 1;
    }
#endif
    wubu_disable_core_dumps();
    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc)
            g_port = atoi(argv[++i]);
        else if (strcmp(argv[i], "--sandbox") == 0)
            g_sandbox = 1;
        else if (strcmp(argv[i], "--auth") == 0 && i + 1 < argc)
            snprintf(g_auth_key, sizeof(g_auth_key), "%s", argv[++i]);
        else if (strcmp(argv[i], "--tls") == 0 && i + 2 < argc) {
            g_use_tls = 1;
            snprintf(g_tls_cert, sizeof(g_tls_cert), "%s", argv[++i]);
            snprintf(g_tls_key, sizeof(g_tls_key), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            snprintf(g_model_path, sizeof(g_model_path), "%s", argv[++i]);
        else if (strcmp(argv[i], "--bin") == 0 && i + 1 < argc)
            snprintf(g_infer_bin, sizeof(g_infer_bin), "%s", argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    /* Environment overrides */
    char *env;
    if ((env = getenv("API_PORT"))) g_port = atoi(env);
    if ((env = getenv("API_AUTH_KEY"))) snprintf(g_auth_key, sizeof(g_auth_key), "%s", env);
    if ((env = getenv("INFER_BIN"))) snprintf(g_infer_bin, sizeof(g_infer_bin), "%s", env);
    if ((env = getenv("MODEL_PATH"))) snprintf(g_model_path, sizeof(g_model_path), "%s", env);

    /* Signal handlers */
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
#ifndef _WIN32
    signal(SIGCHLD, SIG_IGN); /* reap child processes */
#endif

    /* Initialize OpenSSL for TLS */
    SSL_CTX *ssl_ctx = NULL;
    if (g_use_tls) {
        SSL_load_error_strings();
        OPENSSL_init_ssl(0, NULL);
        ssl_ctx = SSL_CTX_new(TLS_server_method());
        if (!ssl_ctx) {
            fprintf(stderr, "Failed to create SSL context\n");
            return 1;
        }
        if (SSL_CTX_use_certificate_file(ssl_ctx, g_tls_cert, SSL_FILETYPE_PEM) <= 0) {
            fprintf(stderr, "Failed to load cert: %s\n", g_tls_cert);
            return 1;
        }
        if (SSL_CTX_use_PrivateKey_file(ssl_ctx, g_tls_key, SSL_FILETYPE_PEM) <= 0) {
            fprintf(stderr, "Failed to load key: %s\n", g_tls_key);
            return 1;
        }
    }

    /* Create socket */
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return 1; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(g_port);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        sock_close(server_fd);
        return 1;
    }

    if (listen(server_fd, 5) < 0) {
        perror("listen");
        sock_close(server_fd);
        return 1;
    }

    print_banner();

    /* Main accept loop */
    while (g_running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            continue;
        }

#ifndef _WIN32
        /* Fork to handle client (simple concurrency — each request gets own process) */
        pid_t pid = fork();
        if (pid == 0) {
            /* Child process */
            sock_close(server_fd);
            SSL *ssl = NULL;
            if (g_use_tls) {
                ssl = SSL_new(ssl_ctx);
                if (ssl) SSL_set_fd(ssl, client_fd);
            }
            handle_client(client_fd, ssl);
            _exit(0);
        } else if (pid > 0) {
            sock_close(client_fd);
        } else {
            perror("fork");
            sock_close(client_fd);
        }
#else
        /* Windows has no fork: handle sequentially. SSE still streams per
         * connection because sse_write flushes each generated line. */
        {
            SSL *ssl = NULL;
            if (g_use_tls) {
                ssl = SSL_new(ssl_ctx);
                if (ssl) SSL_set_fd(ssl, client_fd);
            }
            handle_client(client_fd, ssl);
            sock_close(client_fd);
        }
#endif
    }

    sock_close(server_fd);
    if (ssl_ctx) SSL_CTX_free(ssl_ctx);
#ifdef _WIN32
    WSACleanup();
#endif
    printf("[server] Shutdown complete.\n");
    return 0;
}
