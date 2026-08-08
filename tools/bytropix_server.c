/**
 * bytropix_server.c — Local inference API server with persistent process support.
 *
 * Two modes:
 *   Normal (default):  fork() + popen() per request (model reload ~80s each)
 *   Persistent:        One gen_text_cpu --persist process, binary protocol,
 *                      threads with mutex, KV cache across requests.
 *
 * Endpoints:
 *   POST /v1/chat/completions  — Chat completions (OpenAI-compatible)
 *   POST /v1/completions       — Text completions
 *   GET  /v1/models            — List available models
 *   GET  /health               — Health check
 *
 * Build:
 *   gcc -O2 -g -Wall -o bytropix_server tools/bytropix_server.c \
 *       -ljson-c -lm -lpthread
 *
 * Usage:
 *   ./bytropix_server --port 8001 --persist
 *   ./bytropix_server --port 8001 --bin ./gen_text_cpu --model ~/models/foo.gguf
 *
 * Environment:
 *   INFER_BIN   — path to gen_text_cpu (default: ./gen_text_cpu)
 *   MODEL_PATH  — path to GGUF model (default: ~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf)
 *   API_PORT    — port to listen on (default: 8001)
 *   OMP_THREADS — OpenMP threads (default: 4)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <json-c/json.h>

/* ================================================================
 *  Constants
 * ================================================================ */

#define DEFAULT_PORT      8001
#define MAX_HEADERS       16384
#define MAX_BODY          1048576
#define MAX_PATH          4096
#define MAX_PROMPT        65536
#define MAX_RESULT        131072
#define BACKLOG           16
#define PERSIST_BUF       131072
#define MARKER            "---BINARY---\n"
#define MARKER_LEN        13

/* ================================================================
 *  Config
 * ================================================================ */

static int      g_port = DEFAULT_PORT;
static char     g_infer_bin[1024] = "./gen_text_cpu";
static char     g_model_path[1024] = "";
static int      g_persist_mode = 0;
static int      g_omp_threads = 4;
static volatile int g_running = 1;

/* ================================================================
 *  Persistent process state
 * ================================================================ */

typedef struct {
    pid_t   pid;
    int     stdin_fd;
    int     stdout_fd;
    int     stderr_fd;
    pthread_mutex_t lock;
    /* Read buffer for stdout */
    char    buf[PERSIST_BUF];
    size_t  buf_len;
    /* Config */
    char    bin_path[1024];
    char    model_path[1024];
    int     omp_threads;
} persist_state_t;

static persist_state_t g_persist = {0};

/* ================================================================
 *  Signal handler
 * ================================================================ */

static void handle_signal(int sig) {
    (void)sig;
    g_running = 0;
    fprintf(stderr, "[server] Shutting down...\n");
}

/* ================================================================
 *  Persistent process management
 * ================================================================ */

static bool persist_start(void) {
    /* Build argv */
    char *argv[] = { g_persist.bin_path, "--persist", NULL };

    /* Pipes: parent_stdin -> child_stdout, child_stdin -> parent_stdout */
    int child_stdin[2];  /* parent writes, child reads */
    int child_stdout[2]; /* child writes, parent reads */
    int child_stderr[2]; /* child writes, parent reads */

    if (pipe(child_stdin) < 0 || pipe(child_stdout) < 0 || pipe(child_stderr) < 0) {
        perror("pipe");
        return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return false;
    }

    if (pid == 0) {
        /* Child process */
        close(child_stdin[1]);   /* Close write end */
        close(child_stdout[0]);  /* Close read end */
        close(child_stderr[0]);

        dup2(child_stdin[0], STDIN_FILENO);
        dup2(child_stdout[1], STDOUT_FILENO);
        dup2(child_stderr[1], STDERR_FILENO);

        close(child_stdin[0]);
        close(child_stdout[1]);
        close(child_stderr[1]);

        /* Set environment */
        char omp_buf[16];
        snprintf(omp_buf, sizeof(omp_buf), "%d", g_persist.omp_threads);
        setenv("MODEL", g_persist.model_path, 1);
        setenv("OMP_NUM_THREADS", omp_buf, 1);
        setenv("MOE", "1", 1);
        setenv("CHAT", "1", 1);
        setenv("TEMP", "0.7", 1);
        setenv("TOP_K", "40", 1);

        execvp(argv[0], argv);
        perror("execvp");
        _exit(1);
    }

    /* Parent process */
    close(child_stdin[0]);   /* Close read end */
    close(child_stdout[1]);  /* Close write end */
    close(child_stderr[1]);

    g_persist.pid = pid;
    g_persist.stdin_fd = child_stdin[1];
    g_persist.stdout_fd = child_stdout[0];
    g_persist.stderr_fd = child_stderr[0];
    g_persist.buf_len = 0;

    /* Wait for "[persist] ready" on stderr */
    char ready_buf[256] = "";
    size_t ready_pos = 0;
    while (ready_pos < sizeof(ready_buf) - 1) {
        char ch;
        int n = (int)read(g_persist.stderr_fd, &ch, 1);
        if (n <= 0) {
            fprintf(stderr, "[persist] child died before ready\n");
            return false;
        }
        ready_buf[ready_pos++] = ch;
        ready_buf[ready_pos] = '\0';
        if (ch == '\n') {
            fprintf(stderr, "[persist] %s", ready_buf);
            if (strstr(ready_buf, "[persist] ready")) {
                fprintf(stderr, "[persist] model loaded, ready for requests\n");
                return true;
            }
            ready_pos = 0;
        }
    }
    return false;
}

static void persist_stop(void) {
    if (g_persist.pid > 0) {
        close(g_persist.stdin_fd);
        close(g_persist.stdout_fd);
        close(g_persist.stderr_fd);
        int status;
        waitpid(g_persist.pid, &status, WNOHANG);
        kill(g_persist.pid, SIGTERM);
        waitpid(g_persist.pid, &status, 0);
        g_persist.pid = 0;
    }
}

static bool persist_is_alive(void) {
    if (g_persist.pid <= 0) return false;
    int status;
    pid_t r = waitpid(g_persist.pid, &status, WNOHANG);
    if (r == g_persist.pid) return false; /* Exited */
    if (r < 0 && errno != ECHILD) return false;
    return true;
}

/* Read and discard model init noise from stdout.
 * Called once after persist_start. Reads until first MARKER or EOF. */
static void persist_flush_init_stdout(void) {
    char buf[4096];
    ssize_t n;
    /* Read until we have enough to check for marker */
    size_t total = 0;
    while (1) {
        n = read(g_persist.stdout_fd, buf + total, sizeof(buf) - total - 1);
        if (n <= 0) break;
        total += (size_t)n;
        buf[total] = '\0';
        /* Check for marker */
        char *marker = strstr(buf, MARKER);
        if (marker) {
            /* Keep whatever is after the marker */
            size_t after = (size_t)(marker + MARKER_LEN - buf);
            size_t remain = total - after;
            if (remain > 0) {
                memmove(g_persist.buf, marker + MARKER_LEN, remain);
                g_persist.buf_len = remain;
            }
            return;
        }
        if (total >= sizeof(buf) - 1) break;
    }
}

/* Read one response from the persistent process.
 * Returns allocated string on success, NULL on failure. */
static char *persist_read_response(int timeout_sec) {
    char *marker_ptr;
    size_t marker_offset;

    /* Search for marker in existing buffer */
    marker_ptr = (char *)memmem(g_persist.buf, g_persist.buf_len, MARKER, MARKER_LEN);
    if (!marker_ptr) {
        /* Need to read more */
        fd_set rfds;
        struct timeval tv;
        time_t deadline = time(NULL) + timeout_sec;

        while (time(NULL) < deadline) {
            FD_ZERO(&rfds);
            FD_SET(g_persist.stdout_fd, &rfds);
            tv.tv_sec = 1;
            tv.tv_usec = 0;

            int ret = select(g_persist.stdout_fd + 1, &rfds, NULL, NULL, &tv);
            if (ret < 0) return NULL;
            if (ret == 0) continue; /* Timeout, try again */

            ssize_t n = read(g_persist.stdout_fd,
                             g_persist.buf + g_persist.buf_len,
                             PERSIST_BUF - g_persist.buf_len - 1);
            if (n <= 0) return NULL;
            g_persist.buf_len += (size_t)n;
            g_persist.buf[g_persist.buf_len] = '\0';

            marker_ptr = (char *)memmem(g_persist.buf, g_persist.buf_len,
                                         MARKER, MARKER_LEN);
            if (marker_ptr) break;
        }
        if (!marker_ptr) return NULL; /* Timeout */
    }

    marker_offset = (size_t)(marker_ptr - g_persist.buf);
    size_t after_marker = g_persist.buf_len - marker_offset - MARKER_LEN;

    /* Need at least 8 bytes after marker (result_len + tokens) */
    if (after_marker < 8) return NULL;

    uint32_t result_len;
    memcpy(&result_len, g_persist.buf + marker_offset + MARKER_LEN, 4);
    size_t needed = (size_t)result_len + 4 + 4; /* len + text + tokens */
    if (after_marker < needed) return NULL;

    uint32_t tokens_gen;
    memcpy(&tokens_gen,
           g_persist.buf + marker_offset + MARKER_LEN + 4 + result_len, 4);

    /* Extract result text */
    char *result = (char *)malloc((size_t)result_len + 1);
    if (!result) return NULL;
    memcpy(result,
           g_persist.buf + marker_offset + MARKER_LEN + 4,
           result_len);
    result[result_len] = '\0';

    /* Consume from buffer */
    size_t consumed = marker_offset + MARKER_LEN + needed;
    if (consumed < g_persist.buf_len) {
        memmove(g_persist.buf, g_persist.buf + consumed,
                g_persist.buf_len - consumed);
        g_persist.buf_len -= consumed;
    } else {
        g_persist.buf_len = 0;
    }

    (void)tokens_gen; /* Available for future use */
    return result;
}

/* Send a request to the persistent process.
 * Returns true on success. */
static bool persist_send_request(const char *prompt, int max_tokens, int top_k) {
    uint32_t text_len = (uint32_t)strlen(prompt);

    /* Pack: text_len (4) + prompt bytes + max_tokens (4) + top_k (4) */
    uint8_t header[4];
    memcpy(header, &text_len, 4);

    uint8_t params[8];
    uint32_t mt = (uint32_t)max_tokens;
    uint32_t tk = (uint32_t)top_k;
    memcpy(params, &mt, 4);
    memcpy(params + 4, &tk, 4);

    struct iovec iov[3];
    iov[0].iov_base = header;
    iov[0].iov_len = 4;
    iov[1].iov_base = (void *)prompt;
    iov[1].iov_len = text_len;
    iov[2].iov_base = params;
    iov[2].iov_len = 8;

    size_t total = 4 + text_len + 8;
    size_t written = 0;
    for (int i = 0; i < 3; i++) {
        ssize_t n = write(g_persist.stdin_fd, iov[i].iov_base, iov[i].iov_len);
        if (n <= 0) return false;
        written += (size_t)n;
    }

    return written == total;
}

/* Forward declarations */
static char *json_escape(const char *s);
static void send_response(int fd, int status, const char *content_type,
                           const char *body);
static void send_json_resp(int fd, int status, const char *json);
static void send_error_resp(int fd, int status, const char *msg);

/* Run inference through the persistent process.
 * Returns allocated string (JSON response) or NULL on failure.
 * Caller MUST hold g_persist.lock. */
static char *persist_run_inference(const char *prompt, int max_tokens,
                                    int top_k) {
    /* Check process alive */
    if (!persist_is_alive()) {
        fprintf(stderr, "[persist] process died, restarting...\n");
        persist_stop();
        if (!persist_start()) return NULL;
        persist_flush_init_stdout();
    }

    if (!persist_send_request(prompt, max_tokens, top_k)) {
        fprintf(stderr, "[persist] write failed, restarting...\n");
        persist_stop();
        if (!persist_start()) return NULL;
        persist_flush_init_stdout();
        /* Retry once */
        if (!persist_send_request(prompt, max_tokens, top_k)) return NULL;
    }

    char *result_text = persist_read_response(300);
    if (!result_text) {
        fprintf(stderr, "[persist] read failed\n");
        return NULL;
    }

    /* Build JSON response */
    char *escaped = json_escape(result_text);
    free(result_text);
    if (!escaped) return NULL;

    time_t now = time(NULL);
    char *json_out = (char *)malloc(MAX_RESULT);
    if (!json_out) { free(escaped); return NULL; }

    snprintf(json_out, MAX_RESULT,
        "{"
        "\"id\":\"chatcmpl-%ld\","
        "\"object\":\"chat.completion\","
        "\"created\":%ld,"
        "\"model\":\"bytropix-qwen3.6-persistent\","
        "\"choices\":[{"
          "\"index\":0,"
          "\"message\":{\"role\":\"assistant\",\"content\":%s},"
          "\"finish_reason\":\"stop\""
        "}],"
        "\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}"
        "}",
        (long)now, (long)now, escaped);

    free(escaped);
    return json_out;
}


/* ================================================================
 *  Normal (non-persist) inference via popen
 * ================================================================ */

static char *run_inference(const char *prompt, int max_tokens,
                            int top_k) {
    char cmd[32768];
    snprintf(cmd, sizeof(cmd),
        "MODEL=%s OMP_NUM_THREADS=%d CHAT=1 MOE=1 timeout %d %s \"%.30000s\" %d %d 2>&1",
        g_model_path, g_omp_threads,
        max_tokens > 100 ? max_tokens + 30 : 120,
        g_infer_bin, prompt, max_tokens, top_k);

    FILE *fp = popen(cmd, "r");
    if (!fp) return NULL;

    size_t cap = 65536, len = 0;
    char *output = (char *)malloc(cap);
    if (!output) { pclose(fp); return NULL; }
    output[0] = '\0';

    char line[4096];
    int in_gen = 0;
    char gen_text[MAX_RESULT] = "";
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

        /* Extract generated text */
        if (strstr(line, "Input:") && !in_gen) {
            in_gen = 1;
            continue;
        }
        if (strstr(line, "--- Stats ---")) {
            in_gen = 0;
            continue;
        }
        if (in_gen && line[0] != '\n' && line[0] != '\r') {
            size_t add = strlen(line);
            if (gen_len + add < sizeof(gen_text)) {
                memcpy(gen_text + gen_len, line, add);
                gen_len += add;
            }
        }
    }

    (void)pclose(fp);

    if (gen_len == 0) {
        snprintf(gen_text, sizeof(gen_text), "%.3000s", output);
    }

    /* Strip trailing newlines */
    while (gen_len > 0 && (gen_text[gen_len-1] == '\n' || gen_text[gen_len-1] == '\r'))
        gen_text[--gen_len] = '\0';

    char *escaped = json_escape(gen_text);
    free(output);
    if (!escaped) return NULL;

    time_t now = time(NULL);
    char result[MAX_RESULT];
    snprintf(result, sizeof(result),
        "{"
        "\"id\":\"chatcmpl-%ld\","
        "\"object\":\"chat.completion\","
        "\"created\":%ld,"
        "\"model\":\"bytropix-qwen3.6\","
        "\"choices\":[{"
          "\"index\":0,"
          "\"message\":{\"role\":\"assistant\",\"content\":%s},"
          "\"finish_reason\":\"stop\""
        "}],"
        "\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}"
        "}",
        (long)now, (long)now, escaped);

    free(escaped);
    return strdup(result);
}


/* ================================================================
 *  JSON helpers
 * ================================================================ */



/* Escape a string for JSON. Caller must free. */
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

/* Extract a string value from JSON by key path (simple: one-level lookup).
 * Returns allocated string or NULL. */
static char *json_extract_str(const char *json_body, const char *key) {
    json_object *root = json_tokener_parse(json_body);
    if (!root) return NULL;

    json_object *val;
    if (json_object_object_get_ex(root, key, &val) &&
        json_object_is_type(val, json_type_string)) {
        const char *s = json_object_get_string(val);
        char *result = strdup(s ? s : "");
        json_object_put(root);
        return result;
    }

    /* Try messages[last].content */
    if (strcmp(key, "content") == 0) {
        json_object *msgs;
        if (json_object_object_get_ex(root, "messages", &msgs) &&
            json_object_is_type(msgs, json_type_array)) {
            int len = json_object_array_length(msgs);
            if (len > 0) {
                json_object *last = json_object_array_get_idx(msgs, len - 1);
                if (last && json_object_object_get_ex(last, "content", &val) &&
                    json_object_is_type(val, json_type_string)) {
                    const char *s = json_object_get_string(val);
                    char *result = strdup(s ? s : "");
                    json_object_put(root);
                    return result;
                }
            }
        }
    }

    json_object_put(root);
    return NULL;
}

static int json_extract_int(const char *json_body, const char *key, int def) {
    json_object *root = json_tokener_parse(json_body);
    if (!root) return def;

    json_object *val;
    if (json_object_object_get_ex(root, key, &val) &&
        json_object_is_type(val, json_type_int)) {
        int result = json_object_get_int(val);
        json_object_put(root);
        return result;
    }

    json_object_put(root);
    return def;
}

static double json_extract_double(const char *json_body, const char *key, double def) {
    json_object *root = json_tokener_parse(json_body);
    if (!root) return def;

    json_object *val;
    if (json_object_object_get_ex(root, key, &val) &&
        (json_object_is_type(val, json_type_double) ||
         json_object_is_type(val, json_type_int))) {
        double result = json_object_get_double(val);
        json_object_put(root);
        return result;
    }

    json_object_put(root);
    return def;
}


/* ================================================================
 *  HTTP response helpers
 * ================================================================ */

static void send_response(int fd, int status, const char *content_type,
                           const char *body) {
    char header[4096];
    int n = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Server: bytropix-server/1.0\r\n"
        "Connection: close\r\n"
        "\r\n",
        status,
        status == 200 ? "OK" :
        status == 400 ? "Bad Request" :
        status == 404 ? "Not Found" :
        status == 500 ? "Internal Server Error" : "Unknown",
        content_type ? content_type : "application/json",
        body ? strlen(body) : 0);

    write(fd, header, (size_t)n);
    if (body) write(fd, body, strlen(body));
}

static void send_json_resp(int fd, int status, const char *json) {
    send_response(fd, status, "application/json; charset=utf-8", json);
}

static void send_error_resp(int fd, int status, const char *msg) {
    char body[8192];
    char *escaped = json_escape(msg);
    snprintf(body, sizeof(body),
        "{\"error\":{\"message\":%s,\"type\":\"error\",\"code\":%d}}",
        escaped ? escaped : "\"\"", status);
    free(escaped);
    send_json_resp(fd, status, body);
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

static int parse_request(int fd, http_request_t *req) {
    memset(req, 0, sizeof(*req));

    char buf[MAX_HEADERS + MAX_BODY];
    size_t total = 0;

    while (total < sizeof(buf) - 1) {
        int n = (int)read(fd, buf + total, sizeof(buf) - total - 1);
        if (n <= 0) break;
        total += (size_t)n;
        buf[total] = '\0';
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

    /* Find headers */
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
    if (cl) {
        cl += 15;
        while (*cl == ' ') cl++;
        req->content_length = atoi(cl);
        if (req->content_length > MAX_BODY) req->content_length = MAX_BODY;

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


/* ================================================================
 *  Route handler
 * ================================================================ */

/* Thread-safe inference: either persistent (mutex) or normal (fork). */
static char *do_inference(const char *prompt, int max_tokens, int top_k) {
    if (g_persist_mode) {
        pthread_mutex_lock(&g_persist.lock);
        char *result = persist_run_inference(prompt, max_tokens, top_k);
        pthread_mutex_unlock(&g_persist.lock);
        return result;
    }
    return run_inference(prompt, max_tokens, top_k);
}

static void handle_request(int fd, http_request_t *req) {
    /* Health check */
    if (strcmp(req->path, "/health") == 0 && strcmp(req->method, "GET") == 0) {
        char health[4096];
        snprintf(health, sizeof(health),
            "{"
            "\"status\":\"ok\","
            "\"service\":\"bytropix-inference-api\","
            "\"persist_mode\":%s,"
            "\"model\":\"%s\","
            "\"binary\":\"%s\""
            "}",
            g_persist_mode ? "true" : "false",
            g_model_path, g_infer_bin);
        send_json_resp(fd, 200, health);
        return;
    }

    /* List models */
    if (strcmp(req->path, "/v1/models") == 0 && strcmp(req->method, "GET") == 0) {
        char models[4096];
        char *name = json_escape(
            g_persist_mode ? "bytropix-qwen3.6-persistent" : "bytropix-qwen3.6");
        snprintf(models, sizeof(models),
            "{\"object\":\"list\",\"data\":[{\"id\":%s,\"object\":\"model\","
            "\"created\":1715097600,\"owned_by\":\"bytropix\"}]}",
            name);
        free(name);
        send_json_resp(fd, 200, models);
        return;
    }

    /* CORS preflight */
    if (strcmp(req->method, "OPTIONS") == 0) {
        send_response(fd, 204, "text/plain", "");
        return;
    }

    /* Chat completions / text completions */
    if ((strcmp(req->path, "/v1/chat/completions") == 0 ||
         strcmp(req->path, "/v1/completions") == 0) &&
        strcmp(req->method, "POST") == 0) {
        if (req->body_len == 0) {
            send_error_resp(fd, 400, "Empty request body");
            return;
        }

        /* Parse with json-c */
        char *prompt = NULL;
        int max_tokens = 128;
        int top_k = 40;
        double temperature = 0.7;

        /* For /v1/completions: extract "prompt" field */
        if (strcmp(req->path, "/v1/completions") == 0) {
            prompt = json_extract_str(req->body, "prompt");
        }

        /* For chat: extract messages[last].content */
        if (!prompt) {
            prompt = json_extract_str(req->body, "content");
        }

        /* Extract parameters */
        max_tokens = json_extract_int(req->body, "max_tokens", max_tokens);
        top_k = json_extract_int(req->body, "top_k", top_k);
        temperature = json_extract_double(req->body, "temperature", temperature);

        if (!prompt || prompt[0] == '\0') {
            free(prompt);
            send_error_resp(fd, 400, "No prompt or messages content found");
            return;
        }

        (void)temperature; /* Used by C code env, not by persist mode */

        /* Run inference */
        char *result = do_inference(prompt, max_tokens, top_k);
        free(prompt);

        if (!result) {
            send_error_resp(fd, 500,
                "Inference failed. Check model path, binary, and threads.");
            return;
        }

        send_json_resp(fd, 200, result);
        free(result);
        return;
    }

    /* 404 */
    send_error_resp(fd, 404, "Not found. Available: GET /health, GET /v1/models, "
                             "POST /v1/chat/completions, POST /v1/completions");
}


/* ================================================================
 *  Client handler (thread-based)
 * ================================================================ */

typedef struct {
    int fd;
} client_job_t;

static void *handle_client_thread(void *arg) {
    client_job_t *job = (client_job_t *)arg;
    int fd = job->fd;
    free(job);

    http_request_t req;
    if (parse_request(fd, &req)) {
        handle_request(fd, &req);
    }
    close(fd);
    return NULL;
}


/* ================================================================
 *  Main
 * ================================================================ */

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --port N        Port (default: %d)\n", DEFAULT_PORT);
    fprintf(stderr, "  --persist       Persistent process mode (model loads once)\n");
    fprintf(stderr, "  --bin PATH      gen_text_cpu binary path\n");
    fprintf(stderr, "  --model PATH    Model GGUF path\n");
    fprintf(stderr, "  --threads N     OMP threads (default: 4)\n");
    fprintf(stderr, "  --help          This message\n");
    fprintf(stderr, "\nEnvironment:\n");
    fprintf(stderr, "  INFER_BIN, MODEL_PATH, API_PORT, OMP_THREADS\n");
}

int main(int argc, char **argv) {
    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc)
            g_port = atoi(argv[++i]);
        else if (strcmp(argv[i], "--persist") == 0)
            g_persist_mode = 1;
        else if (strcmp(argv[i], "--bin") == 0 && i + 1 < argc)
            snprintf(g_infer_bin, sizeof(g_infer_bin), "%s", argv[++i]);
        else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            snprintf(g_model_path, sizeof(g_model_path), "%s", argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            g_omp_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    /* Environment overrides */
    char *env;
    if ((env = getenv("API_PORT"))) g_port = atoi(env);
    if ((env = getenv("INFER_BIN"))) snprintf(g_infer_bin, sizeof(g_infer_bin), "%s", env);
    if ((env = getenv("MODEL_PATH"))) snprintf(g_model_path, sizeof(g_model_path), "%s", env);
    if ((env = getenv("OMP_THREADS"))) g_omp_threads = atoi(env);

    /* Default model path */
    if (g_model_path[0] == '\0') {
        const char *home = getenv("HOME");
        snprintf(g_model_path, sizeof(g_model_path), "%s/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf",
                 home ? home : "/home/wubu2");
    }

    /* Validate binary */
    if (access(g_infer_bin, X_OK) != 0) {
        fprintf(stderr, "Binary not found or not executable: %s\n", g_infer_bin);
        return 1;
    }

    /* Unbuffer stderr for real-time output in pipe capture */
    setbuf(stderr, NULL);

    fprintf(stderr, "[server] Starting...\n");
    fprintf(stderr, "[server] Port: %d | Model: %s\n", g_port, g_model_path);
    fprintf(stderr, "[server] Binary: %s | Mode: %s\n", g_infer_bin,
            g_persist_mode ? "PERSISTENT" : "per-request");

    /* Start persistent process if needed */
    if (g_persist_mode) {
        pthread_mutex_init(&g_persist.lock, NULL);
        snprintf(g_persist.bin_path, sizeof(g_persist.bin_path), "%s", g_infer_bin);
        snprintf(g_persist.model_path, sizeof(g_persist.model_path), "%s", g_model_path);
        g_persist.omp_threads = g_omp_threads;

        if (!persist_start()) {
            fprintf(stderr, "[server] Failed to start persistent process\n");
            return 1;
        }
        persist_flush_init_stdout();
        fprintf(stderr, "[server] Persistent process PID=%d ready\n", g_persist.pid);
    }

    /* Signal handlers */
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    signal(SIGPIPE, SIG_IGN);

    /* Socket */
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return 1; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)g_port);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, BACKLOG) < 0) {
        perror("listen");
        close(server_fd);
        return 1;
    }

    printf("[server] Listening on port %d (mode: %s)\n", g_port,
           g_persist_mode ? "persistent" : "per-request");

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

        if (g_persist_mode) {
            /* Thread-based: share persistent process with mutex */
            client_job_t *job = (client_job_t *)malloc(sizeof(client_job_t));
            if (!job) { close(client_fd); continue; }
            job->fd = client_fd;

            pthread_t thread;
            pthread_create(&thread, NULL, handle_client_thread, job);
            pthread_detach(thread);
        } else {
            /* Fork-based: each request gets own process (safe for popen) */
            pid_t pid = fork();
            if (pid == 0) {
                close(server_fd);
                http_request_t req;
                if (parse_request(client_fd, &req))
                    handle_request(client_fd, &req);
                close(client_fd);
                _exit(0);
            } else if (pid > 0) {
                close(client_fd);
            } else {
                perror("fork");
                close(client_fd);
            }
        }
    }

    /* Cleanup */
    close(server_fd);
    if (g_persist_mode) {
        pthread_mutex_destroy(&g_persist.lock);
        persist_stop();
    }
    printf("[server] Shutdown complete.\n");
    return 0;
}
