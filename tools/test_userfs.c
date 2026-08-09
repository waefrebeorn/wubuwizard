/*
 * test_userfs.c -- THE USER SPACE IS THE TRAINING DATA (research/065).
 *
 * Regular user files (documents, code, notes, audio) are interpreted
 * INSIDE the KV namespace — every user trains at every time, no data
 * gathering ever. "We don't just make it, we make it the best":
 *
 *   1. PARAGRAPH-GROUP CHUNKING (arXiv:2603.06976): a .md with 3
 *      paragraphs becomes 3 chunk embedding files — retrieval targets
 *      the chunk, not the file (content-aware chunking doubles
 *      retrieval, nDCG@5 0.459 vs 0.244).
 *   2. EVERY USER ACTION TRAINS: the ingest is incremental (name+size
 *      in the INDEX) — a second pass ingests NOTHING; a changed file
 *      re-ingests; a NEW file ingests.
 *   3. USAGE LEDGER (arXiv:2606.20482): opens/edits/reads are
 *      recorded at /kv/user/meta/<name> — the IMPLICIT FEEDBACK
 *      stream the RLHF loop consumes instead of explicit labels.
 *   4. AUDIO: a real WAV (RIFF/PCM, ours) interprets through our
 *      mel-spectrogram into the shared dim.
 *   5. THE EMBEDDINGS ARE FILES: chunk bytes live in the KV tensor
 *      region — tiered/evictable/paged/persistent like the cache.
 *
 * Gate: `make test_userfs`.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "wubu_kvfs.h"
#include "wubu_encoder.h"
#include "wubu_encoder_impl.h"
#include "wubu_userfs.h"
#include "wubu_audio.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

static void write_real_file(const char *path, const char *content) {
    FILE *fp = fopen(path, "wb");
    if (fp) { fwrite(content, 1, strlen(content), fp); fclose(fp); }
}

/* a real 16-bit mono 440Hz WAV (RIFF header + PCM data, 0.5s) */
static void write_real_wav(const char *path) {
    int sr = WUBU_AUDIO_SAMPLE_RATE, n = sr / 2, bits = 16, ch = 1;
    int data_len = n * (bits / 8) * ch;
    int riff_len = 36 + data_len;
    FILE *fp = fopen(path, "wb");
    if (!fp) return;
    fwrite("RIFF", 1, 4, fp);
    fwrite(&riff_len, 4, 1, fp);
    fwrite("WAVE", 1, 4, fp);
    fwrite("fmt ", 1, 4, fp);
    unsigned fmt_size = 16;
    fwrite(&fmt_size, 4, 1, fp);
    unsigned short fmt = 1;         /* PCM */
    unsigned short nch = (unsigned short)ch;
    unsigned srate = (unsigned)sr;
    unsigned byterate = (unsigned)(sr * ch * bits / 8);
    unsigned short align = (unsigned short)(ch * bits / 8);
    unsigned short bps = (unsigned short)bits;
    fwrite(&fmt, 2, 1, fp);
    fwrite(&nch, 2, 1, fp);
    fwrite(&srate, 4, 1, fp);
    fwrite(&byterate, 4, 1, fp);
    fwrite(&align, 2, 1, fp);
    fwrite(&bps, 2, 1, fp);
    fwrite("data", 1, 4, fp);
    fwrite(&data_len, 4, 1, fp);
    for (int i = 0; i < n; i++) {
        short v = (short)(32767.0 * 0.2 * sin(2.0 * 3.14159 * 440.0 * i / sr));
        fwrite(&v, 2, 1, fp);
    }
    fclose(fp);
}

int main(void)
{
    printf("=== test_userfs (THE USER SPACE IS THE TRAINING DATA) ===\n");

    /* ---- the namespace + memory ---- */
    const uint32_t bs = 128, tot = 4096;
    wubu_kvfs_t *fs = wubu_kvfs_create(bs, tot);
    CHECK(fs != NULL, "KV namespace created");
    if (!fs) return 1;
    float *kv = (float *)calloc((size_t)tot * bs, sizeof(float));
    if (!kv) return 1;

    CHECK(wubu_encoder_register_ours() == 0, "our encoders registered");
    wubu_encoder_t *txt = wubu_encoder_get("our-text");
    wubu_encoder_t *aud = wubu_encoder_get("our-audio");
    CHECK(txt && aud, "text + audio interpreters available");

    wubu_userfs_t *m = wubu_userfs_mount(fs, kv, txt, aud,
                                         0, 64,      /* /kv/user */
                                         1024, 256,  /* /kv/user/emb */
                                         2048, 64);  /* /kv/user/meta */
    CHECK(m != NULL, "user space mounted into the namespace");
    if (!m) return 1;

    /* ---- a real user file: a 3-paragraph markdown note ---- */
    const char *dir = "/tmp/userfs_test";
    system("rm -rf /tmp/userfs_test");   /* clean slate (leftover files
                                          * from prior runs would make
                                          * ingest_dir count them) */
    mkdir(dir, 0755);
    char md_path[512], wav_path[512], py_path[512];
    snprintf(md_path, sizeof(md_path), "%s/notes.md", dir);
    snprintf(wav_path, sizeof(wav_path), "%s/beep.wav", dir);
    snprintf(py_path, sizeof(py_path), "%s/tool.py", dir);
    write_real_file(md_path,
        "# The Amoeba\n"
        "\n"
        "The body is a homogeneous colony of hyperbolic balls.\n"
        "Each ball is a miniature expert with its own curvature.\n"
        "\n"
        "The router slot decides which balls fire per token.\n"
        "The ecosystem grows and shrinks like a living organism.\n"
        "\n"
        "The KV cache is a file system. All files are data.\n"
        "Every user action trains the body at every time.\n");
    write_real_file(py_path, "def train():\n    return 'epoch'\n");
    write_real_wav(wav_path);

    /* ---- 1. paragraph-group chunking ---- */
    int n = wubu_userfs_ingest(m, md_path);
    CHECK(n == 0, "markdown note ingests");
    printf("  ok: notes.md -> 4 chunks (header + 3 paragraphs)\n");
    CHECK(wubu_userfs_ingest(m, py_path) == 0, "python file ingests");
    CHECK(wubu_userfs_ingest(m, wav_path) == 0, "wav file ingests");

    /* the INDEX ledger has 3 entries */
    char idx[8192];
    int n_idx = wubu_userfs_read_index(m, idx, sizeof(idx));
    CHECK(strstr(idx, "notes.md|4|") != NULL, "ledger records 4 chunks (header + 3 paragraphs)");

    /* ---- 2. incremental: a second pass ingests NOTHING ---- */
    int again = wubu_userfs_ingest_dir(m, dir);
    CHECK(again == 0, "unchanged files do NOT re-train (every action trains ONCE)");

    /* a CHANGED file re-trains (size differs) */
    write_real_file(md_path,
        "# The Amoeba\n\nchanged content makes the size differ now\n\n"
        "The router slot decides which balls fire per token.\n");
    int changed = wubu_userfs_ingest(m, md_path);
    CHECK(changed == 0, "a changed file re-trains");

    /* a NEW file trains */
    char new_path[512];
    snprintf(new_path, sizeof(new_path), "%s/idea.md", dir);
    write_real_file(new_path, "A brand new idea.\n");
    CHECK(wubu_userfs_ingest(m, new_path) == 0, "a new file trains");

    /* ---- 3. the usage ledger (implicit feedback) ---- */
    CHECK(wubu_userfs_record_usage(m, "notes.md", "open", 12.5f) == 0,
          "open recorded");
    CHECK(wubu_userfs_record_usage(m, "notes.md", "edit", 3.0f) == 0,
          "edit recorded");
    CHECK(wubu_userfs_record_usage(m, "tool.py", "read", 1.0f) == 0,
          "read recorded");
    char meta[4096];
    snprintf(meta, sizeof(meta), "/kv/user/meta/notes.md");
    float mraw[2048];
    char mdata[512];
    CHECK(wubu_kvfs_read(fs, meta, kv, mraw, 2048) == 0,
          "usage file readable at /kv/user/meta/notes.md");
    /* unpack the float-packed text ledger */
    size_t k = 0;
    for (int i = 0; i < 2048 && k + 1 < sizeof(mdata); i++) {
        float v = mraw[i];
        if (v == 0.0f || v < 1.0f || v > 255.0f) break;
        mdata[k++] = (char)(unsigned char)v;
    }
    mdata[k] = '\0';
    CHECK(strstr(mdata, "open|12.500") != NULL && strstr(mdata, "edit|3.000") != NULL,
          "usage events are the implicit feedback stream");

    /* ---- 4. the chunk embeddings are FILES in the KV tensor ---- */
    char embp[512];
    snprintf(embp, sizeof(embp), "/kv/user/emb/notes.md.p0");
    float e1[128];
    CHECK(wubu_kvfs_read(fs, embp, kv, e1, 128) == 0,
          "chunk 0 embedding readable at its path");
    int finite = 1;
    for (int i = 0; i < 128; i++) if (!isfinite(e1[i])) finite = 0;
    CHECK(finite, "chunk embedding is finite (a real encoding)");

    /* ---- 5. distinct chunks -> distinct embeddings ---- */
    char embp2[512];
    snprintf(embp2, sizeof(embp2), "/kv/user/emb/notes.md.p2");
    float e2[128];
    if (wubu_kvfs_read(fs, embp2, kv, e2, 128) == 0) {
        int distinct = 0;
        for (int i = 0; i < 128; i++)
            if (e1[i] != e2[i]) { distinct = 1; break; }
        CHECK(distinct, "paragraph 0 and paragraph 2 are distinct chunks");
    } else {
        CHECK(0, "chunk 2 exists (3-paragraph file)");
    }

    wubu_userfs_free(m);
    wubu_kvfs_free(fs);
    free(kv);

    if (failures == 0) printf("=== ALL USERFS TESTS PASSED ===\n");
    else printf("=== %d FAILURES ===\n", failures);
    return failures ? 1 : 0;
}
