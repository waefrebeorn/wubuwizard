/*
 * wubu_train_cli.c -- the AGI training loop runner (the seed grows).
 *
 * Reads uint16 token streams (.tok, produced by wubu_tokenc from the
 * corpus on the SD card), trains WuBu-35M with the reference recipe
 * (Muon for matrices, AdamW for the embedding/norms, mean-reduced CE),
 * and checkpoints to the SD card every N steps.
 *
 * Usage:
 *   wubu_train_cli --model models/wubu/model.safetensors
 *                    --tok /home/wubu/sdcard/corpus/tokens/*.tok
 *                    --steps 100 --lr 1e-4 --out /home/wubu/sdcard/corpus/checkpoints/seed-1.st
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <glob.h>
#include <sys/stat.h>
#include "wubu.h"
#include "wubu_train.h"
#include "wubu_grow.h"
#include "wubu_plateau.h"
#include "wubu_diagnosis.h"
#include "wubu_hive.h"
#include "wubu_amoeba.h"
#include "wubu_moe2.h"
#include "wubu_selfimprove.h"
#include "wubu_rsi.h"
#include "wubu_lineage.h"
#include "wubu_contracts.h"
#include "wubu_priority_store.h"

/* FTZ + DAZ: flush denormals (from wuburvc's CPU research — the softmax/
 * exp/backprop tails create subnormals; denormal FP ops are ~100x slower
 * on x86). Sets MXCSR bits 15 (FTZ) + 6 (DAZ). */
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
static void flush_denormals(void)
{
    _mm_setcsr(_mm_getcsr() | (1u << 15) | (1u << 6));
}
#else
static void flush_denormals(void) {}
#endif

static const char *arg_get(int argc, char **argv, const char *name,
                           const char *def)
{
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    return def;
}
static int arg_int(int argc, char **argv, const char *name, int def)
{
    const char *v = arg_get(argc, argv, name, NULL);
    return v ? atoi(v) : def;
}
/* flag presence: --name (no value) */
static int arg_has(int argc, char **argv, const char *name)
{
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], name) == 0) return 1;
    return 0;
}
static float arg_float(int argc, char **argv, const char *name, float def)
{
    const char *v = arg_get(argc, argv, name, NULL);
    return v ? (float)atof(v) : def;
}

/* load a checkpoint dump into a freshly built model (the inverse of
 * save_checkpoint: magic + param count + the raw weights). Returns 0 on
 * success (the model owns the allocated buffers). */
static int load_checkpoint(wubu_model_t *m, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t magic = 0;
    if (fread(&magic, 4, 1, f) != 1 ||
        (magic != 0xBA000001u && magic != 0xBA000002u)) { fclose(f); return -1; }
    int nl = 0;
    if (magic == 0xBA000002u) {
        if (fread(&nl, 4, 1, f) != 1) { fclose(f); return -1; }
        if (nl < 1 || nl > WUBU_MAX_LAYERS) { fclose(f); return -1; }
    }
    long n = 0;
    if (fread(&n, sizeof(long), 1, f) != 1) { fclose(f); return -1; }
    /* the AGNOSTIC loader: the checkpoint's OWN param count drives the
     * geometry (the amoeba doctrine — dims are data). The count implies
     * the runtime dims: embed = vocab*d, per-layer = d*(8+2)*hd... The
     * checkpoint was SAVED with the runtime dims, so the runtime macros
     * (set by the save path / defaults) must match. Build the model at
     * the runtime dims and read EXACTLY that many floats. */
    float *embedding = (float *)malloc(sizeof(float) * (size_t)WUBU_VOCAB * WUBU_DIM);
    float *final_norm = (float *)malloc(sizeof(float) * WUBU_DIM);
    float **sel = (float **)calloc(WUBU_SELECTORS > 0 ? WUBU_SELECTORS : 1, sizeof(float *));
    wubu_block_t *blocks = (wubu_block_t *)calloc(WUBU_MAX_LAYERS, sizeof(wubu_block_t));
    if (!embedding || !final_norm || !sel || !blocks) { fclose(f); return -1; }
    for (int i = 0; i < WUBU_SELECTORS; i++) sel[i] = (float *)malloc(sizeof(float) * WUBU_DIM);
    wubu_block_t *b = blocks;
    const size_t hw = (size_t)WUBU_HEADS * WUBU_HEAD_DIM;      /* q/o width */
    const size_t kw = (size_t)WUBU_KV_HEADS * WUBU_HEAD_DIM;   /* k/v width */
    const size_t gu = (size_t)2 * WUBU_FFN_DIM;                /* gate_up width */
    for (int i = 0; i < WUBU_LAYERS; i++, b++) {
        b->q_proj    = (float *)malloc(sizeof(float) * (size_t)WUBU_DIM * hw);
        b->k_proj    = (float *)malloc(sizeof(float) * (size_t)WUBU_DIM * kw);
        b->v_proj    = (float *)malloc(sizeof(float) * (size_t)WUBU_DIM * kw);
        b->o_proj    = (float *)malloc(sizeof(float) * hw * WUBU_DIM);
        b->g_proj    = (float *)malloc(sizeof(float) * (size_t)WUBU_DIM * WUBU_DIM);
        b->q_norm    = (float *)malloc(sizeof(float) * WUBU_HEAD_DIM);
        b->k_norm    = (float *)malloc(sizeof(float) * WUBU_HEAD_DIM);
        b->attn_norm = (float *)malloc(sizeof(float) * WUBU_DIM);
        b->gate_up   = (float *)malloc(sizeof(float) * (size_t)WUBU_DIM * gu);
        b->down      = (float *)malloc(sizeof(float) * (size_t)WUBU_FFN_DIM * WUBU_DIM);
        b->ffn_norm  = (float *)malloc(sizeof(float) * WUBU_DIM);
        if (!b->q_proj || !b->k_proj || !b->v_proj || !b->o_proj || !b->g_proj ||
            !b->q_norm || !b->k_norm || !b->attn_norm || !b->gate_up || !b->down ||
            !b->ffn_norm) { fclose(f); return -1; }
    }
    if (fread(embedding, sizeof(float), (size_t)WUBU_VOCAB * WUBU_DIM, f) != (size_t)WUBU_VOCAB * WUBU_DIM ||
        fread(final_norm, sizeof(float), WUBU_DIM, f) != (size_t)WUBU_DIM) { fclose(f); return -1; }
    b = blocks;
    for (int i = 0; i < WUBU_LAYERS; i++, b++) {
        if (fread(b->q_proj, sizeof(float), (size_t)WUBU_DIM * hw, f) != (size_t)WUBU_DIM * hw ||
            fread(b->k_proj, sizeof(float), (size_t)WUBU_DIM * kw, f) != (size_t)WUBU_DIM * kw ||
            fread(b->v_proj, sizeof(float), (size_t)WUBU_DIM * kw, f) != (size_t)WUBU_DIM * kw ||
            fread(b->o_proj, sizeof(float), hw * WUBU_DIM, f) != hw * WUBU_DIM ||
            fread(b->g_proj, sizeof(float), (size_t)WUBU_DIM * WUBU_DIM, f) != (size_t)WUBU_DIM * WUBU_DIM ||
            fread(b->q_norm, sizeof(float), WUBU_HEAD_DIM, f) != (size_t)WUBU_HEAD_DIM ||
            fread(b->k_norm, sizeof(float), WUBU_HEAD_DIM, f) != (size_t)WUBU_HEAD_DIM ||
            fread(b->attn_norm, sizeof(float), WUBU_DIM, f) != (size_t)WUBU_DIM ||
            fread(b->gate_up, sizeof(float), (size_t)WUBU_DIM * gu, f) != (size_t)WUBU_DIM * gu ||
            fread(b->down, sizeof(float), (size_t)WUBU_FFN_DIM * WUBU_DIM, f) != (size_t)WUBU_FFN_DIM * WUBU_DIM ||
            fread(b->ffn_norm, sizeof(float), WUBU_DIM, f) != (size_t)WUBU_DIM) { fclose(f); return -1; }
    }
    for (int i = 0; i < WUBU_SELECTORS; i++)
        if (fread(sel[i], sizeof(float), WUBU_DIM, f) != (size_t)WUBU_DIM) { fclose(f); return -1; }
    fclose(f);
    if (wubu_model_init(m, embedding, final_norm, blocks, sel) != 0) return -1;
    if (nl > 0) m->n_layers = nl;   /* the v2 progressive state */
    /* the dump's count is the ACTIVE count (a v1 progressive checkpoint
     * saved with fewer layers); it must not EXCEED the built full count */
    if (n > wubu_parameter_count(m)) { fprintf(stderr, "checkpoint count mismatch (%ld vs %ld)\n", n, wubu_parameter_count(m)); return -1; }
    return 0;
}

/* read a .tok stream into a buffer; returns the token count. */
static long read_tokens(const char *pattern, uint16_t *buf, long cap)
{
    /* read MULTIPLE .tok files via the glob (the reasoning tier is
     * many files: openthoughts-114k.tok, openthoughts3-1.2m.tok, ...) */
    glob_t g;
    if (glob(pattern, 0, NULL, &g) != 0) return -1;
    long n = 0;
    for (size_t gi = 0; gi < g.gl_pathc && n < cap; gi++) {
        FILE *f = fopen(g.gl_pathv[gi], "rb");
        if (!f) continue;
        while (n < cap && fread(&buf[n], 2, 1, f) == 1) n++;
        fclose(f);
    }
    globfree(&g);
    return n;
}

/* write the raw weights to a checkpoint file (a simple float dump --
 * the safetensors save is the next milestone). */
static int save_checkpoint(const wubu_model_t *m, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    /* header: magic v2 (the n_layers) + param count */
    uint32_t magic = 0xBA000002u;
    fwrite(&magic, 4, 1, f);
    int nl = m->n_layers;
    fwrite(&nl, 4, 1, f);
    long n = wubu_parameter_count(m);
    fwrite(&n, sizeof(long), 1, f);
    const size_t hw = (size_t)WUBU_HEADS * WUBU_HEAD_DIM;      /* q/o width */
    const size_t kw = (size_t)WUBU_KV_HEADS * WUBU_HEAD_DIM;   /* k/v width */
    const size_t gu = (size_t)2 * WUBU_FFN_DIM;                /* gate_up width */
    fwrite(m->embedding, sizeof(float), (size_t)WUBU_VOCAB * WUBU_DIM, f);
    fwrite(m->final_norm, sizeof(float), WUBU_DIM, f);
    for (int i = 0; i < WUBU_LAYERS; i++) {
        wubu_block_t *b = (wubu_block_t *)&m->blocks[i];
        fwrite(b->q_proj, sizeof(float), (size_t)WUBU_DIM * hw, f);
        fwrite(b->k_proj, sizeof(float), (size_t)WUBU_DIM * kw, f);
        fwrite(b->v_proj, sizeof(float), (size_t)WUBU_DIM * kw, f);
        fwrite(b->o_proj, sizeof(float), hw * WUBU_DIM, f);
        fwrite(b->g_proj, sizeof(float), (size_t)WUBU_DIM * WUBU_DIM, f);
        fwrite(b->q_norm, sizeof(float), WUBU_HEAD_DIM, f);
        fwrite(b->k_norm, sizeof(float), WUBU_HEAD_DIM, f);
        fwrite(b->attn_norm, sizeof(float), WUBU_DIM, f);
        fwrite(b->gate_up, sizeof(float), (size_t)WUBU_DIM * gu, f);
        fwrite(b->down, sizeof(float), (size_t)WUBU_FFN_DIM * WUBU_DIM, f);
        fwrite(b->ffn_norm, sizeof(float), WUBU_DIM, f);
    }
    for (int i = 0; i < WUBU_SELECTORS; i++)
        fwrite(m->selectors[i], sizeof(float), WUBU_DIM, f);
    fclose(f);
    return 0;
}

/* Rolling checkpoint retention (the file-bloat fix, 2026-08-04): after a
 * step checkpoint is saved, prune this out_path line down to the newest
 * `keep` step checkpoints. keep = 3 normally; if the line's step
 * checkpoints exceed CKPT_LARGE_BYTES (~1 GiB) tighten to 2. The final
 * checkpoint (out_path itself, no -NNNN.st suffix) is the anchor and is
 * never pruned. Without this, a long run accumulates every step
 * checkpoint forever (103 files / 14 GiB of stale seed-40..47 lines). */
#define CKPT_KEEP_DEFAULT 3
#define CKPT_KEEP_TIGHT   2
#define CKPT_LARGE_BYTES  (1024LL * 1024 * 1024)

static int cmp_step(const void *a, const void *b)
{
    const char *fa = *(const char *const *)a;
    const char *fb = *(const char *const *)b;
    const char *sa = strrchr(fa, '-');
    const char *sb = strrchr(fb, '-');
    long ia = sa ? atol(sa + 1) : 0;
    long ib = sb ? atol(sb + 1) : 0;
    return (ia > ib) - (ia < ib);
}

static void prune_step_checkpoints(const char *out_path)
{
    char pattern[576];
    snprintf(pattern, sizeof(pattern), "%s-*.st", out_path);
    glob_t g;
    if (glob(pattern, 0, NULL, &g) != 0 || g.gl_pathc == 0) {
        if (g.gl_pathc) globfree(&g);
        return;
    }
    qsort(g.gl_pathv, g.gl_pathc, sizeof(char *), cmp_step);
    long long total = 0;
    for (size_t i = 0; i < g.gl_pathc; i++) {
        struct stat st;
        if (stat(g.gl_pathv[i], &st) == 0) total += (long long)st.st_size;
    }
    int keep = (total > CKPT_LARGE_BYTES) ? CKPT_KEEP_TIGHT : CKPT_KEEP_DEFAULT;
    if ((int)g.gl_pathc <= keep) { globfree(&g); return; }
    for (size_t i = 0; i + (size_t)keep < g.gl_pathc; i++) {
        if (remove(g.gl_pathv[i]) == 0)
            printf("  prune -> %s (rolling keep %d)\n", g.gl_pathv[i], keep);
    }
    globfree(&g);
}

int main(int argc, char **argv)
{
    const char *model_path = arg_get(argc, argv, "--model",
                                     "models/wubu/model.safetensors");
    const char *tok_glob = arg_get(argc, argv, "--tok",
                                   "/home/wubu/sdcard/corpus/tokens/cosmopedia-v2-00000.tok");
    const char *out_path = arg_get(argc, argv, "--out",
                                   "/home/wubu/sdcard/corpus/checkpoints/seed.st");
    const char *resume = arg_get(argc, argv, "--resume", NULL);
    int max_steps = arg_int(argc, argv, "--steps", 50);
    float lr = arg_float(argc, argv, "--lr", 1e-4f);
    float muon_lr = arg_float(argc, argv, "--muon-lr", 1e-3f);
    float adam_lr = arg_float(argc, argv, "--adam-lr", 1e-3f);
    int seq = arg_int(argc, argv, "--seq", 128);
    int ckpt_every = arg_int(argc, argv, "--ckpt", 10);
    int grow_check = arg_int(argc, argv, "--grow-check", 0);
    /* Phase 1: the closed loop is the DEFAULT — every batch emits a
     * Diagnosis cell. --diag-every 1 = full mutate-gate per batch;
     * larger values still RECORD every batch (only the mutation cycle
     * is gated by the interval). There is no silent path. */
    int diag_every = arg_int(argc, argv, "--diag-every", 1);
    /* AN47: the ENDURANCE path — --resume <base> also loads the
     * .hive archive + the .prio sidecar so a kill/restart continues
     * the colony's fitness history, not zero */
    /* AN47: --ckpt-hive N saves the hive + prio sidecars every N steps
     * (not just at teardown) so the endurance run survives a kill */
    int ckpt_hive = arg_int(argc, argv, "--ckpt-hive", 0);
    int base_layers = arg_int(argc, argv, "--base-layers", 0);
    int init_random = arg_has(argc, argv, "--init-random");
    flush_denormals();   /* the wuburvc CPU speed trick (MXCSR FTZ+DAZ) */

    wubu_model_t m;
    if (resume) {
        printf("wubu_train_cli: resuming from %s ...\n", resume);
        /* the agnostic resume: set the runtime dims BEFORE loading (the
         * checkpoint was saved at the runtime geometry; the defaults
         * are the aligned WuBu1 geometry unless already set) */
        if (WUBU_RUNTIME_DIMS.dim == 0) wubu_runtime_dims_default();
        if (load_checkpoint(&m, resume) != 0) {
            fprintf(stderr, "cannot load checkpoint %s\n", resume);
            return 1;
        }
    } else if (init_random) {
        printf("wubu_train_cli: FROM-SCRATCH random init at the runtime dims "
               "(the amoeba doctrine: dims are data, no pretrained weights)\n");
        if (wubu_model_random_init(&m) != 0) {
            fprintf(stderr, "cannot build the from-scratch model\n");
            return 1;
        }
    } else {
        printf("wubu_train_cli: loading %s ...\n", model_path);
        if (wubu_load(&m, model_path) != 0) {
            fprintf(stderr, "cannot load %s\n", model_path);
            return 1;
        }
    }
    printf("wubu_train_cli: %ld parameters\n", wubu_parameter_count(&m));
    if (base_layers > 0) {
        if (base_layers > WUBU_LAYERS) base_layers = WUBU_LAYERS;
        m.n_layers = base_layers;   /* the progressive start (Bu: start small) */
        printf("wubu_train_cli: progressive start at %d layers\n", m.n_layers);
    }

    /* the corpus glob (multiple .tok files) — cap 1<<29 = 512M tokens
     * (1GB): the reasoning tier (openthoughts-114k 244M + ot3 39M +
     * gpt-5.6 17M + ...) needs the room; the old 16M cap truncated. */
    uint16_t *corpus = (uint16_t *)malloc(sizeof(uint16_t) * (1 << 29));
    long corpus_n = read_tokens(tok_glob, corpus, 1 << 29);
    if (corpus_n <= 0) {
        fprintf(stderr, "cannot read corpus %s\n", tok_glob);
        return 1;
    }
    printf("wubu_train_cli: corpus %ld tokens\n", corpus_n);

    wubu_buf_t b;
    if (wubu_buf_alloc(&b, WUBU_MAX_SEQ) != 0) return 1;
    wubu_train_t tr;
    if (wubu_train_init(&tr, &m) != 0) return 1;

    wubu_train_cfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.lr = lr;
    cfg.muon_lr = muon_lr;    /* the Moonlight RMS-0.2 scale makes the Muon
                                 group LR comparable to AdamW (recipe:
                                 2e-2 from scratch; 1e-3 for fine-tuning
                                 the released checkpoint) */
    cfg.adam_lr = adam_lr;
    cfg.weight_decay = 0.1f;
    cfg.muon_momentum = 0.95f;
    cfg.grad_clip = 1.0f;     /* the recipe's global-norm clip */
    cfg.warmup_steps = (uint32_t)(max_steps / 10);
    cfg.max_steps = (uint32_t)max_steps;

    /* the training loop: sliding windows over the corpus */
    uint16_t win[WUBU_MAX_SEQ];
    long pos = 0;
    double loss_ema = -1;
    float loss_hist[64];
    int hist_n = 0;

    /* the CLOSED CONTROL LOOP (the user directive: the colony is a
     * self-modifying AGI, not an inference engine): every batch ->
     * structured Diagnosis -> hive fitness cells -> amoeba mutate ->
     * validate (loss tol + prover) -> archive/graveyard. The organs
     * live for the run; the hive is the ONLY fitness recorder. */
    wubu_hive_t diag_tissue;
    wubu_amoeba_t diag_amoeba;
    wubu_moe2_t diag_agents;
    wubu_diag_loop_t diag_loop;
    wubu_hive_init(&diag_tissue);
    wubu_moe2_init(&diag_agents, 42);
    wubu_amoeba_cfg_t acfg;
    memset(&acfg, 0, sizeof(acfg));
    acfg.grow_util = 0.7; acfg.grow_grad = 0.3;
    acfg.shrink_util = 0.02; acfg.shrink_grad = 0.1;
    acfg.entropy_min = 0.05; acfg.loss_tol = 0.05;
    acfg.split_eps = 0.01; acfg.max_cells = 8; acfg.min_cells = 2;
    wubu_amoeba_init(&diag_amoeba, &acfg, &diag_tissue, &diag_agents);
    wubu_diag_loop_init(&diag_loop, &diag_tissue, &diag_amoeba,
                        &diag_agents, 256, 128);
    /* the RSI-powered mutation engine (P20): the trace/span operator
     * writes into the hive; the experience loop feeds the diagnose;
     * the amoeba calls the RSI proposals (directive #4: RSI is the
     * mutation engine, not a parallel system). */
    wubu_selfimprove_t si;
    wubu_selfimprove_init(&si, &diag_tissue);
    /* Phase 1: the lineage tracker + the runtime contracts are part
     * of the gate (a mutation survives only through fitness + prover +
     * contracts + lineage pressure). */
    wubu_lineage_tracker_t lineage;
    wubu_lineage_init(&lineage, 64, 8, 0.2f);
    wubu_contracts_t contracts;
    wubu_contracts_init(&contracts, &diag_tissue);
    diag_loop.lineage = &lineage;
    diag_loop.contracts = &contracts;
    /* Phase 2: the priority store — the diagnose consults it before
     * every mutation (BI + the online Fisher + the ledger) */
    wubu_priority_store_t prio;
    wubu_prio_init(&prio, 0.5f, 0.3f);
    diag_loop.prio = &prio;
    /* AN47: the ENDURANCE resume — the colony's fitness history + the
     * priority evidence come back from the sidecars (a kill/restart
     * continues, it does not reset) */
    if (resume) {
        char hpath[640], ppath[640];
        snprintf(hpath, sizeof(hpath), "%s.hive", resume);
        snprintf(ppath, sizeof(ppath), "%s.prio", resume);
        int hn = wubu_diag_load(&diag_loop, hpath);
        if (hn > 0) {
            printf("  closed loop: resumed %d fitness cells from %s\n", hn, hpath);
            /* replay the resumed ledger into the hive tissue (the walk
             * reads the live hive + the archive both) */
            for (int i = 0; i < diag_loop.ledger_n; i++) {
                wubu_fitness_cell_t *c = (wubu_fitness_cell_t *)
                    calloc(1, sizeof(wubu_fitness_cell_t));
                if (c) { *c = diag_loop.ledger[i]; wubu_hive_insert(&diag_tissue, c); }
            }
        } else {
            printf("  closed loop: no .hive archive at %s (fresh colony)\n", hpath);
        }
        FILE *pf = fopen(ppath, "rb");
        if (pf) {
            static char pbuf[8192];
            long pn = (long)fread(pbuf, 1, sizeof(pbuf), pf);
            fclose(pf);
            int pr = wubu_prio_load(&prio, pbuf, pn);
            if (pr > 0) printf("  priority store: resumed %d cells from %s\n", pr, ppath);
        }
    }
    for (int step = 1; step <= max_steps; step++) {
        if (pos + seq > corpus_n) pos = 0;   /* epoch wrap */
        for (int i = 0; i < seq; i++) win[i] = corpus[pos + i];
        pos += seq;
        float loss = wubu_train_step_loop(&m, &tr, &b, win, (size_t)seq,
                                           &cfg, (uint32_t)step);
        loss_ema = loss_ema < 0 ? loss : 0.9 * loss_ema + 0.1 * loss;
        /* the ema history is a SHIFTING window (NOT a ring: the plateau
         * detector reads the last-32 ARRAY elements; a ring wraps and
         * the detector would fit its slope over the wrong data) */
        if (hist_n >= 64) {
            memmove(loss_hist, loss_hist + 1, 63 * sizeof(float));
            loss_hist[63] = (float)loss_ema;
        } else {
            loss_hist[hist_n++] = (float)loss_ema;
        }
        /* the closed loop: every batch -> Diagnosis -> hive fitness
         * cell; the mutation cycle runs every diag_every batches */
        {
            wubu_diag_record_t rec;
            memset(&rec, 0, sizeof(rec));
            rec.batch = (uint64_t)step;
            rec.epoch = 1;
            rec.loss = loss;
            rec.loss_ema = (float)loss_ema;
            rec.fitness = (float)loss_ema;
            /* the PRE-mutation fitness: the previous batch's ema (the
             * current ema was just pushed to loss_hist[hist_n-1], so
             * the previous is [hist_n-2]; the gate compares held-out
             * AFTER the mutation to this — the delta is the real
             * improvement/regression) */
            rec.prev_fitness = (float)(hist_n > 1
                ? loss_hist[hist_n - 2] : loss_hist[hist_n > 0 ? hist_n - 1 : 0]);
            rec.n_experts = MOE2_N_EXPERTS;
            rec.grad_norm_mean = (float)(tr.grad_norm_sum /
                                         (tr.micro_steps > 0 ? tr.micro_steps : 1));
            wubu_diag_loss_surface(&rec, loss_hist, hist_n);
            wubu_diag_collect_grads(&diag_loop, &tr, m.n_layers,
                                    WUBU_DIM, WUBU_FFN_DIM,
                                    WUBU_KV_HEADS * WUBU_HEAD_DIM,
                                    WUBU_HEADS * WUBU_HEAD_DIM);
            /* Phase 2: the online Fisher (EWC) — the real grad norm
             * feeds the diagonal per batch (what the loss cares
             * about); the diagnose gate protects those cells */
            {
                float gn = (float)(tr.grad_norm_sum /
                                   (tr.micro_steps > 0 ? tr.micro_steps : 1));
                for (int c = 0; c < diag_loop.n_cells_alloc; c++) {
                    wubu_prio_register(&prio, (uint8_t)c, 2,
                                       diag_loop.cell_grads[c], (uint64_t)step);
                    wubu_prio_update_fisher(&prio, (uint8_t)c,
                                            gn > 0 ? gn : diag_loop.cell_grads[c],
                                            0.1f);
                }
            }
            if (diag_every > 0 && step % diag_every == 0) {
                wubu_diag_cycle(&diag_loop, &rec, (float)loss_ema);
                /* the RSI engine proposes a mutation under the gate;
                 * the proposal runs through the amoeba's mutate (the
                 * amoeba is the ONLY mutation operator) */
                wubu_mutation_t mut;
                float vscore = rec.grad_norm_mean > 0 ? 0.8f : 0.4f;
                if (wubu_selfimprove_step(&si, vscore, rec.slope < 0 ? 0.9f : 0.4f,
                                          1.0f, 0.5f, &mut)) {
                    /* the trace span: this batch's outcome */
                    wubu_selfimprove_trace(&si, WUBU_SPAN_BATCH,
                                           (uint64_t)step, (uint64_t)step,
                                           (float)loss_ema, 1, 0, 0xFF);
                    wubu_selfimprove_report(&si, 1);
                }
            } else {
                wubu_diag_record(&diag_loop, &rec);
            }
        }
        if (grow_check > 0 && step % grow_check == 0) {
            if (hist_n >= 32 && m.n_layers < WUBU_LAYERS &&
                /* the adaptive threshold: the absolute floor OR 0.5% of
                 * the loss magnitude -- the 0.001 floor alone sat below
                 * the fine-tune-scale noise (~2.8 loss, ~0.02 slope
                 * noise) and the growth never fired */
                wubu_plateau_detect(loss_hist, hist_n, 32,
                                    0.001f > 0.005f * (float)loss_ema
                                        ? 0.001f : 0.005f * (float)loss_ema)) {
                int pos_g = m.n_layers / 2;   /* the progressive deepening */
                /* NOTE: wubu_grow_insert_block already incremented m.n_layers
                 * before we get here.  wubu_train_grow expects the PRE-grow
                 * layer count (it frees the displaced-inactive slot at index
                 * [n_layers] before the shift).  Passing the post-increment
                 * value off-by-one frees a live slot, then the shift aliases
                 * the previous layer's pointer into that index → double-free
                 * at teardown (wubu_train_free). */
                int pre_layers = m.n_layers - 1;
                if (wubu_grow_insert_block(&m, pos_g) &&
                    wubu_train_grow(&tr, pos_g, pre_layers)) {
                    printf("  GROW at step %d: n_layers %d -> %d (plateau)\n",
                           step, m.n_layers - 1, m.n_layers);
                }
            }
        }
        if (step % 5 == 0 || step == 1)
            printf("  step %4d: loss %.4f (ema %.4f)\n", step, loss, loss_ema);
        /* Phase 1: the finite guard — every step the model must be
         * finite (NaN/Inf = a silent corruption the colony cannot
         * diagnose; abort before the next batch). */
        {
            const float *emb = m.embedding;
            int bad = 0;
            if (emb) {
                for (int i = 0; i < WUBU_VOCAB * WUBU_DIM && !bad; i += 4096) {
                    size_t n = (size_t)(WUBU_VOCAB * WUBU_DIM - i < 4096
                                            ? WUBU_VOCAB * WUBU_DIM - i : 4096);
                    if (!wubu_contracts_finite(emb + i, n)) bad = 1;
                }
            }
            if (bad) {
                printf("  FATAL: the model went non-finite at step %d — aborting\n", step);
                break;
            }
        }
        if (step % ckpt_every == 0) {
            char ck[512];
            snprintf(ck, sizeof(ck), "%s-%04d.st", out_path, step);
            if (save_checkpoint(&m, ck) == 0) {
                printf("  checkpoint -> %s\n", ck);
                prune_step_checkpoints(out_path);  /* rolling keep 3 (2 if large) */
                /* AN47: the endurance sidecars ride along — the hive
                 * + the priority store save at the same cadence so a
                 * kill/restart resumes from THIS step, not teardown */
                if (ckpt_hive > 0 && step % ckpt_hive == 0) {
                    char hck[640];
                    snprintf(hck, sizeof(hck), "%s.hive", ck);
                    if (wubu_diag_save(&diag_loop, hck) == 0)
                        printf("  colony sidecars -> %s.hive/.prio\n", ck);
                    char pck[640];
                    snprintf(pck, sizeof(pck), "%s.prio", ck);
                    static char pbuf[8192];
                    long pn = wubu_prio_save(&prio, pbuf, (long)sizeof(pbuf));
                    if (pn > 0) {
                        FILE *pf = fopen(pck, "wb");
                        if (pf) { fwrite(pbuf, 1, (size_t)pn, pf); fclose(pf); }
                    }
                }
            }
        }
    }
    if (save_checkpoint(&m, out_path) == 0)
        printf("final checkpoint -> %s\n", out_path);

    /* the closed loop teardown: report the colony's vitals + the RSI
     * engine + archive the fitness ledger (the hive walk reads it) */
    {
        char stats[256];
        wubu_diag_stats(&diag_loop, stats, sizeof(stats));
        printf("  closed loop: %s\n", stats);
        char sistats[256];
        wubu_selfimprove_stats(&si, sistats, sizeof(sistats));
        printf("  selfimprove: %s\n", sistats);
        char lstats[256], cstats[256];
        wubu_lineage_stats(&lineage, lstats, sizeof(lstats));
        wubu_contracts_stats(&contracts, cstats, sizeof(cstats));
        printf("  lineage: %s\n", lstats);
        printf("  contracts: %s\n", cstats);
        char pstats[256];
        wubu_prio_stats(&prio, pstats, sizeof(pstats));
        printf("  priority store: %s\n", pstats);
        printf("  closed loop: hive live cells %zu (fitness archive)\n",
               wubu_hive_live(&diag_tissue));
        char arch[640];
        snprintf(arch, sizeof(arch), "%s.hive", out_path);
        if (wubu_diag_save(&diag_loop, arch) == 0)
            printf("  closed loop: archive -> %s (wubu_hive_walk %s --accepted)\n",
                   arch, arch);
        /* Phase 2: the priority sidecar — the checkpoint's ledger
         * (BI + Fisher + precision deltas + the mutation history) */
        char psck[640];
        snprintf(psck, sizeof(psck), "%s.prio", out_path);
        static char pbuf[8192];
        long pn = wubu_prio_save(&prio, pbuf, (long)sizeof(pbuf));
        if (pn > 0) {
            FILE *pf = fopen(psck, "wb");
            if (pf) { fwrite(pbuf, 1, (size_t)pn, pf); fclose(pf); }
            printf("  priority store: sidecar -> %s (%ld bytes)\n", psck, pn);
        }
        wubu_diag_loop_free(&diag_loop);
        wubu_amoeba_free(&diag_amoeba);
        wubu_moe2_free(&diag_agents);
        wubu_hive_clear(&diag_tissue);
    }

    wubu_train_free(&tr);
    wubu_free(&m, &b);
    free(corpus);
    printf("wubu_train_cli: done\n");
    return 0;
}
