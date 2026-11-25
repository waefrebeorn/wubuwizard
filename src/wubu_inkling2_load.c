/* wubu_inkling2_load.c -- EG-118 step 2: resolve the runtime tensor map.
 *
 * Step 1 (wubu_inkling2_meta.c) proved the file can be read and described. This step turns that into
 * something a forward pass can index: every tensor the graph needs, resolved to (info, file offset),
 * with a required/optional flag per family taken from the measured census rather than from a guess.
 *
 * The census, from the file itself:
 *   21 families on every layer
 *   13 more on the 12 loop layers  (loop_{attn,k,v,mlp}_{down,gain,forget} + attn_res_query)
 *    5 more on the 4 full-attention layers (5, 11, 17, 23): q/k/v input norms + k/v dynamic conv proj
 *    2 scale companions on 20 of 24 layers (ffn_{gate_up,down}_exps.scale)
 * The loop gate names map to the four message slots as attn=query, k=key, v=value, mlp=mlp, which is
 * the split the reference implementation uses (messages.unbind(-2)).
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "wubu_inkling2.h"

/* Resolution through the step-1 naming helpers, so both steps agree on what a tensor is called. */
static int resolve_layer(gguf_ctx *ctx, int il, const char *family, int64_t base, wubu_in2_tensor_t *out,
                         int required)
{
    char name[160];
    out->info = NULL;
    out->file_offset = 0;
    out->required = required;
    if (!wubu_inkling2_layer_tensor(ctx, il, family, name, sizeof name)) return 0;
    gguf_tensor_info *ti = gguf_find_tensor(ctx, name);
    if (!ti) return 0;
    out->info = ti;
    out->file_offset = base + (int64_t)ti->data_offset;
    return 1;
}

static int resolve_global(gguf_ctx *ctx, const char *name, int64_t base, wubu_in2_tensor_t *out, int required)
{
    out->info = NULL;
    out->file_offset = 0;
    out->required = required;
    gguf_tensor_info *ti = NULL;
    if (wubu_inkling2_global_tensor(ctx, name, &ti) != 1 || !ti) return 0;
    out->info = ti;
    out->file_offset = base + (int64_t)ti->data_offset;
    return 1;
}

/* Full-attention layer set: every 6th layer from 5. Derived from the config rather than hard-coded so a
 * different Inkling size uses the same rule. */
static int is_full_attn_layer(const wubu_inkling2_cfg_t *cfg, int il)
{
    if (cfg->attn_res_block <= 1) return il == cfg->n_layer_phys - 1;
    return (il % cfg->attn_res_block) == (cfg->attn_res_block - 1);
}

int wubu_inkling2_load(gguf_ctx *ctx, const wubu_inkling2_cfg_t *cfg, wubu_inkling2_model_t *m)
{
    if (!ctx || !cfg || !m) return WUBU_INKLING2_ERR_MISSING;
    memset(m, 0, sizeof *m);
    m->n_layer = cfg->n_layer_phys;

    const int64_t base = (int64_t)ctx->data_blob_offset;

    /* Globals. token_embd_norm is the embedding norm the reference applies before layer 0. */
    resolve_global(ctx, "token_embd.weight",      base, &m->tok_embd,  1);
    resolve_global(ctx, "output.weight",          base, &m->out_w,     1);
    resolve_global(ctx, "output_norm.weight",     base, &m->out_norm,  1);
    resolve_global(ctx, "output_res_query.weight", base, &m->out_res_q, 0);
    resolve_global(ctx, "loop_res_query.weight",  base, &m->msg_query, 1);
    resolve_global(ctx, "loop_kv_res_query.weight", base, &m->final_res_q, 0);

    static const char *slots[4] = { "attn", "k", "v", "mlp" };

    for (int il = 0; il < m->n_layer; ++il) {
        wubu_in2_layer_t *L = &m->layer[il];
        L->is_full_attn = is_full_attn_layer(cfg, il);
        L->is_loop_layer = (il >= cfg->loop_core_start && il < cfg->loop_core_end);

        /* 21 near-universal families. attn_norm is the exception: the 4 full-attention layers have no
         * attn_norm at all, because their per-q/k/v input norms replace it (verified: blk.5/11/17/23
         * are the only blocks without blk.N.attn_norm.weight). */
        resolve_layer(ctx, il, "attn_norm.weight",        base, &L->attn_norm, L->is_full_attn ? 0 : 1);
        resolve_layer(ctx, il, "attn_q.weight",           base, &L->wq,        1);
        resolve_layer(ctx, il, "attn_k.weight",           base, &L->wk,        1);
        resolve_layer(ctx, il, "attn_v.weight",           base, &L->wv,        1);
        resolve_layer(ctx, il, "attn_output.weight",      base, &L->wo,        1);
        resolve_layer(ctx, il, "attn_gate.weight",        base, &L->wgate,     1);
        resolve_layer(ctx, il, "attn_rel.weight",         base, &L->wrel,      1);
        resolve_layer(ctx, il, "attn_rel_bias.weight",    base, &L->rel_bias,  1);
        resolve_layer(ctx, il, "attn_q_clip.weight",      base, &L->q_clip,    1);
        resolve_layer(ctx, il, "attn_rel_clip.weight",    base, &L->rel_clip,  1);
        resolve_layer(ctx, il, "attn_sink_proj.weight",   base, &L->sink,      1);
        resolve_layer(ctx, il, "attn_k_conv.weight",      base, &L->k_conv,    1);
        resolve_layer(ctx, il, "attn_v_conv.weight",      base, &L->v_conv,    1);
        resolve_layer(ctx, il, "attn_post_conv.weight",   base, &L->attn_post_conv, 1);
        resolve_layer(ctx, il, "ffn_norm.weight",         base, &L->ffn_norm,  1);
        resolve_layer(ctx, il, "ffn_gate_up_exps.weight", base, &L->gate_up,   1);
        resolve_layer(ctx, il, "ffn_down_exps.weight",    base, &L->down,      1);
        resolve_layer(ctx, il, "ffn_gate_inp.weight",     base, &L->router,    1);
        resolve_layer(ctx, il, "exp_probs_b.bias",        base, &L->exp_bias,  1);
        resolve_layer(ctx, il, "ffn_post_conv.weight",    base, &L->ffn_post_conv, 1);
        resolve_layer(ctx, il, "ffn_res_query.weight",    base, &L->ffn_res_q, 1);

        /* Optional: per-expert scales (20 of 24 layers) - the runtime multiplier that takes the
         * expert weights from +-hundreds down to the same range as the dense tensors. */
        if (!resolve_layer(ctx, il, "ffn_gate_up_exps.scale", base, &L->gate_up_scale, 1))
            L->gate_up_scale.required = 0;
        if (!resolve_layer(ctx, il, "ffn_down_exps.scale", base, &L->down_scale, 1))
            L->down_scale.required = 0;

        /* Depth-mix query: present on every layer except 0, so the block residual runs from layer 1
         * onward and not only inside the loop core. Resolving it only for loop layers would have
         * silently dropped the mix across layers 1..5 and 18..23. */
        resolve_layer(ctx, il, "attn_res_query.weight", base, &L->attn_res_q, il >= 1 ? 1 : 0);

        /* Loop layers: the four message gates. */
        if (L->is_loop_layer) {
            for (int s = 0; s < 4; ++s) {
                char fam[48];
                snprintf(fam, sizeof fam, "loop_%s_down.weight", slots[s]);
                resolve_layer(ctx, il, fam, base, &L->loop_down[s], 1);
                snprintf(fam, sizeof fam, "loop_%s_gain.weight", slots[s]);
                resolve_layer(ctx, il, fam, base, &L->loop_gain[s], 1);
                snprintf(fam, sizeof fam, "loop_%s_forget.weight", slots[s]);
                resolve_layer(ctx, il, fam, base, &L->loop_forget[s], 1);
            }
        }

        /* Full-attention layers: separate q/k/v input norms and the dynamic-tap conv projections. */
        if (L->is_full_attn) {
            resolve_layer(ctx, il, "attn_q_inp_norm.weight", base, &L->q_inp_norm, 1);
            resolve_layer(ctx, il, "attn_k_inp_norm.weight", base, &L->k_inp_norm, 1);
            resolve_layer(ctx, il, "attn_v_inp_norm.weight", base, &L->v_inp_norm, 1);
            resolve_layer(ctx, il, "attn_k_conv_proj.weight", base, &L->k_conv_proj, 1);
            resolve_layer(ctx, il, "attn_v_conv_proj.weight", base, &L->v_conv_proj, 1);
        }
    }

    /* Census: count absent required tensors and absent optional ones separately. A missing required
     * tensor means the forward pass would silently read zeros, so the load reports it as an error. */
    m->n_missing_required = 0;
    m->n_optional_absent = 0;

    wubu_in2_tensor_t *globals[] = { &m->tok_embd, &m->out_w, &m->out_norm,
                                     &m->out_res_q, &m->msg_query, &m->final_res_q };
    for (size_t i = 0; i < sizeof globals / sizeof globals[0]; ++i) {
        if (globals[i]->info) continue;
        if (globals[i]->required) ++m->n_missing_required; else ++m->n_optional_absent;
    }

    for (int il = 0; il < m->n_layer; ++il) {
        wubu_in2_layer_t *L = &m->layer[il];
        wubu_in2_tensor_t *fields[] = {
            &L->attn_norm, &L->wq, &L->wk, &L->wv, &L->wo, &L->wgate, &L->wrel, &L->rel_bias,
            &L->q_clip, &L->rel_clip, &L->sink, &L->k_conv, &L->v_conv, &L->attn_post_conv,
            &L->ffn_norm, &L->gate_up, &L->down, &L->router, &L->exp_bias, &L->ffn_post_conv,
            &L->ffn_res_q, &L->gate_up_scale, &L->down_scale,
        };
        static const char *fnames[] = {
            "attn_norm", "attn_q", "attn_k", "attn_v", "attn_output", "attn_gate", "attn_rel",
            "attn_rel_bias", "attn_q_clip", "attn_rel_clip", "attn_sink_proj", "attn_k_conv",
            "attn_v_conv", "attn_post_conv", "ffn_norm", "ffn_gate_up_exps", "ffn_down_exps",
            "ffn_gate_inp", "exp_probs_b", "ffn_post_conv", "ffn_res_query", "gate_up_scale",
            "down_scale",
        };
        for (size_t i = 0; i < sizeof fields / sizeof fields[0]; ++i) {
            if (fields[i]->info) continue;
            if (fields[i]->required) {
                ++m->n_missing_required;
                if (getenv("IN2_VERBOSE")) fprintf(stderr, "  MISSING blk.%d.%s\n", il, fnames[i]);
            } else ++m->n_optional_absent;
        }
        if (L->is_loop_layer) {
            wubu_in2_tensor_t *lf[] = { &L->attn_res_q };
            for (size_t i = 0; i < sizeof lf / sizeof lf[0]; ++i)
                if (!lf[i]->info && lf[i]->required) ++m->n_missing_required;
            for (int s = 0; s < 4; ++s) {
                wubu_in2_tensor_t *g[] = { &L->loop_down[s], &L->loop_gain[s], &L->loop_forget[s] };
                for (int k = 0; k < 3; ++k)
                    if (!g[k]->info && g[k]->required) ++m->n_missing_required;
            }
        }
        if (L->is_full_attn) {
            wubu_in2_tensor_t *f[] = { &L->q_inp_norm, &L->k_inp_norm, &L->v_inp_norm,
                                       &L->k_conv_proj, &L->v_conv_proj };
            static const char *fn[] = { "attn_q_inp_norm", "attn_k_inp_norm", "attn_v_inp_norm",
                                        "attn_k_conv_proj", "attn_v_conv_proj" };
            for (size_t i = 0; i < sizeof f / sizeof f[0]; ++i)
                if (!f[i]->info && f[i]->required) {
                    ++m->n_missing_required;
                    if (getenv("IN2_VERBOSE")) fprintf(stderr, "  MISSING blk.%d.%s\n", il, fn[i]);
                }
        }
    }

    return m->n_missing_required ? WUBU_INKLING2_ERR_MISSING_TENSOR : WUBU_INKLING2_OK;
}

void wubu_inkling2_model_print(const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg)
{
    static const char *cls[3] = { "pre/post", "loop    ", "full    " };
    printf("physical layers %d (logical %d), core %d..%d x%d loops\n",
           m->n_layer, cfg->n_layer, cfg->loop_core_start, cfg->loop_core_end - 1, cfg->n_loops);
    for (int il = 0; il < m->n_layer; ++il) {
        const wubu_in2_layer_t *L = &m->layer[il];
        int n = 0, nopt = 0;
        const wubu_in2_tensor_t *f[] = {
            &L->attn_norm, &L->wq, &L->wk, &L->wv, &L->wo, &L->wgate, &L->wrel, &L->rel_bias,
            &L->q_clip, &L->rel_clip, &L->sink, &L->k_conv, &L->v_conv, &L->attn_post_conv,
            &L->ffn_norm, &L->gate_up, &L->down, &L->router, &L->exp_bias, &L->ffn_post_conv,
            &L->ffn_res_q, &L->attn_res_q,
        };
        for (size_t i = 0; i < sizeof f / sizeof f[0]; ++i) if (f[i]->info) ++n;
        for (int s = 0; s < 4; ++s) {
            if (L->loop_down[s].info) ++n;
            if (L->loop_gain[s].info) ++n;
            if (L->loop_forget[s].info) ++n;
        }
        if (L->gate_up_scale.info) ++nopt;
        if (L->down_scale.info) ++nopt;
        if (L->q_inp_norm.info) ++n;
        printf("  blk.%02d %s %2d resolved (+%d scales) %s%s\n", il, cls[L->is_full_attn ? 2 : (L->is_loop_layer ? 1 : 0)],
               n, nopt, L->is_loop_layer ? "loop " : "", L->is_full_attn ? "full-attn" : "");
    }
    printf("missing required: %d   optional absent: %d\n", m->n_missing_required, m->n_optional_absent);
}

/* Map the whole GGUF once instead of seeking and reading per call.
 *
 * The engine was reading the model from disk PER ROW: in2_dequant_slice mallocs, _fseeki64 + fread, dequantises,
 * frees. Fusing dequantisation into the dot made that worse, not better - one read per row means 8 experts x
 * (2304 + 2304) rows = 36,864 reads per layer, x 20 sliding layers x 3 loop passes = 2,211,840 seek+read pairs
 * plus mallocs for every single token. At 20-50 us per pair that is 44-111 s, which is the measured 77 s/token:
 * the engine was never compute-bound, it was syscall-bound, and it delivered 0.17 GFLOP/s (0.2% of one core's
 * AVX2 peak) with an effective 14 MB/s of weight bandwidth against 40-60 GB/s of DDR5.
 *
 * A read-only mapping turns every one of those into a memcpy from the page cache. The file is 8.4 GB and the box
 * has 16 GB, so after the first pass it stays resident; if the OS does evict, it faults back a 4 KB page rather
 * than paying a seek. Falls back to the seek/fread path if mapping is unavailable, so behaviour is unchanged
 * where it cannot work. */
#ifdef _WIN32
static HANDLE  g_map_h = NULL;
static const unsigned char *g_view = NULL;
static int64_t g_view_size = 0;
static int     g_map_state = 0;   /* 0 = untried, 1 = mapped, -1 = unavailable */

static int in2_try_map(gguf_ctx *ctx)
{
    if (g_map_state) return g_map_state > 0;
    g_map_state = -1;
    if (!ctx->file) return 0;
    const int fd = _fileno(ctx->file);
    if (fd < 0) return 0;
    const intptr_t osf = _get_osfhandle(fd);
    if (osf == -1) return 0;
    const int64_t size = _filelengthi64(fd);
    if (size <= 0) return 0;
    g_map_h = CreateFileMappingA((HANDLE)osf, NULL, PAGE_READONLY,
                                 (DWORD)((uint64_t)size >> 32), (DWORD)((uint64_t)size & 0xffffffffu), NULL);
    if (!g_map_h) return 0;
    g_view = (const unsigned char *)MapViewOfFile(g_map_h, FILE_MAP_READ, 0, 0, 0);
    if (!g_view) { CloseHandle(g_map_h); g_map_h = NULL; return 0; }
    g_view_size = size;
    g_map_state = 1;
    return 1;
}
#endif

/* Direct view of a tensor in the file mapping. The kernels take a const row pointer and a stride, so they can read
 * the mapping in place - copying each matrix out first was ~1.7 GB per decode token of pure memcpy on data that is
 * already mapped and never modified. NULL when the mapping is unavailable, in which case callers fall back to read. */
const unsigned char *wubu_inkling2_map_ptr(gguf_ctx *ctx, const wubu_in2_tensor_t *t)
{
#ifdef _WIN32
    if (!ctx || !t || !t->info || t->file_offset < 0) return NULL;
    if (!in2_try_map(ctx)) return NULL;
    if (t->file_offset > g_view_size) return NULL;
    return g_view + t->file_offset;
#else
    (void)ctx; (void)t;
    return NULL;
#endif
}

int wubu_inkling2_read(gguf_ctx *ctx, const wubu_in2_tensor_t *t, void *dst, int64_t nbytes)
{
    if (!ctx || !t || !t->info) return WUBU_INKLING2_ERR_MISSING_TENSOR;
#ifdef _WIN32
    if (t->file_offset >= 0 && in2_try_map(ctx)) {
        if (t->file_offset > g_view_size || nbytes < 0 || t->file_offset + nbytes > g_view_size)
            return WUBU_INKLING2_ERR_IO;
        memcpy(dst, g_view + t->file_offset, (size_t)nbytes);
        return WUBU_INKLING2_OK;
    }
#endif
    /* 64-bit seek: offsets in this file exceed 2 GB and a 32-bit seek silently reads the wrong bytes. */
    if (_fseeki64(ctx->file, (long long)t->file_offset, SEEK_SET) != 0) return WUBU_INKLING2_ERR_IO;
    if (fread(dst, 1, (size_t)nbytes, ctx->file) != (size_t)nbytes) return WUBU_INKLING2_ERR_IO;
    return WUBU_INKLING2_OK;
}

int wubu_inkling2_read_f32(gguf_ctx *ctx, const wubu_in2_tensor_t *t, float *dst, int64_t max_elems)
{
    if (!ctx || !t || !t->info || !dst) return WUBU_INKLING2_ERR_MISSING_TENSOR;
    const gguf_tensor_info *ti = t->info;
    int64_t ne = 1;
    for (int d = 0; d < ti->n_dims; ++d) ne *= ti->dims[d];
    if (ne > max_elems) return WUBU_INKLING2_ERR_SHAPE;

    if (ti->ggml_type == 0) {  /* F32: straight read */
        return wubu_inkling2_read(ctx, t, dst, ne * 4);
    }
    const int64_t nbytes = gguf_raw_size(ti->ggml_type, ne);
    if (nbytes <= 0) return WUBU_INKLING2_ERR_UNKNOWN_TYPE;
    unsigned char *raw = (unsigned char *)malloc((size_t)nbytes);
    if (!raw) return WUBU_INKLING2_ERR_IO;
    int rc = wubu_inkling2_read(ctx, t, raw, nbytes);
    if (rc == WUBU_INKLING2_OK) gguf_dequantize(raw, ti->ggml_type, ne, dst);
    free(raw);
    return rc;
}
