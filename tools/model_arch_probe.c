/* model_arch_probe.c — print the architecture registry's view of a model. */
#include "wubu_model_arch.h"
#include <stdio.h>

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        fprintf(stderr, "registered architectures:\n");
        const wubu_arch_entry_t *r = wubu_arch_registry();
        for (int i = 0; i < wubu_arch_registry_count(); i++)
            fprintf(stderr, "  %-14s -> %s / %s / %s\n", r[i].gguf_arch,
                    wubu_arch_id_name(r[i].id), wubu_family_name(r[i].family),
                    wubu_layout_name(r[i].layout));
        return 2;
    }
    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    wubu_arch_spec_t spec;
    if (!wubu_arch_probe(ctx, &spec)) { fprintf(stderr, "probe failed\n"); gguf_close(ctx); return 1; }
    wubu_arch_print(&spec, stdout);
    gguf_close(ctx);
    return 0;
}
