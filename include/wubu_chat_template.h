#ifndef WUBU_CHAT_TEMPLATE_H
#define WUBU_CHAT_TEMPLATE_H

/*
 * wubu_chat_template.h -- EG-14: minimal Jinja-subset chat template
 * renderer (llama.cpp minja / TGI lineage), reimplemented in C11.
 *
 * Supported subset: literal text, {{ expr }}, {% for v in messages %} ...
 * {% endfor %}, {% if v.role == 'role' %} ... {% else %} ... {% endif %}.
 * Expressions resolve v.role / v.content for the loop variable plus the
 * bos/eos globals supplied by the caller. Malformed templates fail closed.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *role;
    const char *content;
} wubu_chat_message_t;

/* Parse a template string. Returns an opaque template or NULL on error. */
void *wubu_chat_template_compile(const char *text);
void wubu_chat_template_free(void *tpl);

/* Render into out (cap bytes). Returns output length or -1 on error. */
int wubu_chat_template_render(void *tpl,
                              const wubu_chat_message_t *messages, int n_messages,
                              const char *bos, const char *eos,
                              char *out, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_CHAT_TEMPLATE_H */
