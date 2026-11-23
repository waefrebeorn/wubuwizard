/*
 * wubu_chat_template.c -- EG-14 minimal Jinja subset. Pure C11.
 * Nodes are individually allocated so recursive parsing never holds a
 * pointer into a reallocated array.
 */
#include "wubu_chat_template.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef enum { N_TEXT, N_EXPR, N_FOR, N_IF } node_kind_t;
typedef struct node {
    node_kind_t kind;
    char *text;                 /* literal, expression, or loop variable */
    char *role;                 /* IF expected role */
    struct node *body, *alt, *next;
} node_t;

typedef struct { node_t *root; } template_t;

typedef struct {
    const char *src;
    size_t len, pos;
    int ok;
} parser_t;

static char *dup_trim(const char *s, size_t n) {
    size_t a = 0, b = n;
    char *out;
    while (a < b && isspace((unsigned char)s[a])) a++;
    while (b > a && isspace((unsigned char)s[b - 1])) b--;
    out = (char *)malloc(b - a + 1);
    if (!out) return NULL;
    memcpy(out, s + a, b - a);
    out[b - a] = '\0';
    return out;
}

static char *dup_raw(const char *s, size_t n) {
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

static node_t *new_node(node_kind_t k) {
    node_t *n = (node_t *)calloc(1, sizeof(*n));
    if (n) n->kind = k;
    return n;
}

static void free_nodes(node_t *n) {
    while (n) {
        node_t *next = n->next;
        free_nodes(n->body);
        free_nodes(n->alt);
        free(n->text);
        free(n->role);
        free(n);
        n = next;
    }
}

static int consume_tag(parser_t *p, char **tag_out) {
    size_t start, end;
    if (p->pos + 2 > p->len || p->src[p->pos] != '{' || p->src[p->pos + 1] != '%')
        return -1;
    start = p->pos + 2;
    end = start;
    while (end + 1 < p->len && !(p->src[end] == '%' && p->src[end + 1] == '}'))
        end++;
    if (end + 1 >= p->len) return -1;
    *tag_out = dup_trim(p->src + start, end - start);
    if (!*tag_out) return -1;
    p->pos = end + 2;
    return 0;
}

static node_t *parse_seq(parser_t *p, const char *const *stops, int nstops,
                         const char **hit) {
    node_t head, *cur = &head;
    const char *found = NULL;
    memset(&head, 0, sizeof(head));
    if (hit) *hit = NULL;

    while (p->ok && p->pos < p->len) {
        if (p->pos + 1 < p->len && p->src[p->pos] == '{' &&
            p->src[p->pos + 1] == '%') {
            char *tag = NULL;
            int i, matched = 0;
            node_t *n;
            if (consume_tag(p, &tag) != 0) { p->ok = 0; break; }
            for (i = 0; i < nstops; i++) {
                if (!strcmp(tag, stops[i])) {
                    matched = 1;
                    found = stops[i];
                    if (hit) *hit = stops[i];
                    break;
                }
            }
            if (matched) { free(tag); break; }

            if (!strncmp(tag, "for ", 4)) {
                const char *endstop[] = { "endfor" };
                char *var = dup_trim(tag + 4, strlen(tag + 4));
                char *in;
                if (!var) { free(tag); p->ok = 0; break; }
                in = strstr(var, " in ");
                if (in) *in = '\0';
                n = new_node(N_FOR);
                if (!n) { free(var); free(tag); p->ok = 0; break; }
                n->text = var;
                n->body = parse_seq(p, endstop, 1, NULL);
            } else if (!strncmp(tag, "if ", 3)) {
                const char *ifstops[] = { "else", "endif" };
                const char *which = NULL;
                char *eq = strstr(tag + 3, "==");
                n = new_node(N_IF);
                if (!n) { free(tag); p->ok = 0; break; }
                if (eq) {
                    size_t rl;
                    n->role = dup_trim(eq + 2, strlen(eq + 2));
                    if (!n->role) { free(tag); p->ok = 0; break; }
                    rl = strlen(n->role);
                    if (rl >= 2 && n->role[0] == '\'' && n->role[rl - 1] == '\'') {
                        char *clean = dup_trim(n->role + 1, rl - 2);
                        free(n->role);
                        n->role = clean;
                        if (!clean) { free(tag); p->ok = 0; break; }
                    }
                }
                n->body = parse_seq(p, ifstops, 2, &which);
                if (p->ok && which && !strcmp(which, "else")) {
                    const char *endstop[] = { "endif" };
                    n->alt = parse_seq(p, endstop, 1, NULL);
                }
            } else {
                free(tag);
                p->ok = 0;
                break;
            }
            free(tag);
            if (!p->ok) break;
            cur->next = n;
            cur = n;
            continue;
        }
        if (p->pos + 1 < p->len && p->src[p->pos] == '{' &&
            p->src[p->pos + 1] == '{') {
            size_t start = p->pos + 2, end = start;
            node_t *n;
            while (end + 1 < p->len &&
                   !(p->src[end] == '}' && p->src[end + 1] == '}')) end++;
            if (end + 1 >= p->len) { p->ok = 0; break; }
            n = new_node(N_EXPR);
            if (!n) { p->ok = 0; break; }
            n->text = dup_trim(p->src + start, end - start);
            if (!n->text) { free(n); p->ok = 0; break; }
            p->pos = end + 2;
            cur->next = n;
            cur = n;
            continue;
        }
        {
            size_t start = p->pos, end = p->pos;
            node_t *n;
            while (end < p->len) {
                if (end + 1 < p->len && p->src[end] == '{' &&
                    (p->src[end + 1] == '{' || p->src[end + 1] == '%')) break;
                end++;
            }
            n = new_node(N_TEXT);
            if (!n) { p->ok = 0; break; }
            n->text = dup_raw(p->src + start, end - start);
            if (!n->text) { free(n); p->ok = 0; break; }
            p->pos = end;
            cur->next = n;
            cur = n;
        }
    }
    if (nstops > 0 && strcmp(stops[0], "__TOP__") != 0 && !found)
        p->ok = 0; /* unclosed for/if block */
    return head.next;
}

void *wubu_chat_template_compile(const char *text) {
    parser_t p;
    template_t *t;
    const char *top[] = { "__TOP__" };
    if (!text) return NULL;
    memset(&p, 0, sizeof(p));
    p.src = text;
    p.len = strlen(text);
    p.ok = 1;
    t = (template_t *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->root = parse_seq(&p, top, 1, NULL);
    if (!p.ok) {
        free_nodes(t->root);
        free(t);
        return NULL;
    }
    return t;
}

typedef struct {
    char *out;
    size_t cap, len;
    const wubu_chat_message_t *messages;
    int n_messages;
    const wubu_chat_message_t *cur;
    const char *var;
    const char *bos, *eos;
    int ok;
} render_ctx_t;

static void append(render_ctx_t *r, const char *s) {
    size_t n;
    if (!s || !r->ok) return;
    n = strlen(s);
    if (r->len + n + 1 > r->cap) { r->ok = 0; return; }
    memcpy(r->out + r->len, s, n);
    r->len += n;
    r->out[r->len] = '\0';
}

static void render_seq(render_ctx_t *r, node_t *n, int depth) {
    if (depth > 16) { r->ok = 0; return; }
    for (; n && r->ok; n = n->next) {
        if (n->kind == N_TEXT) {
            append(r, n->text);
        } else if (n->kind == N_EXPR) {
            const char *v = n->text;
            if (r->var && v) {
                size_t vl = strlen(r->var);
                if (!strncmp(v, r->var, vl) && v[vl] == '.') {
                    const char *field = v + vl + 1;
                    if (!strcmp(field, "role") && r->cur) append(r, r->cur->role);
                    else if (!strcmp(field, "content") && r->cur) append(r, r->cur->content);
                    continue;
                }
            }
            if (v && !strcmp(v, "bos")) append(r, r->bos);
            else if (v && !strcmp(v, "eos")) append(r, r->eos);
            /* Unknown expressions render empty. */
        } else if (n->kind == N_FOR) {
            const char *saved_var = r->var;
            const wubu_chat_message_t *saved_cur = r->cur;
            r->var = n->text;
            for (int i = 0; i < r->n_messages && r->ok; i++) {
                r->cur = &r->messages[i];
                render_seq(r, n->body, depth + 1);
            }
            r->var = saved_var;
            r->cur = saved_cur;
        } else if (n->kind == N_IF) {
            int take = 0;
            if (r->cur && r->cur->role && n->role)
                take = !strcmp(r->cur->role, n->role);
            render_seq(r, take ? n->body : n->alt, depth + 1);
        }
    }
}

int wubu_chat_template_render(void *tpl,
                              const wubu_chat_message_t *messages, int n_messages,
                              const char *bos, const char *eos,
                              char *out, size_t cap) {
    template_t *t = (template_t *)tpl;
    render_ctx_t r;
    if (!t || !out || cap == 0 || n_messages < 0 || (n_messages > 0 && !messages))
        return -1;
    memset(&r, 0, sizeof(r));
    r.out = out;
    r.cap = cap;
    r.messages = messages;
    r.n_messages = n_messages;
    r.bos = bos ? bos : "";
    r.eos = eos ? eos : "";
    r.ok = 1;
    out[0] = '\0';
    render_seq(&r, t->root, 0);
    return r.ok ? (int)r.len : -1;
}

void wubu_chat_template_free(void *tpl) {
    template_t *t = (template_t *)tpl;
    if (!t) return;
    free_nodes(t->root);
    free(t);
}
