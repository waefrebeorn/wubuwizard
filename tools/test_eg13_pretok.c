/* test_eg13_pretok.c -- EG-13 gate: pre-tokenizer family registry and NFC
 * fallback. Prints: test_eg13_pretok: PASS. */
#include <stdio.h>
#include <string.h>

#include "wubu_pretoken.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

static int pieces(const wubu_pretoken_t *p, const char *fam,
                  const char *text, size_t off[16], size_t len[16]) {
    return wubu_pretoken_split(p, fam, text, strlen(text), off, len, 16);
}

int main(void) {
    wubu_pretoken_t *p = wubu_pretoken_create(8);
    CHECK(p != NULL);
    if (!p) return 1;
    const char *fams[] = { "default", "llama3", "gpt2", "qwen2", "deepseek" };
    for (size_t i = 0; i < sizeof(fams)/sizeof(fams[0]); i++) {
        CHECK(wubu_pretoken_register_builtin(p, fams[i]) == 0);
        CHECK(wubu_pretoken_supported(p, fams[i]) == 1);
    }
    CHECK(wubu_pretoken_supported(p, "unknown") == 0);
    CHECK(wubu_pretoken_supported(NULL, "llama3") == 0);

    size_t off[16], len[16];
    int n_default = pieces(p, "default", "Hi, world!", off, len);
    int n_llama = pieces(p, "llama3", "Hi, world!", off, len);
    printf("EG-13 pieces: default=%d llama3=%d\n", n_default, n_llama);
    CHECK(n_default == 2);
    CHECK(n_llama == 4);
    for (int i = 0; i < n_llama; i++) {
        CHECK(off[i] + len[i] <= strlen("Hi, world!"));
    }
    CHECK(pieces(p, "unknown", "Hi", off, len) == -1);

    /* Family dispatch is observable on mixed letter/digit input. */
    int q = pieces(p, "qwen2", "abc123", off, len);
    int l = pieces(p, "llama3", "abc123", off, len);
    int d = pieces(p, "deepseek", "abc12", off, len);
    printf("EG-13 abc123: qwen2=%d llama3=%d deepseek(abc12)=%d\n", q, l, d);
    CHECK(q == 2);
    CHECK(l == 1);
    CHECK(d == 3);

    /* NFC composition of decomposed acute. */
    const char decomposed[] = { 'e', (char)0xCC, (char)0x81, 0 };
    unsigned char out[32];
    int n = wubu_pretoken_nfc(decomposed, 3, out, sizeof(out));
    printf("EG-13 NFC bytes=%d", n);
    for (int i = 0; i < n; i++) printf(" %02X", out[i]);
    printf("\n");
    CHECK(n == 2);
    CHECK(out[0] == 0xC3 && out[1] == 0xA9); /* é */

    /* Unknown combining mark and too-small buffer fail safely. */
    const char unknown[] = { 'e', (char)0xCC, (char)0x88, 0 };
    n = wubu_pretoken_nfc(unknown, 3, out, sizeof(out));
    CHECK(n == 3);
    CHECK(out[0] == 'e' && out[1] == 0xCC && out[2] == 0x88);
    CHECK(wubu_pretoken_nfc(decomposed, 3, out, 1) == -1);
    CHECK(wubu_pretoken_nfc(NULL, 3, out, sizeof(out)) == -1);
    CHECK(wubu_pretoken_nfc(decomposed, 3, NULL, sizeof(out)) == -1);

    CHECK(wubu_pretoken_create(0) == NULL);
    CHECK(wubu_pretoken_register_builtin(p, "unknown") == -1);
    wubu_pretoken_free(p);
    if (failures) {
        fprintf(stderr, "test_eg13_pretok: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg13_pretok: PASS\n");
    return 0;
}
