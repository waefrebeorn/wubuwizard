/* test_eg02_tok_meta.c -- EG-02 gate: GGUF tokenizer metadata completeness.
 * Verifies the extractor reads tokenizer.ggml.pre, tokenizer.ggml.token_type,
 * tokenizer.ggml.add_bos_token and tokenizer.chat_template in addition to the
 * vocab/merge/special-id fields it already exposed. Prints: test_eg02_tok_meta: PASS. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_gguf_tokenizer.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    const char *path = "tests/fixtures/tok_meta.gguf";
    wubu_gguf_tokenizer_data_t d;
    CHECK(wubu_gguf_tokenizer_extract(path, &d));
    printf("EG-02 vocab=%d merges=%d bos=%d eos=%d pad=%d\n",
           d.n_tokens, d.n_merges, d.bos_id, d.eos_id, d.pad_id);
    CHECK(d.n_tokens == 3);
    CHECK(d.n_merges == 1);
    CHECK(d.bos_id == 1);
    CHECK(d.eos_id == 2);
    CHECK(strcmp(d.tokens[0], "<unk>") == 0);

    printf("EG-02 pre=%s token_type_n=%d add_bos=%d\n",
           d.pre_tokenizer ? d.pre_tokenizer : "(null)",
           d.n_token_type, d.add_bos);
    CHECK(d.pre_tokenizer != NULL);
    CHECK(d.pre_tokenizer && strcmp(d.pre_tokenizer, "llama3") == 0);
    CHECK(d.n_token_type == 3);
    CHECK(d.token_type != NULL);
    CHECK(d.token_type && d.token_type[0] == 1 &&
          d.token_type[1] == 2 && d.token_type[2] == 3);
    CHECK(d.add_bos == 1);

    printf("EG-02 chat_template=%s\n",
           d.chat_template ? d.chat_template : "(null)");
    CHECK(d.chat_template != NULL);
    CHECK(d.chat_template && strstr(d.chat_template, "{{") != NULL);

    /* Absent path must fail cleanly with zeroed output, no leaked pointers. */
    wubu_gguf_tokenizer_data_t empty;
    CHECK(wubu_gguf_tokenizer_extract(NULL, &empty) == false);
    CHECK(empty.tokens == NULL && empty.pre_tokenizer == NULL);

    wubu_gguf_tokenizer_free(&d);
    CHECK(d.pre_tokenizer == NULL && d.token_type == NULL &&
          d.chat_template == NULL);

    if (failures) {
        fprintf(stderr, "test_eg02_tok_meta: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg02_tok_meta: PASS\n");
    return 0;
}
