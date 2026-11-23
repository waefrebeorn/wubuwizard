/* test_eg14_template.c -- EG-14 gate: Jinja-subset chat template renderer.
 * Prints: test_eg14_template: PASS. */
#include <stdio.h>
#include <string.h>

#include "wubu_chat_template.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    const char *tmpl =
        "{% for m in messages %}"
        "{% if m.role == 'user' %}User: {{ m.content }}\n"
        "{% else %}Assistant: {{ m.content }}\n"
        "{% endif %}{% endfor %}";
    void *t = wubu_chat_template_compile(tmpl);
    CHECK(t != NULL);
    if (!t) return 1;

    wubu_chat_message_t msgs[2] = {
        { "user", "Hi" },
        { "assistant", "Hello" }
    };
    char out[256];
    int n = wubu_chat_template_render(t, msgs, 2, "", "", out, sizeof(out));
    printf("EG-14 rendered=%d [%s]\n", n, out);
    CHECK(n == (int)strlen("User: Hi\nAssistant: Hello\n"));
    CHECK(strcmp(out, "User: Hi\nAssistant: Hello\n") == 0);

    /* bos/eos globals and a simple loop. */
    void *t2 = wubu_chat_template_compile(
        "{{ bos }}{% for m in messages %}{{ m.content }}{% endfor %}{{ eos }}");
    CHECK(t2 != NULL);
    if (t2) {
        wubu_chat_message_t one[1] = { { "user", "X" } };
        n = wubu_chat_template_render(t2, one, 1, "<bos>", "<eos>",
                                       out, sizeof(out));
        printf("EG-14 globals=[%s]\n", out);
        CHECK(n == (int)strlen("<bos>X<eos>"));
        CHECK(strcmp(out, "<bos>X<eos>") == 0);
    }

    /* Fail closed on malformed templates and bad arguments. */
    CHECK(wubu_chat_template_compile("{% for m in messages %}x") == NULL);
    CHECK(wubu_chat_template_compile(NULL) == NULL);
    CHECK(wubu_chat_template_render(t, msgs, 2, "", "", out, 4) == -1);
    CHECK(wubu_chat_template_render(NULL, msgs, 2, "", "", out, sizeof(out)) == -1);
    CHECK(wubu_chat_template_render(t, NULL, 2, "", "", out, sizeof(out)) == -1);
    CHECK(wubu_chat_template_render(t, msgs, -1, "", "", out, sizeof(out)) == -1);

    wubu_chat_template_free(t);
    wubu_chat_template_free(t2);
    if (failures) {
        fprintf(stderr, "test_eg14_template: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg14_template: PASS\n");
    return 0;
}
