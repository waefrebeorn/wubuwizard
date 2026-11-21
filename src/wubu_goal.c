/* wubu_goal.c -- persistent goals with an evidence-first judge.
 * C11, no third-party deps, status-as-value. See include/wubu_goal.h. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_goal.h"

#define WUBU_GOAL_TEXT_MAX 512
#define WUBU_GOAL_NOTE_MAX 128
#define WUBU_GOAL_MAGIC    "WGOAL1\0"
#define WUBU_GOAL_VERSION  1

struct wubu_goal {
    char text[WUBU_GOAL_TEXT_MAX];
    char note[WUBU_GOAL_NOTE_MAX];
    int turns;
    int max_turns;
    wubu_goal_status_t status;
};

static int has_token(const char *hay, const char *needle)
{
    return hay && needle && strstr(hay, needle) != NULL;
}

static void note_set(wubu_goal_t *g, const char *src)
{
    g->note[0] = '\0';
    if (!src) return;
    size_t n = strlen(src);
    if (n >= sizeof(g->note)) n = sizeof(g->note) - 1;
    memcpy(g->note, src, n);
    g->note[n] = '\0';
}

wubu_goal_t *wubu_goal_create(const char *text, int max_turns)
{
    if (!text || !text[0] || max_turns < 1) return NULL;
    wubu_goal_t *g = (wubu_goal_t *)calloc(1, sizeof(*g));
    if (!g) return NULL;
    if (wubu_goal_set_text(g, text) != 0) { free(g); return NULL; }
    g->max_turns = max_turns;
    g->status = WUBU_GOAL_ACTIVE;
    return g;
}

void wubu_goal_free(wubu_goal_t *g) { free(g); }

int wubu_goal_set_text(wubu_goal_t *g, const char *text)
{
    if (!g || !text || !text[0]) return -1;
    size_t n = strlen(text);
    if (n >= sizeof(g->text)) n = sizeof(g->text) - 1;
    memcpy(g->text, text, n);
    g->text[n] = '\0';
    return 0;
}

const char *wubu_goal_text(const wubu_goal_t *g) { return g ? g->text : NULL; }
int wubu_goal_turns(const wubu_goal_t *g) { return g ? g->turns : -1; }
int wubu_goal_max_turns(const wubu_goal_t *g) { return g ? g->max_turns : -1; }
wubu_goal_status_t wubu_goal_status(const wubu_goal_t *g)
{
    return g ? g->status : WUBU_GOAL_BLOCKED;
}
const char *wubu_goal_note(const wubu_goal_t *g) { return g ? g->note : NULL; }

wubu_goal_verdict_t wubu_goal_judge(wubu_goal_t *g, const char *response,
                                    const char *evidence,
                                    wubu_goal_verify_fn verify, void *userdata)
{
    if (!g || !response) return WUBU_GOAL_INVALID;
    if (g->status != WUBU_GOAL_ACTIVE) {
        if (g->status == WUBU_GOAL_DONE) return WUBU_GOAL_FINISHED;
        if (g->status == WUBU_GOAL_BLOCKED) return WUBU_GOAL_IS_BLOCKED;
        return WUBU_GOAL_IS_PAUSED;
    }

    g->turns++;
    if (has_token(response, "[BLOCKED")) {
        const char *p = strstr(response, "[BLOCKED");
        const char *close = p ? strchr(p, ']') : NULL;
        if (close) {
            const char *colon = strchr(p, ':');
            if (colon && colon < close) {
                char tmp[WUBU_GOAL_NOTE_MAX];
                const char *s = colon + 1;
                while (*s == ' ' || *s == '\t') s++;
                size_t n = (size_t)(close - s);
                if (n >= sizeof(tmp)) n = sizeof(tmp) - 1;
                memcpy(tmp, s, n);
                tmp[n] = '\0';
                note_set(g, tmp);
            }
        }
        g->status = WUBU_GOAL_BLOCKED;
        return WUBU_GOAL_IS_BLOCKED;
    }

    /* Report isolation: a self-declared [GOAL_DONE] is not evidence. */
    if (has_token(response, "[GOAL_DONE]") &&
        has_token(evidence, "[PASS]") &&
        verify && verify(g->text, evidence ? evidence : "", userdata) > 0) {
        g->status = WUBU_GOAL_DONE;
        return WUBU_GOAL_FINISHED;
    }

    if (g->turns >= g->max_turns) {
        note_set(g, "turn budget exhausted");
        g->status = WUBU_GOAL_PAUSED;
        return WUBU_GOAL_IS_PAUSED;
    }
    return WUBU_GOAL_CONTINUE;
}

int wubu_goal_preempt(wubu_goal_t *g)
{
    if (!g) return -1;
    if (g->status == WUBU_GOAL_ACTIVE) {
        note_set(g, "preempted by user message");
        g->status = WUBU_GOAL_PAUSED;
    }
    return 0;
}

int wubu_goal_should_continue(const wubu_goal_t *g)
{
    return g && g->status == WUBU_GOAL_ACTIVE && g->turns < g->max_turns;
}

const char *wubu_goal_continuation_prompt(const wubu_goal_t *g)
{
    static char buf[WUBU_GOAL_TEXT_MAX + 256];
    if (!g) return "";
    snprintf(buf, sizeof(buf),
             "Continue working toward the standing goal: %s\n"
             "Do not stop until the goal is achieved or explicitly blocked. "
             "Provide an evidence receipt with [PASS] and finish with [GOAL_DONE].",
             g->text);
    return buf;
}

int wubu_goal_save(const wubu_goal_t *g, const char *path)
{
    if (!g || !path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int ok = 1;
    ok &= fwrite(WUBU_GOAL_MAGIC, 1, 7, f) == 7;
    int32_t version = WUBU_GOAL_VERSION;
    ok &= fwrite(&version, sizeof(version), 1, f) == 1;
    ok &= fwrite(g->text, 1, sizeof(g->text), f) == sizeof(g->text);
    ok &= fwrite(g->note, 1, sizeof(g->note), f) == sizeof(g->note);
    ok &= fwrite(&g->turns, sizeof(g->turns), 1, f) == 1;
    ok &= fwrite(&g->max_turns, sizeof(g->max_turns), 1, f) == 1;
    int32_t st = (int32_t)g->status;
    ok &= fwrite(&st, sizeof(st), 1, f) == 1;
    if (fclose(f) != 0) ok = 0;
    return ok ? 0 : -1;
}

wubu_goal_t *wubu_goal_load(const char *path)
{
    if (!path) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[7];
    int32_t version = 0, st = 0;
    wubu_goal_t *g = (wubu_goal_t *)calloc(1, sizeof(*g));
    int ok = g != NULL;
    if (ok) ok = fread(magic, 1, 7, f) == 7 && memcmp(magic, WUBU_GOAL_MAGIC, 7) == 0;
    if (ok) ok = fread(&version, sizeof(version), 1, f) == 1 && version == WUBU_GOAL_VERSION;
    if (ok) ok = fread(g->text, 1, sizeof(g->text), f) == sizeof(g->text);
    if (ok) ok = fread(g->note, 1, sizeof(g->note), f) == sizeof(g->note);
    if (ok) ok = fread(&g->turns, sizeof(g->turns), 1, f) == 1;
    if (ok) ok = fread(&g->max_turns, sizeof(g->max_turns), 1, f) == 1;
    if (ok) ok = fread(&st, sizeof(st), 1, f) == 1;
    if (ok && (g->text[0] == '\0' || g->turns < 0 || g->max_turns < 1 ||
               st < WUBU_GOAL_ACTIVE || st > WUBU_GOAL_BLOCKED)) ok = 0;
    if (ok) g->status = (wubu_goal_status_t)st;
    fclose(f);
    if (!ok) { free(g); return NULL; }
    return g;
}
