/* test_eg39_agentstate.c -- EG-39 gate: agent state resume, no re-prefill. */
#include <stdio.h>

#include "wubu_agent_state.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_agent_store_t s;
    int base[10], extended[12], diverged[4];
    int prefill = -1, turn = -1, code;
    int i;

    for (i = 0; i < 10; i++) base[i] = i + 1;
    for (i = 0; i < 12; i++) extended[i] = i + 1;
    diverged[0] = 9; diverged[1] = 8; diverged[2] = 7; diverged[3] = 6;

    wubu_agent_store_init(&s);
    CHECK(wubu_agent_state_save(&s, "agent-1", base, 10) == 0);
    CHECK(wubu_agent_slot(&s, "agent-1")->turn == 1);

    code = wubu_agent_state_resume(&s, "agent-1", base, 10, &prefill, &turn);
    printf("EG-39 exact code=%d prefill=%d turn=%d\n", code, prefill, turn);
    CHECK(code == 1 && prefill == 0 && turn == 1);

    code = wubu_agent_state_resume(&s, "agent-1", extended, 12,
                                   &prefill, &turn);
    printf("EG-39 extend code=%d prefill=%d turn=%d\n", code, prefill, turn);
    CHECK(code == 2 && prefill == 2 && turn == 2);

    code = wubu_agent_state_resume(&s, "agent-1", extended, 12,
                                   &prefill, &turn);
    CHECK(code == 1 && prefill == 0 && turn == 2);

    code = wubu_agent_state_resume(&s, "agent-1", diverged, 4,
                                   &prefill, &turn);
    printf("EG-39 diverged code=%d prefill=%d turn=%d\n", code, prefill, turn);
    CHECK(prefill == 4 && turn == 3);

    code = wubu_agent_state_resume(&s, "agent-2", base, 10,
                                   &prefill, &turn);
    CHECK(code == 0 && prefill == 10 && turn == 0);

    CHECK(wubu_agent_state_save(NULL, "x", base, 10) == -1);
    CHECK(wubu_agent_state_save(&s, "", base, 10) == -1);
    CHECK(wubu_agent_state_resume(&s, "agent-1", base, 0, &prefill, &turn) == -1);
    CHECK(wubu_agent_state_resume(&s, "agent-1", base, 10, NULL, &turn) == -1);

    if (failures) {
        fprintf(stderr, "test_eg39_agentstate: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg39_agentstate: PASS\n");
    return 0;
}
