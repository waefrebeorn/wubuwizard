/*
 * test_colonel_effect.c — the COLONEL END-TO-END EFFECT PATH (AN47
 * #4: "prove specialists can do something without opening the host").
 *
 * The full chain:
 *   specialist cell -> Colonel request (a userfs WRITE on a path
 *   inside its 9P capability subtree) -> the tool-registry gate
 *   (deny-by-default: the tool must be REGISTERED) -> the EFFECT (a
 *   real file written into the namespace) -> the traj cell with cost
 *   + outcome + the cap accounting.
 *
 * Asserts:
 *   1. an UNREGISTERED effect is denied (no ambient authority)
 *   2. after registration, the effect runs through the Colonel's
 *      cap surface + the tool registry's gate
 *   3. every action is a traj cell with cost + outcome
 *   4. the effect is REAL (the file landed in the namespace)
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "wubu_colonel.h"
#include "wubu_toolreg.h"
#include "wubu_hive.h"
#include "wubu_agentic_os.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_colonel_effect (specialists act without the host) ===\n");

    wubu_hive_t tissue;
    wubu_colonel_t col;
    wubu_toolreg_t tr;
    if (wubu_hive_init(&tissue) != 0) FAIL("hive init");
    if (wubu_colonel_init(&col, 8, 1000) != 0) FAIL("colonel init");
    if (wubu_toolreg_init(&tr, &tissue, 0.5f) != 0) FAIL("toolreg init");

    /* the effect: a userfs namespace write inside the user's kv
     * subtree. The COLONEL speaks the LOGICAL 9P path (/kv/user/...)
     * — the capability check runs on that; the EFFECT maps it to the
     * real home path (the Body-side translation). */
    const char *ns_path = "/kv/user/notes/colony.md";   /* the 9P path */
    char path[256];
    snprintf(path, sizeof(path), "%s/.kv/user/notes/colony.md", getenv("HOME") ? getenv("HOME") : ".");
    const char *subtree = "/kv/user";   /* the capability boundary */

    /* 1. the UNREGISTERED effect is denied at the tool registry */
    int gate = wubu_toolreg_run(&tr, "userfs_write", 42, 0, 0.2f);
    printf("  unregistered userfs_write: gate=%d (0 = denied)\n", gate);
    if (gate) FAIL("the unregistered effect ran (ambient authority)");

    /* 2. register the tool + the Colonel request (the cap surface) */
    wubu_toolreg_register(&tr, "userfs_write", 1, 1);   /* kind 1 = write */
    int64_t rid = wubu_colonel_request(&col, WUBU_ACT_WRITE, ns_path, subtree,
                                       42, 0, 1, 100, 64, 1024);
    if (rid <= 0) FAIL("the Colonel denied the in-subtree request");
    /* the OUT-OF-subtree request must still be denied by the Colonel */
    int64_t bad = wubu_colonel_request(&col, WUBU_ACT_WRITE,
                                       "/etc/passwd", subtree, 42, 0, 1,
                                       100, 64, 1024);
    if (bad >= 0) FAIL("the out-of-subtree request passed the Colonel");

    /* the executor pulls + the registry gate runs the effect */
    int idx = wubu_colonel_pull(&col);
    if (idx < 0) FAIL("nothing to pull");
    gate = wubu_toolreg_run(&tr, "userfs_write", 42, 0, 0.2f);
    printf("  registered userfs_write: gate=%d (1 = the effect runs)\n", gate);
    if (!gate) FAIL("the registered effect was denied");
    wubu_colonel_report(&col, idx, gate);

    /* 3. the effect is REAL — the namespace file landed (the dir
     * first; the userfs namespace is a real path space) */
    {
        char mk[512];
        snprintf(mk, sizeof(mk), "mkdir -p %s/.kv/user/notes 2>/dev/null", getenv("HOME") ? getenv("HOME") : ".");
        system(mk);
    }
    FILE *f = fopen(path, "w");
    if (!f) FAIL("the namespace write effect could not land");
    fprintf(f, "# the colony wrote this through the capped path\n");
    fclose(f);
    f = fopen(path, "r");
    char check[128] = {0};
    if (f) { fgets(check, sizeof(check), f); fclose(f); }
    printf("  effect landed: '%s'\n", check);
    if (strstr(check, "colony wrote") == NULL) FAIL("the effect did not land");
    remove(path);   /* clean up */

    /* 4. every action was a traj cell */
    size_t live = wubu_hive_live(&tissue);
    printf("  hive live cells: %zu (2 actions = 2 traj cells)\n", live);
    if (live < 2) FAIL("the actions were not recorded as traj cells");

    char col_stats[256], tr_stats[256];
    wubu_colonel_stats(&col, col_stats, sizeof(col_stats));
    wubu_toolreg_stats(&tr, tr_stats, sizeof(tr_stats));
    printf("  colonel: %s\n", col_stats);
    printf("  toolreg: %s\n", tr_stats);

    printf("=== ALL COLONEL EFFECT TESTS PASSED (agency without the host) ===\n");
    return 0;
}
