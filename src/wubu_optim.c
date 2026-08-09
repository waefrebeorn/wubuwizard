/* wubu_optim.c — the unified optimizer dispatch (the plumbing).
 * See include/wubu_optim.h for the design (research/069).
 */
#include "wubu_optim.h"
#include "wubu_backprop.h"
#include <math.h>
#include <string.h>

void wubu_optim_init(wubu_optim_t *o, wubu_optim_kind_t kind,
                     int use_qlearner, int use_tgt)
{
    if (!o) return;
    memset(o, 0, sizeof(*o));
    o->kind = kind;
    o->use_qlearner = use_qlearner;
    o->use_tgt = use_tgt;
    o->rsgd_radius = 1.0f;
    o->rsgd_clip = 1.0f;
    o->lr = 1e-4f;
    o->loss_prev = -1.0f;
    if (use_qlearner) qlearner_init(&o->ql);
    if (use_tgt) {
        o->tgt.r = 1.0f;
        o->tgt.theta = 0.0f;
        o->tgt.vel = 0.0f;
        o->tgt.acc = 0.0f;
        o->tgt.in_ravine = 0;
        o->tgt.steps_in = 0;
    }
}

float wubu_tgt_update(wubu_tgt_t *t, float loss, float theta)
{
    if (!t) return 1.0f;
    float prev = t->loss_prev;
    /* the curvature proxy: how fast the loss is changing now vs before */
    if (t->loss_prev_ok) {
        float dl = loss - prev;
        t->acc = 0.9f * t->acc + 0.1f * (dl > 0 ? dl : -dl);
    }
    /* velocity along the trajectory: the loss drop per step (uses the
     * OLD loss — compute BEFORE overwriting loss_prev) */
    t->vel = (t->loss_prev_ok) ? (prev - loss) : 0.0f;
    t->loss_prev = loss;
    t->loss_prev_ok = 1;
    /* the polar frame: theta is the update direction (caller-provided);
     * the radius r shrinks as we fall toward the basin center. */
    t->theta = theta;
    if (t->acc < 1e-4f) {
        /* flat region — not in a ravine */
        t->in_ravine = 0;
        t->steps_in = 0;
        t->r = 1.0f;
    } else if (t->vel > 0.0f) {
        /* loss still dropping fast — deep inside the ravine, keep falling */
        t->in_ravine = 1;
        t->steps_in++;
        t->r = t->r * 0.95f;   /* falling toward the pole */
    } else {
        /* loss flat but curvature high — a ravine FLOOR / saddle:
         * the trap. The odometer marks it so the caller can eject. */
        t->steps_in++;
        if (t->steps_in > 8) {
            t->in_ravine = 1;   /* stuck — the anti-gravity kick is due */
            t->r = 2.0f;        /* push OUT of the ravine */
        }
    }
    return t->r;
}

float wubu_optim_step(wubu_optim_t *o, wubu_model_t *m, wubu_train_t *tr,
                      float loss, float muon_lr, float adam_lr)
{
    if (!o) return muon_lr;

    wubu_train_cfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.muon_lr = muon_lr;
    cfg.adam_lr = adam_lr;
    cfg.weight_decay = 0.1f;
    cfg.grad_clip = 1.0f;
    cfg.muon_momentum = 0.95f;

    /* the qlearner: adapt the LR from the loss signal (the metabolism) */
    if (o->use_qlearner) {
        float new_lr = qlearner_step(&o->ql, loss);
        if (new_lr > 0.0f) {
            o->lr = new_lr;
            cfg.muon_lr = o->lr;
            cfg.adam_lr = o->lr;
        }
    }

    /* the TGT odometer: track the trajectory (theta from the loss sign) */
    {
        float theta = (loss < o->loss_prev || o->loss_prev < 0) ? 0.0f : 3.14159f;
        wubu_tgt_update(&o->tgt, loss, theta);
        /* if the odometer says we're stuck (r pushed to 2.0), give the
         * update an anti-gravity kick: bump the LR briefly */
        if (o->tgt.r > 1.5f) {
            cfg.muon_lr *= 1.5f;
            cfg.adam_lr *= 1.5f;
        }
    }
    o->loss_prev = loss;

    /* the actual optimizer step needs the model + train state */
    if (!m || !tr) return cfg.muon_lr;

    /* the geometry split (the core): Muon for the flat matrices (the
     * whole S7 body today), AdamW for the 1-D params. When the
     * hyperbolic cells wire in (RSGD kind), their vectors route to
     * rsgd_step instead — the flat skeleton Muons, the curved cells
     * roll. */
    switch (o->kind) {
        case WUBU_OPT_MUON:
            /* the full BP4 optimizer: Muon matrices + AdamW 1-D */
            wubu_bp_muon_step(m, tr, &cfg, (uint32_t)(tr->micro_steps + 1));
            break;
        case WUBU_OPT_ADAMW:
            /* AdamW only (norms/embeddings/selectors) */
            cfg.muon_lr = 0.0f;
            wubu_bp_muon_step(m, tr, &cfg, (uint32_t)(tr->micro_steps + 1));
            break;
        case WUBU_OPT_RSGD:
            /* RSGD for the non-Euclidean params — the caller must have
             * staged the Poincaré vectors in the model (wired when the
             * hyperbolic cells land). The flat part still Muons. */
            wubu_bp_muon_step(m, tr, &cfg, (uint32_t)(tr->micro_steps + 1));
            break;
        default:
            break;
    }
    return cfg.muon_lr;
}
