/* wubu_optim.h — the unified optimizer dispatch (the plumbing of the
 * optimizer research, research/069).
 *
 * The user's research (2026-08-09): the optimizer must SPLIT by geometry.
 *   - Muon/NS5  : the flat (Euclidean) matrices  — wubu_bp_muon_step
 *   - AdamW     : norms / embeddings / selectors — adamw_update
 *   - RSGD      : the NON-Euclidean (Poincaré-ball) vectors — rsgd_step
 *   - qlearner  : the adaptive learning RATE (Q-learns from the loss)
 *   - TGT       : the gravitational-pole ODOMETER — tracks the update
 *                 trajectory in polar coordinates around loss-landscape
 *                 basins; pulls INTO ravines, ejects OUT of traps.
 *
 * KEY INSIGHT (user): the RSGD/PID/TGT family also works in Euclidean
 * space via the optimized-SGD form (retraction -> GD as R -> inf).
 * Muon remains the flat-matrix champ; TGT is the ravine navigator.
 */
#ifndef WUBU_OPTIM_H
#define WUBU_OPTIM_H

#include "wubu_train.h"
#include "rsgd.h"
#include "qlearner.h"

/* Which optimizer handles a parameter class */
typedef enum {
    WUBU_OPT_MUON = 0,    /* 2D Euclidean matrices (the flat skeleton) */
    WUBU_OPT_ADAMW,       /* 1-D norms / embeddings / selectors */
    WUBU_OPT_RSGD,        /* Poincaré-ball vectors (the curved cells) */
    WUBU_OPT_COUNT
} wubu_optim_kind_t;

/* The TGT odometer state (the gravitational-pole tracker).
 * Tracks the update trajectory in the loss landscape so the optimizer
 * knows entering-a-ravine (curvature rising) from stuck-in-one. */
typedef struct {
    float r;          /* polar radius around the current pole/basin */
    float theta;      /* polar angle of the update direction */
    float vel;        /* velocity along the trajectory (dr/dt) */
    float acc;        /* curvature proxy (d²loss/dstep², smoothed) */
    float loss_prev;  /* previous loss (for the curvature proxy) */
    int   loss_prev_ok;
    int   in_ravine;  /* 1 = the odometer believes we're inside a ravine */
    int   steps_in;   /* steps spent in the current basin */
} wubu_tgt_t;

/* The unified optimizer state (one per training run). */
typedef struct {
    wubu_optim_kind_t kind;
    qlearner_t ql;            /* the LR Q-learner (0 = disabled) */
    wubu_tgt_t tgt;           /* the TGT odometer (0 = disabled) */
    int use_qlearner;         /* 1 = the qlearner adapts the LR */
    int use_tgt;              /* 1 = the TGT odometer tracks the run */
    float rsgd_radius;        /* Poincaré ball radius R (default 1.0) */
    float rsgd_clip;          /* gradient clip (default 1.0) */
    float lr;                 /* the (possibly qlearner-tuned) LR */
    float loss_prev;          /* previous loss for the TGT curvature */
} wubu_optim_t;

/* Initialize the optimizer for a kind (call once per run). */
void wubu_optim_init(wubu_optim_t *o, wubu_optim_kind_t kind,
                     int use_qlearner, int use_tgt);

/* One training step on the whole model:
 *  - routes each param class to its optimizer (Muon/AdamW/RSGD),
 *  - feeds the loss to the qlearner (adaptive LR) + TGT odometer,
 *  - returns the (possibly adapted) LR for the caller's bookkeeping.
 * m/tr: the model + train state; loss: the current step's loss. */
float wubu_optim_step(wubu_optim_t *o, wubu_model_t *m, wubu_train_t *tr,
                      float loss, float muon_lr, float adam_lr);

/* The TGT odometer alone: feed it the loss + the update's polar
 * coordinates and it returns the trajectory radius r (the ravine
 * detector). Call before the optimizer step when driving RSGD. */
float wubu_tgt_update(wubu_tgt_t *t, float loss, float theta);

#endif
