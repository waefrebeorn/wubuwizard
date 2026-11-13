/*
 * wubu_tokens.h — Unified design tokens for CLI + GUI + themes.
 *
 * ADR-002 (wubuwizard): single source of truth for all visual identity.
 * Generated from research 066-J3 (design token file) + Theme J rank-3.
 */

#ifndef WUBU_TOKENS_H
#define WUBU_TOKENS_H

/* ============================================================
 * Color tokens (shared CLI ANSI + GUI hex + theme engine)
 * ============================================================ */

/* Primary accent — WuBu green (#00C853) */
#define WUBU_TOKEN_GREEN       0x00C853u
#define WUBU_TOKEN_GREEN_ANSI  "\x1b[38;2;0;200;83m"

/* Secondary — deep blue (#1A237E) */
#define WUBU_TOKEN_BLUE        0x1A237Eu
#define WUBU_TOKEN_BLUE_ANSI   "\x1b[38;2;26;35;126m"

/* Background — dark (#0D1117) */
#define WUBU_TOKEN_BG          0x0D1117u
#define WUBU_TOKEN_BG_ANSI     "\x1b[38;2;13;17;23m"

/* Surface — card background (#161B22) */
#define WUBU_TOKEN_SURFACE     0x161B22u
#define WUBU_TOKEN_SURFACE_ANSI "\x1b[38;2;22;27;34m"

/* Border — muted (#30363D) */
#define WUBU_TOKEN_BORDER      0x30363Du
#define WUBU_TOKEN_BORDER_ANSI "\x1b[38;2;48;54;61m"

/* Text primary — near-white (#E6EDF3) */
#define WUBU_TOKEN_TEXT        0xE6EDF3u
#define WUBU_TOKEN_TEXT_ANSI   "\x1b[38;2;230;237;243m"

/* Text secondary — muted (#8B949E) */
#define WUBU_TOKEN_TEXT_MUTED  0x8B949Eu
#define WUBU_TOKEN_TEXT_MUTED_ANSI "\x1b[38;2;139;148;158m"

/* Error — red (#F85149) */
#define WUBU_TOKEN_ERROR       0xF85149u
#define WUBU_TOKEN_ERROR_ANSI  "\x1b[38;2;248;81;73m"

/* Warning — amber (#D29922) */
#define WUBU_TOKEN_WARNING     0xD29922u
#define WUBU_TOKEN_WARNING_ANSI "\x1b[38;2;210;153;34m"

/* ============================================================
 * Layout tokens (shared CLI + GUI)
 * ============================================================ */

#define WUBU_TOKEN_BORDER_WIDTH      2
#define WUBU_TOKEN_TITLE_BAR_HEIGHT  24
#define WUBU_TOKEN_PADDING_X         4
#define WUBU_TOKEN_PADDING_Y         4
#define WUBU_TOKEN_BUTTON_SPACING    2

/* ============================================================
 * CLI-specific tokens
 * ============================================================ */

#define WUBU_TOKEN_BANNER_WIDTH      60
#define WUBU_TOKEN_SECTION_WIDTH     60

/* ANSI escape helpers */
#define WUBU_TOKEN_RESET       "\x1b[0m"
#define WUBU_TOKEN_BOLD        "\x1b[1m"
#define WUBU_TOKEN_DIM         "\x1b[2m"

/* ============================================================
 * GUI-specific: theme color resolver
 * ============================================================ */

/* Maps a logical color name to a theme-adjusted hex value.
 * In the GUI this goes through the theme engine; in the CLI
 * it maps to the nearest ANSI 24-bit escape. */
static inline unsigned int wubu_token_color(const char *name) {
    if (name[0]=='g' && name[1]=='r') return WUBU_TOKEN_GREEN;   /* "green"   */
    if (name[0]=='b' && name[1]=='l') return WUBU_TOKEN_BLUE;    /* "blue"    */
    if (name[0]=='b' && name[1]=='g') return WUBU_TOKEN_BG;      /* "bg"      */
    if (name[0]=='t' && name[1]=='e') return WUBU_TOKEN_TEXT;    /* "text"    */
    if (name[0]=='m') return WUBU_TOKEN_TEXT_MUTED;              /* "muted"   */
    if (name[0]=='e') return WUBU_TOKEN_ERROR;                   /* "error"   */
    return WUBU_TOKEN_TEXT;
}

#endif /* WUBU_TOKENS_H */

/* ---- Layout tokens --------------------------------------------------
 *
 * There was no layout scale here at all when the audit ran: every surface
 * invented its own margins (2, 4, 6, 7, 8, 10, 12, 20 appear independently
 * across the shell), which is why windows, menus and the taskbar disagree by a
 * pixel or two everywhere.  A spacing system only works if it has ONE owner,
 * so these are the numbers and nothing else may hardcode geometry.
 *
 * The grid is 4px.  Anything not on it needs a reason written down.
 */

#define WUBU_SPACE_2XS    2
#define WUBU_SPACE_XS     4
#define WUBU_SPACE_SM     8
#define WUBU_SPACE_MD     12
#define WUBU_SPACE_LG     16
#define WUBU_SPACE_XL     24
#define WUBU_SPACE_2XL    32

/* Corner radii.  Win11 uses 8 at 100% scale for windows and menus. */
#define WUBU_RADIUS_NONE  0
#define WUBU_RADIUS_SM    4
#define WUBU_RADIUS_MD    8
#define WUBU_RADIUS_LG    12

/* Borders. */
#define WUBU_BORDER_HAIRLINE  1
#define WUBU_BORDER_THIN      2

/* ---- Control metrics (Windows 11 at 100% scale) ---- */

#define WUBU_CONTROL_H     32   /* a standard button / list row */
#define WUBU_ICON_SM       16   /* title-bar and menu icon */
#define WUBU_ICON_MD       24   /* desktop icon glyph */
#define WUBU_ICON_LG       32

#define WUBU_TITLE_BAR_H   32   /* Win11 caption height */
#define WUBU_TITLE_BTN_W   46   /* caption button width */
#define WUBU_TITLE_BTN_H   32   /* flush with the caption bar */
#define WUBU_TITLE_PAD     WUBU_SPACE_SM   /* icon and title inset */

#define WUBU_TASKBAR_H     48   /* Win11 taskbar height */
#define WUBU_MENU_PAD      WUBU_SPACE_MD   /* menu content inset */
#define WUBU_MENU_ROW_H    26

/* Type sizes (the VBE draws at a fixed 8px cell; these are cap heights used
 * for vertical centring, not point sizes). */
#define WUBU_FONT_CAP_H    8
#define WUBU_FONT_BODY_H   8

