/* wubu_video.c -- Video frame extraction (ours: AVI container + MJPEG).
 *
 * AVI (RIFF) is a public container: 'RIFF' + 'AVI ' + LIST chunks.
 * The 'movi' LIST holds '00dc' (MJPEG) / '01wb' (audio) chunks.
 * MJPEG frames are complete JPEG bitstreams — our decoder handles
 * them. H.264 ('H264'/'x264'/'AVC1') frames are NAL units: we record
 * the codec and count frames (decode is a later leg; the file is
 * still ingested with metadata).
 *
 * C11.
 */
#include "wubu_video.h"
#include "wubu_jpeg.h"

#include <stdlib.h>
#include <string.h>

static uint32_t rd_le32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint32_t rd_be32(const unsigned char *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

int wubu_video_decode(const unsigned char *data, size_t n,
                      wubu_video_info_t *info,
                      float ***frames, int max_frames) {
    if (!data || !info || n < 12) return -1;
    memset(info, 0, sizeof(*info));
    if (frames) *frames = NULL;

    /* RIFF header */
    if (memcmp(data, "RIFF", 4) != 0 || memcmp(data + 8, "AVI ", 4) != 0)
        return -1;

    int width = 0, height = 0, fps_num = 0, fps_den = 0;
    char fourcc[8] = {0};
    int total_mjpeg = 0, total_nals = 0;
    int frame_cap = max_frames > 0 ? max_frames : 32;
    float **farr = (float **)calloc((size_t)frame_cap, sizeof(float *));
    if (!farr) return -1;
    int decoded = 0;

    /* walk the RIFF tree: find 'movi' LIST, read '00dc' chunks */
    size_t pos = 12;
    int in_movi = 0;
    size_t movi_end = 0;
    while (pos + 8 <= n) {
        uint32_t cid = rd_be32(data + pos);
        uint32_t csz = rd_le32(data + pos + 4);
        if (cid == 0x4C495354u) {          /* LIST */
            if (pos + 12 > n) break;
            uint32_t ltype = rd_be32(data + pos + 8);
            if (ltype == 0x6864726Cu) {    /* hdrl */
                /* parse the avih + strh for size/fps/codec */
                size_t p = pos + 12;
                size_t lend = pos + 8 + csz;
                while (p + 8 <= lend && p + 8 <= n) {
                    uint32_t hc = rd_be32(data + p);
                    uint32_t hs = rd_le32(data + p + 4);
                    if (p + 8 + hs > n) break;
                    if (hc == 0x61766968u && hs >= 56) {        /* avih */
                        uint32_t f = rd_le32(data + p + 8 + 20); /* dwRate */
                        uint32_t d = rd_le32(data + p + 8 + 24); /* dwScale */
                        if (d > 0) { fps_num = (int)f; fps_den = (int)d; }
                    } else if (hc == 0x73747268u && hs >= 48) { /* strh */
                        const unsigned char *sh = data + p + 8;
                        memcpy(fourcc, sh + 16, 4);             /* fccHandler */
                        fourcc[4] = '\0';
                    } else if (hc == 0x73747266u && hs >= 40) { /* strf */
                        const unsigned char *sf = data + p + 8;
                        width = (int)rd_le32(sf + 16);          /* biWidth */
                        height = (int)rd_le32(sf + 20);         /* biHeight */
                    }
                    p += 8 + hs + (hs & 1);
                }
            } else if (ltype == 0x6D6F7669u) {  /* movi */
                in_movi = 1;
                movi_end = pos + 8 + csz;
                if (movi_end > n) movi_end = n;
                /* walk INSIDE the movi LIST: its chunks start after the
                 * LIST header (8) + the 4-byte type */
                pos += 12;
                continue;
            }
        } else if (in_movi && (cid == 0x30646330u ||   /* 00dc: mjpeg */
                               cid == 0x30306463u)) { /* 00dc alternate */
            size_t cend = pos + 8 + csz;
            if (cend > movi_end && movi_end > 0) cend = movi_end;
            if (cend > n) cend = n;
            size_t dsz = cend - (pos + 8);
            total_mjpeg++;
            if (decoded < frame_cap && dsz > 4) {
                float *rgb = NULL;
                int w = 0, h = 0;
                if (wubu_jpeg_decode(data + pos + 8, dsz, &rgb, &w, &h) == 0) {
                    if (width == 0) { width = w; height = h; }
                    farr[decoded++] = rgb;
                }
            }
            pos = cend;
            continue;
        } else if (in_movi && (cid == 0x31637661u ||   /* avc1 */
                               cid == 0x34363248u ||   /* H264 */
                               cid == 0x34363278u ||   /* x264 */
                               cid == 0x31637664u)) {  /* dvc1 */
            total_nals++;
            pos += 8 + csz + (csz & 1);
            continue;
        }
        pos += 8 + csz + (csz & 1);
        if (in_movi && pos >= movi_end) break;
    }

    info->width = width;
    info->height = height;
    info->fps_num = fps_num;
    info->fps_den = fps_den ? fps_den : 1;
    memcpy(info->codec_fourcc, fourcc, 8);
    info->codec_mjpeg = (total_mjpeg > 0);
    info->n_frames = decoded;

    if (decoded == 0) {
        free(farr);
        if (frames) *frames = NULL;
        /* not an error: the container parsed, the codec is NAL-only or
         * unsupported (info is the deliverable) */
        return total_mjpeg + total_nals > 0 ? 0 : -1;
    }
    if (frames) *frames = farr;
    else {
        for (int i = 0; i < decoded; i++) free(farr[i]);
        free(farr);
    }
    return decoded;
}
