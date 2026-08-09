/* wubu_video.h -- Video frame extraction (ours: AVI container + MJPEG).
 *
 * The user's directive: "we need to get images and videos working...
 * I will take whatever reverse engineer online work exists."
 *
 * Video is the hardest format. Our approach (no codec licensing):
 *   - AVI container (RIFF) parse: OURS — the public format is trivial
 *     (LIST movi, '00dc' chunks = frames).
 *   - Motion-JPEG frames (AVI with MJPEG codec): each frame IS a JPEG,
 *     decoded by OUR wubu_jpeg decoder. MJPEG is the most common
 *     editable-video codec and needs zero licensing.
 *   - H.264/HEVC: extracted as NAL units (the container part, ours);
 *     the bitstream decode is a later leg (the file is still ingested,
 *     metadata recorded).
 *
 * The extraction contract: wubu_video_decode returns every frame as
 * RGB floats [0,1] (w*h*3 each), ready for our ViT image encoder —
 * the same shared embedding space as images.
 *
 * C11, self-contained (wubu_jpeg for MJPEG frames).
 */
#ifndef WUBU_VIDEO_H
#define WUBU_VIDEO_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int width, height;         /* frame size */
    int fps_num, fps_den;      /* frame rate */
    int n_frames;              /* frames extracted */
    int codec_mjpeg;           /* 1 if MJPEG (fully decoded), 0 if the
                                * codec is NAL-only (h264/hevc) */
    char codec_fourcc[8];
} wubu_video_info_t;

/* Decode an AVI file: extract up to max_frames frames. Every MJPEG
 * frame is decoded to RGB floats [0,1] and stored in *frames
 * (w*h*3 floats each; the caller frees each + the array). Returns the
 * frame count, or -1 on failure. Non-MJPEG codecs still report info
 * (n_frames = 0 frames decoded, codec_fourcc set). */
int wubu_video_decode(const unsigned char *data, size_t n,
                      wubu_video_info_t *info,
                      float ***frames, int max_frames);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_VIDEO_H */
