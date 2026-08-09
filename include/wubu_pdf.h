/* wubu_pdf.h -- PDF text extraction (ours: object scan + FlateDecode).
 *
 * PDF is a public format (ISO 32000): objects with streams, content
 * streams with text operators (Tj, TJ, '). We scan for stream objects,
 * inflate FlateDecode streams with zlib, and extract the text operators
 * ourselves. No licensing — the format is public and the code is ours.
 *
 * C11, self-contained (zlib only).
 */
#ifndef WUBU_PDF_H
#define WUBU_PDF_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Extract text from a PDF blob. Returns a malloc'd NUL-terminated text
 * buffer (caller frees) or NULL on failure. */
char *wubu_pdf_extract_text(const unsigned char *data, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_PDF_H */
