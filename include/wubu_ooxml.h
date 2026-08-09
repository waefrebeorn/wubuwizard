/* wubu_ooxml.h -- Office text extraction (docx/xlsx/pptx, ours).
 *
 * The office formats are ZIP archives of XML parts (ECMA-376):
 *   docx -> word/document.xml       (<w:t> runs)
 *   xlsx -> xl/sharedStrings.xml + xl/worksheets/sheet*.xml (<t> cells)
 *   pptx -> ppt/slides/slide*.xml   (<a:t> runs)
 * We extract the text with our ZIP reader + a minimal XML tag scanner
 * (ours, no XML library). The user's directive: "I have already gotten
 * an office program inside of our repository, but we will eventually
 * need to work in those other file types."
 *
 * C11, self-contained (wubu_zip + zlib).
 */
#ifndef WUBU_OOXML_H
#define WUBU_OOXML_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Extract text from a docx/xlsx/pptx blob. Returns a malloc'd
 * NUL-terminated text buffer (caller frees) or NULL on failure.
 * The text is the document content (paragraphs separated by \n). */
char *wubu_ooxml_extract_text(const unsigned char *data, size_t n);

/* Detect which office family the blob is (by ZIP entry names):
 * returns "docx", "xlsx", "pptx", or NULL. */
const char *wubu_ooxml_detect(const unsigned char *data, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_OOXML_H */
