/* wubu_ooxml.c -- Office text extraction (docx/xlsx/pptx, ours).
 *
 * ZIP + minimal XML scanning — no XML library, no licensing. We parse
 * the container (wubu_zip) and pull the text parts:
 *   - docx: word/document.xml, collect <w:t>...</w:t> content
 *   - xlsx: xl/sharedStrings.xml (all <t>), then xl/worksheets/sheet*.xml
 *   - pptx: ppt/slides/slide*.xml (<a:t>)
 * XML entity decode: &amp; &lt; &gt; &quot; &apos; &#NNN;
 *
 * C11.
 */
#include "wubu_ooxml.h"
#include "wubu_zip.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* collect the content of tags named `tag` (exact match on the element
 * name) from an XML buffer into the text output */
static void collect_tag_text(const unsigned char *xml, size_t n,
                             const char *tag, char *out, size_t outcap,
                             size_t *outlen, int para_break) {
    size_t tlen = strlen(tag);
    size_t i = 0;
    while (i < n && *outlen + 8 < outcap) {
        if (xml[i] != '<') { i++; continue; }
        /* is this <tag or <tag ...? */
        size_t j = i + 1;
        while (j < n && (isalnum(xml[j]) || xml[j] == ':')) j++;
        if (j - (i + 1) == tlen && memcmp(xml + i + 1, tag, tlen) == 0) {
            /* skip to the matching > (attributes) */
            while (j < n && xml[j] != '>') j++;
            if (j >= n) break;
            j++;   /* now at content */
            /* copy until </tag or < */
            while (j < n && xml[j] != '<' && *outlen + 4 < outcap) {
                if (xml[j] == '&') {
                    /* entity decode */
                    if (j + 5 <= n && memcmp(xml + j, "&amp;", 5) == 0) {
                        out[(*outlen)++] = '&'; j += 5;
                    } else if (j + 4 <= n && memcmp(xml + j, "&lt;", 4) == 0) {
                        out[(*outlen)++] = '<'; j += 4;
                    } else if (j + 4 <= n && memcmp(xml + j, "&gt;", 4) == 0) {
                        out[(*outlen)++] = '>'; j += 4;
                    } else if (j + 6 <= n && memcmp(xml + j, "&quot;", 6) == 0) {
                        out[(*outlen)++] = '"'; j += 6;
                    } else if (j + 6 <= n && memcmp(xml + j, "&apos;", 6) == 0) {
                        out[(*outlen)++] = '\''; j += 6;
                    } else if (j + 3 <= n && xml[j+1] == '#' &&
                               isdigit(xml[j+2])) {
                        int code = 0;
                        size_t k = j + 2;
                        while (k < n && isdigit(xml[k])) {
                            code = code * 10 + (xml[k] - '0');
                            k++;
                        }
                        if (k < n && xml[k] == ';' && code > 0 && code < 256)
                            out[(*outlen)++] = (char)code;
                        j = k + 1;
                    } else {
                        out[(*outlen)++] = xml[j++];
                    }
                } else {
                    out[(*outlen)++] = xml[j++];
                }
            }
        }
        i = j > i ? j : i + 1;
    }
    if (para_break && *outlen + 2 < outcap) {
        out[(*outlen)++] = '\n';
        out[*outlen] = '\0';
    }
}

/* extract text from one XML part with the given text-tag names */
static void xml_part_text(const unsigned char *part, size_t n,
                          const char *const *tags, int ntags,
                          char *out, size_t outcap, size_t *outlen) {
    for (int t = 0; t < ntags; t++)
        collect_tag_text(part, n, tags[t], out, outcap, outlen,
                         t == 0 ? 1 : 0);
}

const char *wubu_ooxml_detect(const unsigned char *data, size_t n) {
    wubu_zip_t *z = wubu_zip_open(data, n);
    if (!z) return NULL;
    const char *kind = NULL;
    for (size_t i = 0; i < wubu_zip_count(z); i++) {
        const char *nm = wubu_zip_entry(z, i)->name;
        if (strncmp(nm, "word/", 5) == 0) { kind = "docx"; break; }
        if (strncmp(nm, "xl/", 3) == 0)   { kind = "xlsx"; break; }
        if (strncmp(nm, "ppt/", 4) == 0)  { kind = "pptx"; break; }
    }
    wubu_zip_free(z);
    return kind;
}

char *wubu_ooxml_extract_text(const unsigned char *data, size_t n) {
    wubu_zip_t *z = wubu_zip_open(data, n);
    if (!z) return NULL;
    const char *kind = wubu_ooxml_detect(data, n);
    if (!kind) { wubu_zip_free(z); return NULL; }

    size_t cap = 64 * 1024;
    char *text = (char *)calloc(cap, 1);
    if (!text) { wubu_zip_free(z); return NULL; }
    size_t outlen = 0;

    static const char *docx_tags[] = {"w:t", "w:br"};
    static const char *pptx_tags[] = {"a:t", "a:br"};
    static const char *xlsx_tags[] = {"t"};

    if (strcmp(kind, "docx") == 0) {
        int idx = wubu_zip_find(z, "word/document.xml");
        if (idx >= 0) {
            unsigned char *part; size_t plen;
            if (wubu_zip_extract(z, data, n, (size_t)idx, &part, &plen) == 0) {
                xml_part_text(part, plen, docx_tags, 2, text, cap, &outlen);
                free(part);
            }
        }
    } else if (strcmp(kind, "xlsx") == 0) {
        /* shared strings first, then worksheet cells */
        int idx = wubu_zip_find(z, "xl/sharedStrings.xml");
        if (idx >= 0) {
            unsigned char *part; size_t plen;
            if (wubu_zip_extract(z, data, n, (size_t)idx, &part, &plen) == 0) {
                xml_part_text(part, plen, xlsx_tags, 1, text, cap, &outlen);
                free(part);
            }
        }
        for (size_t i = 0; i < wubu_zip_count(z); i++) {
            const char *nm = wubu_zip_entry(z, i)->name;
            if (strncmp(nm, "xl/worksheets/sheet", 19) == 0) {
                unsigned char *part; size_t plen;
                if (wubu_zip_extract(z, data, n, i, &part, &plen) == 0) {
                    xml_part_text(part, plen, xlsx_tags, 1, text, cap, &outlen);
                    free(part);
                }
            }
        }
    } else if (strcmp(kind, "pptx") == 0) {
        for (size_t i = 0; i < wubu_zip_count(z); i++) {
            const char *nm = wubu_zip_entry(z, i)->name;
            if (strncmp(nm, "ppt/slides/slide", 16) == 0) {
                unsigned char *part; size_t plen;
                if (wubu_zip_extract(z, data, n, i, &part, &plen) == 0) {
                    xml_part_text(part, plen, pptx_tags, 2, text, cap, &outlen);
                    free(part);
                }
            }
        }
    }
    wubu_zip_free(z);
    if (outlen == 0) { free(text); return NULL; }
    text[outlen] = '\0';
    return text;
}
