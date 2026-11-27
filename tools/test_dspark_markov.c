/* test_dspark_markov.c -- validate dspark_joint_proj against the reference.
 * Reads Jw.bin (768x5120 f32 row-major), b.bin (768), x.bin (5120),
 * y_ref.bin (768) from dspark_joint-1-0, prints correlation. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dspark_markov.c"

static float *readbin(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "no %s\n", p); exit(2); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    *n = (size_t)(sz/4); float *a = malloc((size_t)sz); 
    if (fread(a, 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr,"read %s\n",p); exit(2); }
    fclose(f); return a;
}
int main(int argc, char**argv) {
    const char *dir = argc>1 ? argv[1] : ".";
    char p[512];
    size_t nj, nb, nx, ny;
    snprintf(p, sizeof p, "%s/Jw.bin", dir);    float *J = readbin(p, &nj);
    snprintf(p, sizeof p, "%s/b.bin",  dir);    float *b = readbin(p, &nb);
    snprintf(p, sizeof p, "%s/x.bin",  dir);    float *x = readbin(p, &nx);
    snprintf(p, sizeof p, "%s/y_ref.bin", dir); float *y = readbin(p, &ny);
    size_t R=256, E=2304;
    if (nj != 3*R*(2*R+2*E)) { fprintf(stderr, "Jw dims %zu\n", nj); return 3; }
    if (nx != 2*R+2*E) { fprintf(stderr, "x dims %zu\n", nx); return 3; }
    if (nb != 3*R || ny != 3*R) { fprintf(stderr, "b/ref dims %zu/%zu\n", nb, ny); return 3; }
    float *out = malloc(3*R*sizeof(float));
    dspark_joint_proj(J, b, x, R, E, out);
    double sxy=0, sxx=0, syy=0, mx=0, my=0; 
    for (size_t r=0;r<3*R;++r){ mx+=out[r]; my+=y[r]; } mx/=(double)(3*R); my/=(double)(3*R);
    for (size_t r=0;r<3*R;++r){ double o=out[r]-mx, yy=y[r]-my; sxy+=o*yy; sxx+=o*o; syy+=yy*yy; }
    double corr = sxy/(sqrt(sxx)*sqrt(syy));
    double d=0, d2=0; for(size_t r=0;r<3*R;++r){ d+=fabs(out[r]-y[r]); d2+=(out[r]-y[r])*(out[r]-y[r]); }
    printf("corr=%.8f mean|d|=%f rmsd=%.6f\n", corr, d/(double)(3*R), sqrt(d2/(double)(3*R)));
    return fabs(corr) > 0.999 ? 0 : 1;
}
