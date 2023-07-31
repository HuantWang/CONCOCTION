#include <stdlib.h>
#include "ok_png.c" 
#include <stdio.h>
#include "ok_png.h"

int main(int _argc, char **_argv) {
    FILE *file = fopen(_argv[1], "rb");
    ok_png image = ok_png_read(file, OK_PNG_COLOR_FORMAT_RGBA);
    fclose(file);
if (image.data) {
        printf("Got image! Size: %li x %li\n", (long)image.width, (long)image.height);
        free(image.data);
    }
    return 0;
}
