#include <stdio.h>
#include <stdlib.h>
#include "ok_jpg.h"
#include "ok_jpg.c"
 
int main(int _argc, char **_argv) {
    FILE *file = fopen("/home/xrz/Downloads/project/test_jpg/2.jpg", "rb");
    ok_jpg image = ok_jpg_read(file, OK_JPG_COLOR_FORMAT_RGBA);
    fclose(file);
    if (image.data) {
        printf("Got image! Size: %li x %li\n", (long)image.width, (long)image.height);
        free(image.data);
    }
    return 0;
}
