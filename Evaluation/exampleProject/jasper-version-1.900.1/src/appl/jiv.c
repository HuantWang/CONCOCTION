/*
 * Copyright (c) 2002-2003 Michael David Adams.
 * All rights reserved.
 */

/* __START_OF_JASPER_LICENSE__
 * 
 * JasPer License Version 2.0
 * 
 * Copyright (c) 2001-2006 Michael David Adams
 * Copyright (c) 1999-2000 Image Power, Inc.
 * Copyright (c) 1999-2000 The University of British Columbia
 * 
 * All rights reserved.
 * 
 * Permission is hereby granted, free of charge, to any person (the
 * "User") obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, and/or sell copies of the Software, and to permit
 * persons to whom the Software is furnished to do so, subject to the
 * following conditions:
 * 
 * 1.  The above copyright notices and this permission notice (which
 * includes the disclaimer below) shall be included in all copies or
 * substantial portions of the Software.
 * 
 * 2.  The name of a copyright holder shall not be used to endorse or
 * promote products derived from the Software without specific prior
 * written permission.
 * 
 * THIS DISCLAIMER OF WARRANTY CONSTITUTES AN ESSENTIAL PART OF THIS
 * LICENSE.  NO USE OF THE SOFTWARE IS AUTHORIZED HEREUNDER EXCEPT UNDER
 * THIS DISCLAIMER.  THE SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS
 * "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT OF THIRD PARTY RIGHTS.  IN NO
 * EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL
 * INDIRECT OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING
 * FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 * NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
 * WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.  NO ASSURANCES ARE
 * PROVIDED BY THE COPYRIGHT HOLDERS THAT THE SOFTWARE DOES NOT INFRINGE
 * THE PATENT OR OTHER INTELLECTUAL PROPERTY RIGHTS OF ANY OTHER ENTITY.
 * EACH COPYRIGHT HOLDER DISCLAIMS ANY LIABILITY TO THE USER FOR CLAIMS
 * BROUGHT BY ANY OTHER ENTITY BASED ON INFRINGEMENT OF INTELLECTUAL
 * PROPERTY RIGHTS OR OTHERWISE.  AS A CONDITION TO EXERCISING THE RIGHTS
 * GRANTED HEREUNDER, EACH USER HEREBY ASSUMES SOLE RESPONSIBILITY TO SECURE
 * ANY OTHER INTELLECTUAL PROPERTY RIGHTS NEEDED, IF ANY.  THE SOFTWARE
 * IS NOT FAULT-TOLERANT AND IS NOT INTENDED FOR USE IN MISSION-CRITICAL
 * SYSTEMS, SUCH AS THOSE USED IN THE OPERATION OF NUCLEAR FACILITIES,
 * AIRCRAFT NAVIGATION OR COMMUNICATION SYSTEMS, AIR TRAFFIC CONTROL
 * SYSTEMS, DIRECT LIFE SUPPORT MACHINES, OR WEAPONS SYSTEMS, IN WHICH
 * THE FAILURE OF THE SOFTWARE OR SYSTEM COULD LEAD DIRECTLY TO DEATH,
 * PERSONAL INJURY, OR SEVERE PHYSICAL OR ENVIRONMENTAL DAMAGE ("HIGH
 * RISK ACTIVITIES").  THE COPYRIGHT HOLDERS SPECIFICALLY DISCLAIM ANY
 * EXPRESS OR IMPLIED WARRANTY OF FITNESS FOR HIGH RISK ACTIVITIES.
 * 
 * __END_OF_JASPER_LICENSE__
 */

/******************************************************************************\
* Includes
\******************************************************************************/

#include <jasper/jasper.h>
#include <GL/glut.h>
#include <stdlib.h>
#include <math.h>

/******************************************************************************\
*
\******************************************************************************/

#define MAXCMPTS	256
#define BIGPANAMOUNT	0.90
#define SMALLPANAMOUNT	0.05
#define	BIGZOOMAMOUNT	2.0
#define	SMALLZOOMAMOUNT	1.41421356237310

#define	min(x, y)	(((x) < (y)) ? (x) : (y))
#define	max(x, y)	(((x) > (y)) ? (x) : (y))

typedef struct {

	/* The number of image files to view. */
	int numfiles;

	/* The names of the image files. */
	char **filenames;

	/* The title for the window. */
	char *title;

	/* The time to wait before advancing to the next image (in ms). */
	int tmout;

	/* Loop indefinitely over all images. */
	int loop;

	int verbose;

} cmdopts_t;

typedef struct {

	int width;

	int height;

	GLshort *data;

} pixmap_t;

typedef struct {

	/* The index of the current image file. */
	int filenum;

	/* The image. */
	jas_image_t *image;
	jas_image_t *altimage;

	float botleftx;
	float botlefty;
	float toprightx;
	float toprighty;

	int viewportwidth;
	int viewportheight;

	/* The image for display. */
	pixmap_t vp;

	/* The active timer ID. */
	int activetmid;

	/* The next available timer ID. */
	int nexttmid;

	int monomode;

	int cmptno;

} gs_t;

/******************************************************************************\
*
\******************************************************************************/

static void displayfunc(void);
static void reshapefunc(int w, int h);
static void keyboardfunc(unsigned char key, int x, int y);
static void specialfunc(int key, int x, int y);
static void timerfunc(int value);

static void usage(void);
static void nextimage(void);
static void previmage(void);
static void nextcmpt(void);
static void prevcmpt(void);
static int loadimage(void);
static void unloadimage(void);
static int jas_image_render2(jas_image_t *image, int cmptno, float vtlx, float vtly,
  float vsx, float vsy, int vw, int vh, GLshort *vdata);
static int jas_image_render(jas_image_t *image, float vtlx, float vtly,
  float vsx, float vsy, int vw, int vh, GLshort *vdata);

static void dumpstate(void);
static int pixmap_resize(pixmap_t *p, int w, int h);
static void pixmap_clear(pixmap_t *p);
static void cmdinfo(void);

static void cleanupandexit(int);
static void init(void);

static void zoom(float sx, float sy);
static void pan(float dx, float dy);
static void panzoom(float dx, float dy, float sx, float sy);
static void render(void);

/******************************************************************************\
*
\******************************************************************************/

jas_opt_t opts[] = {
	{'V', "version", 0},
	{'v', "v", 0},
	{'h', "help", 0},
	{'w', "wait", JAS_OPT_HASARG},
	{'l', "loop", 0},
	{'t', "title", JAS_OPT_HASARG},
	{-1, 0, 0}
};

char *cmdname = 0;
cmdopts_t cmdopts;
gs_t gs;
jas_stream_t *streamin = 0;

/******************************************************************************\
*
\******************************************************************************/

int main(int argc, char **argv)
{
	int c;

	init();

	/* Determine the base name of this command. */
	if ((cmdname = strrchr(argv[0], '/'))) {
		++cmdname;
	} else {
		cmdname = argv[0];
	}

	/* Initialize the JasPer library. */
	if (jas_init()) {
		abort();
	}

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutCreateWindow(cmdname);
	glutReshapeFunc(reshapefunc);
	glutDisplayFunc(displayfunc);
	glutSpecialFunc(specialfunc);
	glutKeyboardFunc(keyboardfunc);

	cmdopts.numfiles = 0;
	cmdopts.filenames = 0;
	cmdopts.title = 0;
	cmdopts.tmout = 0;
	cmdopts.loop = 0;
	cmdopts.verbose = 0;

	while ((c = jas_getopt(argc, argv, opts)) != EOF) {
		switch (c) {
		case 'w':
			cmdopts.tmout = atof(jas_optarg) * 1000;
			break;
		case 'l':
			cmdopts.loop = 1;
			break;
		case 't':
			cmdopts.title = jas_optarg;
			break;
		case 'v':
			cmdopts.verbose = 1;
			break;
		case 'V':
			printf("%s\n", JAS_VERSION);
			fprintf(stderr, "libjasper %s\n", jas_getversion());
			cleanupandexit(EXIT_SUCCESS);
			break;
		default:
		case 'h':
			usage();
			break;
		}
	}

	if (jas_optind < argc) {
		/* The images are to be read from one or more explicitly named
		  files. */
		cmdopts.numfiles = argc - jas_optind;
		cmdopts.filenames = &argv[jas_optind];
	} else {
		/* The images are to be read from standard input. */
		static char *null = 0;
		cmdopts.filenames = &null;
		cmdopts.numfiles = 1;
	}

	streamin = jas_stream_fdopen(0, "rb");

	/* Load the next image. */
	nextimage();

	/* Start the GLUT main event handler loop. */
	glutMainLoop();

	return EXIT_SUCCESS;
}

/******************************************************************************\
*
\******************************************************************************/

static void cmdinfo()
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static void cmdinfo() {\n");
	printf("fprintf(stderr, 'JasPer Image Viewer (Version _s). ', 	  JAS_VERSION);\n");
	fprintf(stderr, "JasPer Image Viewer (Version %s).\n",
	  JAS_VERSION);
	printf("fprintf(stderr, 'Copyright (c) 2002-2003 Michael David Adams. ' 	  'All rights reserved. ');\n");
	fprintf(stderr, "Copyright (c) 2002-2003 Michael David Adams.\n"
	  "All rights reserved.\n");
	printf("fprintf(stderr, '_s ', JAS_NOTES);\n");
	fprintf(stderr, "%s\n", JAS_NOTES);
printf("------function end!------\n");
}

static char *helpinfo[] = {
"The following options are supported:\n",
"    --help                  Print this help information and exit.\n",
"    --version               Print version information and exit.\n",
"    --loop                  Loop indefinitely through images.\n",
"    --wait N                Advance to next image after N seconds.\n",
0
};

static void usage()
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static void usage() {\n");
	printf("char *s;\n");
	char *s;
	printf("int i;\n");
	int i;
	printf("cmdinfo();\n");
	cmdinfo();
	printf("fprintf(stderr, 'usage: _s [options] [file1 file2 ...] ', cmdname);\n");
	fprintf(stderr, "usage: %s [options] [file1 file2 ...]\n", cmdname);
	for (i = 0, s = helpinfo[i]; s; ++i, s = helpinfo[i]) {printf("for(i = 0, s = helpinfo[i];s;++i, s = helpinfo[i])\n");
	
		printf("fprintf(stderr, '_s', s);\n");
		fprintf(stderr, "%s", s);
	}
	printf("cleanupandexit(EXIT_FAILURE);\n");
	cleanupandexit(EXIT_FAILURE);
printf("------function end!------\n");
}

/******************************************************************************\
* GLUT Callback Functions
\******************************************************************************/

/* Display callback function. */

static void displayfunc()
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static void displayfunc() {\n");

	printf("float w;\n");
	float w;
	printf("float h;\n");
	float h;
	printf("int regbotleftx;\n");
	int regbotleftx;
	printf("int regbotlefty;\n");
	int regbotlefty;
	printf("int regtoprightx;\n");
	int regtoprightx;
	printf("int regtoprighty;\n");
	int regtoprighty;
	printf("int regtoprightwidth;\n");
	int regtoprightwidth;
	printf("int regtoprightheight;\n");
	int regtoprightheight;
	printf("int regwidth;\n");
	int regwidth;
	printf("int regheight;\n");
	int regheight;
	printf("float x;\n");
	float x;
	printf("float y;\n");
	float y;
	printf("float xx;\n");
	float xx;
	printf("float yy;\n");
	float yy;

	if (cmdopts.verbose) {printf("if(cmdopts.verbose)\n");
	
		printf("fprintf(stderr, 'displayfunc() ');\n");
		fprintf(stderr, "displayfunc()\n");
	}

	printf("regbotleftx = max(ceil(gs.botleftx), 0);\n");
regbotleftx = max(ceil(gs.botleftx), 0);
	printf("regbotlefty = max(ceil(gs.botlefty), 0);\n");
regbotlefty = max(ceil(gs.botlefty), 0);
	regtoprightx = min(gs.vp.width, floor(gs.toprightx));
	regtoprighty = min(gs.vp.height, floor(gs.toprighty));
	printf("regwidth = regtoprightx - regbotleftx;\n");
regwidth = regtoprightx - regbotleftx;
	printf("regheight = regtoprighty - regbotlefty;\n");
regheight = regtoprighty - regbotlefty;
	printf("w = gs.toprightx - gs.botleftx;\n");
w = gs.toprightx - gs.botleftx;
	printf("h = gs.toprighty - gs.botlefty;\n");
h = gs.toprighty - gs.botlefty;
	printf("x = (regbotleftx - gs.botleftx) / w;\n");
x = (regbotleftx - gs.botleftx) / w;
	printf("y = (regbotlefty - gs.botlefty) / h;\n");
y = (regbotlefty - gs.botlefty) / h;
	printf("xx = (regtoprightx - gs.botleftx) / w;\n");
xx = (regtoprightx - gs.botleftx) / w;
	printf("yy = (regtoprighty - gs.botlefty) / h;\n");
yy = (regtoprighty - gs.botlefty) / h;

	printf("assert(regwidth > 0);\n");
	assert(regwidth > 0);
	printf("assert(regheight > 0);\n");
	assert(regheight > 0);
	printf("assert(abs(((double) regheight / regwidth) - ((double) gs.viewportheight / gs.viewportwidth)) < 1e-5);\n");
	assert(abs(((double) regheight / regwidth) - ((double) gs.viewportheight / gs.viewportwidth)) < 1e-5);

	glClear(GL_COLOR_BUFFER_BIT);
	glPixelStorei(GL_UNPACK_ALIGNMENT, sizeof(GLshort));
	glPixelStorei(GL_UNPACK_ROW_LENGTH, gs.vp.width);
	glPixelStorei(GL_UNPACK_SKIP_PIXELS, regbotleftx);
	glPixelStorei(GL_UNPACK_SKIP_ROWS, regbotlefty);
	printf("glRasterPos2f(x * gs.viewportwidth, y * gs.viewportheight);\n");
	glRasterPos2f(x * gs.viewportwidth, y * gs.viewportheight);
	printf("glPixelZoom((xx - x) * ((double) gs.viewportwidth) / regwidth, (yy - y) * ((double) gs.viewportheight) / regheight);\n");
	glPixelZoom((xx - x) * ((double) gs.viewportwidth) / regwidth, (yy - y) * ((double) gs.viewportheight) / regheight);
	glDrawPixels(regwidth, regheight, GL_RGBA, GL_UNSIGNED_SHORT,
	  gs.vp.data);
	printf("glFlush();\n");
	glFlush();
	printf("glutSwapBuffers();\n");
	glutSwapBuffers();

printf("------function end!------\n");
}

/* Reshape callback function. */

static void reshapefunc(int w, int h)
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static void reshapefunc(int w, int h) {\n");
	if (cmdopts.verbose) {printf("if(cmdopts.verbose)\n");
	
		printf("fprintf(stderr, 'reshapefunc(_d, _d) ', w, h);\n");
		fprintf(stderr, "reshapefunc(%d, %d)\n", w, h);
		printf("dumpstate();\n");
		dumpstate();
	}

	printf("glViewport(0, 0, w, h);\n");
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	printf("glLoadIdentity();\n");
	glLoadIdentity();
	printf("gluOrtho2D(0, w, 0, h);\n");
	gluOrtho2D(0, w, 0, h);
	glMatrixMode(GL_MODELVIEW);
	printf("glLoadIdentity();\n");
	glLoadIdentity();
	printf("glTranslatef(0, 0, 0);\n");
	glTranslatef(0, 0, 0);
	printf("glRasterPos2i(0, 0);\n");
	glRasterPos2i(0, 0);

	printf("zoom((double) gs.viewportwidth / w, (double) gs.viewportheight / h);\n");
	zoom((double) gs.viewportwidth / w, (double) gs.viewportheight / h);
	printf("gs.viewportwidth = w;\n");
gs.viewportwidth = w;
	printf("gs.viewportheight = h;\n");
gs.viewportheight = h;

printf("------function end!------\n");
}

/* Keyboard callback function. */

static void keyboardfunc(unsigned char key, int x, int y)
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static void keyboardfunc(unsigned char key, int x, int y) {\n");
	if (cmdopts.verbose) {printf("if(cmdopts.verbose)\n");
	
		printf("fprintf(stderr, 'keyboardfunc(_d, _d, _d) ', key, x, y);\n");
		fprintf(stderr, "keyboardfunc(%d, %d, %d)\n", key, x, y);
	}

	printf("switch(key)\n");
	switch (key) {
	case ' ':
		printf("nextimage();\n");
		nextimage();
		break;
	case '\b':
		printf("previmage();\n");
		previmage();
		break;
	case '>':
		printf("zoom(BIGZOOMAMOUNT, BIGZOOMAMOUNT);\n");
		zoom(BIGZOOMAMOUNT, BIGZOOMAMOUNT);
		printf("glutPostRedisplay();\n");
		glutPostRedisplay();
		break;
	case '.':
		printf("zoom(SMALLZOOMAMOUNT, SMALLZOOMAMOUNT);\n");
		zoom(SMALLZOOMAMOUNT, SMALLZOOMAMOUNT);
		printf("glutPostRedisplay();\n");
		glutPostRedisplay();
		break;
	case '<':
		printf("zoom(1.0 / BIGZOOMAMOUNT, 1.0 / BIGZOOMAMOUNT);\n");
		zoom(1.0 / BIGZOOMAMOUNT, 1.0 / BIGZOOMAMOUNT);
		printf("glutPostRedisplay();\n");
		glutPostRedisplay();
		break;
	case ',':
		printf("zoom(1.0 / SMALLZOOMAMOUNT, 1.0 / SMALLZOOMAMOUNT);\n");
		zoom(1.0 / SMALLZOOMAMOUNT, 1.0 / SMALLZOOMAMOUNT);
		printf("glutPostRedisplay();\n");
		glutPostRedisplay();
		break;
	case 'c':
		printf("nextcmpt();\n");
		nextcmpt();
		break;
	case 'C':
		printf("prevcmpt();\n");
		prevcmpt();
		break;
	case 'h':
		printf("fprintf(stderr, 'h             help ');\n");
		fprintf(stderr, "h             help\n");
		printf("fprintf(stderr, '>             zoom in (large) ');\n");
		fprintf(stderr, ">             zoom in (large)\n");
		printf("fprintf(stderr, ',             zoom in (small) ');\n");
		fprintf(stderr, ",             zoom in (small)\n");
		printf("fprintf(stderr, '<             zoom out (large) ');\n");
		fprintf(stderr, "<             zoom out (large)\n");
		printf("fprintf(stderr, '.             zoom out (small) ');\n");
		fprintf(stderr, ".             zoom out (small)\n");
		printf("fprintf(stderr, 'down arrow    pan down ');\n");
		fprintf(stderr, "down arrow    pan down\n");
		printf("fprintf(stderr, 'up arrow      pan up ');\n");
		fprintf(stderr, "up arrow      pan up\n");
		printf("fprintf(stderr, 'left arrow    pan left ');\n");
		fprintf(stderr, "left arrow    pan left\n");
		printf("fprintf(stderr, 'right arrow   pan right ');\n");
		fprintf(stderr, "right arrow   pan right\n");
		printf("fprintf(stderr, 'space         next image ');\n");
		fprintf(stderr, "space         next image\n");
		printf("fprintf(stderr, 'backspace     previous image ');\n");
		fprintf(stderr, "backspace     previous image\n");
		printf("fprintf(stderr, 'q             quit ');\n");
		fprintf(stderr, "q             quit\n");
		break;
	case 'q':
		printf("cleanupandexit(EXIT_SUCCESS);\n");
		cleanupandexit(EXIT_SUCCESS);
		break;
	}
printf("------function end!------\n");
}

/* Special keyboard callback function. */

static void specialfunc(int key, int x, int y)
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static void specialfunc(int key, int x, int y) {\n");
	if (cmdopts.verbose) {printf("if(cmdopts.verbose)\n");
	
		printf("fprintf(stderr, 'specialfunc(_d, _d, _d) ', key, x, y);\n");
		fprintf(stderr, "specialfunc(%d, %d, %d)\n", key, x, y);
	}

	printf("switch(key)\n");
	switch (key) {
	case GLUT_KEY_UP:
		{
			float panamount;
			panamount = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) ?
			  BIGPANAMOUNT : SMALLPANAMOUNT;
			pan(0.0, panamount * (gs.toprighty - gs.botlefty));
			glutPostRedisplay();
		}
		break;
	case GLUT_KEY_DOWN:
		{
			float panamount;
			panamount = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) ?
			  BIGPANAMOUNT : SMALLPANAMOUNT;
			pan(0.0, -panamount * (gs.toprighty - gs.botlefty));
			glutPostRedisplay();
		}
		break;
	case GLUT_KEY_LEFT:
		{
			float panamount;
			panamount = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) ?
			  BIGPANAMOUNT : SMALLPANAMOUNT;
			pan(-panamount * (gs.toprightx - gs.botleftx), 0.0);
			glutPostRedisplay();
		}
		break;
	case GLUT_KEY_RIGHT:
		{
			float panamount;
			panamount = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) ?
			  BIGPANAMOUNT : SMALLPANAMOUNT;
			pan(panamount * (gs.toprightx - gs.botleftx), 0.0);
			glutPostRedisplay();
		}
		break;
	default:
		break;
	}
printf("------function end!------\n");
}

/* Timer callback function. */

static void timerfunc(int value)
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static void timerfunc(int value) {\n");
	if (cmdopts.verbose) {printf("if(cmdopts.verbose)\n");
	
		printf("fprintf(stderr, 'timerfunc(_d) ', value);\n");
		fprintf(stderr, "timerfunc(%d)\n", value);
	}
	if (value == gs.activetmid) {printf("if(value == gs.activetmid)\n");
	
		printf("nextimage();\n");
		nextimage();
	}
printf("------function end!------\n");
}

/******************************************************************************\
*
\******************************************************************************/

static void zoom(float sx, float sy)
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static void zoom(float sx, float sy) {\n");
	printf("panzoom(0, 0, sx, sy);\n");
	panzoom(0, 0, sx, sy);
printf("------function end!------\n");
}

static void pan(float dx, float dy)
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static void pan(float dx, float dy) {\n");
	printf("panzoom(dx, dy, 1.0, 1.0);\n");
	panzoom(dx, dy, 1.0, 1.0);
printf("------function end!------\n");
}

static void panzoom(float dx, float dy, float sx, float sy)
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static void panzoom(float dx, float dy, float sx, float sy) {\n");
	printf("float w;\n");
	float w;
	printf("float h;\n");
	float h;
	printf("float cx;\n");
	float cx;
	printf("float cy;\n");
	float cy;
	printf("int reginh;\n");
	int reginh;
	printf("int reginv;\n");
	int reginv;

	reginh = (gs.botleftx >= 0 && gs.toprightx <= gs.vp.width);
	reginv = (gs.botlefty >= 0 && gs.toprighty <= gs.vp.height);

	if (cmdopts.verbose) {printf("if(cmdopts.verbose)\n");
	
		printf("fprintf(stderr, 'start of panzoom ');\n");
		fprintf(stderr, "start of panzoom\n");
		printf("dumpstate();\n");
		dumpstate();
		printf("fprintf(stderr, 'reginh=_d reginv=_d ', reginh, reginv);\n");
		fprintf(stderr, "reginh=%d reginv=%d\n", reginh, reginv);
	}

	if (dx || dy) {printf("if(dx || dy)\n");
	
		printf("gs.botleftx += dx;\n");
gs.botleftx += dx;
		printf("gs.botlefty += dy;\n");
gs.botlefty += dy;
		printf("gs.toprightx += dx;\n");
gs.toprightx += dx;
		printf("gs.toprighty += dy;\n");
gs.toprighty += dy;
	}

	if (sx != 1.0 || sy != 1.0) {printf("if(sx != 1.0 || sy != 1.0)\n");
	
		printf("cx = (gs.botleftx + gs.toprightx) / 2.0;\n");
cx = (gs.botleftx + gs.toprightx) / 2.0;
		printf("cy = (gs.botlefty + gs.toprighty) / 2.0;\n");
cy = (gs.botlefty + gs.toprighty) / 2.0;
		printf("w = gs.toprightx - gs.botleftx;\n");
w = gs.toprightx - gs.botleftx;
		printf("h = gs.toprighty - gs.botlefty;\n");
h = gs.toprighty - gs.botlefty;
		printf("gs.botleftx = cx - 0.5 * w / sx;\n");
gs.botleftx = cx - 0.5 * w / sx;
		printf("gs.botlefty = cy - 0.5 * h / sy;\n");
gs.botlefty = cy - 0.5 * h / sy;
		printf("gs.toprightx = cx + 0.5 * w / sx;\n");
gs.toprightx = cx + 0.5 * w / sx;
		printf("gs.toprighty = cy + 0.5 * h / sy;\n");
gs.toprighty = cy + 0.5 * h / sy;
	}

	if (reginh) {printf("if(reginh)\n");
	
		if (gs.botleftx < 0) {printf("if(gs.botleftx < 0)\n");
		
			printf("dx = -gs.botleftx;\n");
dx = -gs.botleftx;
			printf("gs.botleftx += dx;\n");
gs.botleftx += dx;
			printf("gs.toprightx += dx;\n");
gs.toprightx += dx;
		} else if (gs.toprightx > gs.vp.width) {printf("if()\n");
		
			dx = gs.vp.width - gs.toprightx;
			printf("gs.botleftx += dx;\n");
gs.botleftx += dx;
			printf("gs.toprightx += dx;\n");
gs.toprightx += dx;
		}
	}
	if (gs.botleftx < 0 || gs.toprightx > gs.vp.width) {printf("if()\n");
	
		printf("float w;\n");
		float w;
		printf("w = gs.toprightx - gs.botleftx;\n");
w = gs.toprightx - gs.botleftx;
		gs.botleftx = 0.5 * gs.vp.width - 0.5 * w;
		gs.toprightx = 0.5 * gs.vp.width + 0.5 * w;
	}

	if (reginv) {printf("if(reginv)\n");
	
		if (gs.botlefty < 0) {printf("if(gs.botlefty < 0)\n");
		
			printf("dy = -gs.botlefty;\n");
dy = -gs.botlefty;
			printf("gs.botlefty += dy;\n");
gs.botlefty += dy;
			printf("gs.toprighty += dy;\n");
gs.toprighty += dy;
		} else if (gs.toprighty > gs.vp.height) {printf("if()\n");
		
			dy = gs.vp.height - gs.toprighty;
			printf("gs.botlefty += dy;\n");
gs.botlefty += dy;
			printf("gs.toprighty += dy;\n");
gs.toprighty += dy;
		}
	}
	if (gs.botlefty < 0 || gs.toprighty > gs.vp.height) {printf("if()\n");
	
		printf("float h;\n");
		float h;
		printf("h = gs.toprighty - gs.botlefty;\n");
h = gs.toprighty - gs.botlefty;
		gs.botlefty = 0.5 * gs.vp.height - 0.5 * h;
		gs.toprighty = 0.5 * gs.vp.height + 0.5 * h;
	}

	if (cmdopts.verbose) {printf("if(cmdopts.verbose)\n");
	
		printf("fprintf(stderr, 'end of panzoom ');\n");
		fprintf(stderr, "end of panzoom\n");
		printf("dumpstate();\n");
		dumpstate();
	}
printf("------function end!------\n");
}

static void nextcmpt()
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static void nextcmpt() {\n");
	if (gs.monomode) {printf("if(gs.monomode)\n");
	
		if (gs.cmptno == jas_image_numcmpts(gs.image) - 1) {printf("if(gs.cmptno == jas_image_numcmpts(gs.image) - 1)\n");
		
			if (gs.altimage) {printf("if(gs.altimage)\n");
			
				printf("gs.monomode = 0;\n");
gs.monomode = 0;
			} else {
				printf("gs.cmptno = 0;\n");
gs.cmptno = 0;
			}
		} else {
			printf("++gs.cmptno;\n");
++gs.cmptno;
		}
	} else {
		printf("gs.monomode = 1;\n");
gs.monomode = 1;
		printf("gs.cmptno = 0;\n");
gs.cmptno = 0;
	}
	printf("render();\n");
	render();
	printf("glutPostRedisplay();\n");
	glutPostRedisplay();
printf("------function end!------\n");
}

static void prevcmpt()
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static void prevcmpt() {\n");
	if (gs.monomode) {printf("if(gs.monomode)\n");
	
		if (!gs.cmptno) {printf("if(!gs.cmptno)\n");
		
			printf("gs.monomode = 0;\n");
gs.monomode = 0;
		} else {
			printf("--gs.cmptno;\n");
--gs.cmptno;
		}
	} else {
		printf("gs.monomode = 1;\n");
gs.monomode = 1;
		printf("gs.cmptno = jas_image_numcmpts(gs.image) - 1;\n");
gs.cmptno = jas_image_numcmpts(gs.image) - 1;
	}
	printf("render();\n");
	render();
	printf("glutPostRedisplay();\n");
	glutPostRedisplay();
printf("------function end!------\n");
}

static void nextimage()
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static void nextimage() {\n");
	printf("int n;\n");
	int n;
	printf("unloadimage();\n");
	unloadimage();
	for (n = cmdopts.numfiles; n > 0; --n) {printf("for(n = cmdopts.numfiles;n > 0;--n)\n");
	
		printf("++gs.filenum;\n");
++gs.filenum;
		if (gs.filenum >= cmdopts.numfiles) {printf("if(gs.filenum >= cmdopts.numfiles)\n");
		
			if (cmdopts.loop) {printf("if(cmdopts.loop)\n");
			
				printf("gs.filenum = 0;\n");
gs.filenum = 0;
			} else {
				printf("cleanupandexit(EXIT_SUCCESS);\n");
				cleanupandexit(EXIT_SUCCESS);
			}
		}
		if (!loadimage()) {printf("if(!loadimage())\n");
		
			printf("return;\n");
			printf("------function end!------\n");
			return;
		}
		printf("fprintf(stderr, 'cannot load image ');\n");
		fprintf(stderr, "cannot load image\n");
	}
	printf("cleanupandexit(EXIT_SUCCESS);\n");
	cleanupandexit(EXIT_SUCCESS);
printf("------function end!------\n");
}

static void previmage()
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static void previmage() {\n");
	printf("int n;\n");
	int n;
	printf("unloadimage();\n");
	unloadimage();
	for (n = cmdopts.numfiles; n > 0; --n) {printf("for(n = cmdopts.numfiles;n > 0;--n)\n");
	
		printf("--gs.filenum;\n");
--gs.filenum;
		if (gs.filenum < 0) {printf("if(gs.filenum < 0)\n");
		
			if (cmdopts.loop) {printf("if(cmdopts.loop)\n");
			
				printf("gs.filenum = cmdopts.numfiles - 1;\n");
gs.filenum = cmdopts.numfiles - 1;
			} else {
				printf("cleanupandexit(EXIT_SUCCESS);\n");
				cleanupandexit(EXIT_SUCCESS);
			}
		}
		if (!loadimage()) {printf("if(!loadimage())\n");
		
			printf("return;\n");
			printf("------function end!------\n");
			return;
		}
	}
	printf("cleanupandexit(EXIT_SUCCESS);\n");
	cleanupandexit(EXIT_SUCCESS);
printf("------function end!------\n");
}

static int loadimage()
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static int loadimage() {\n");
	printf("int reshapeflag;\n");
	int reshapeflag;
	printf("jas_stream_t *in;\n");
	jas_stream_t *in;
	printf("int scrnwidth;\n");
	int scrnwidth;
	printf("int scrnheight;\n");
	int scrnheight;
	printf("int vh;\n");
	int vh;
	printf("int vw;\n");
	int vw;
	printf("char *pathname;\n");
	char *pathname;
	printf("jas_cmprof_t *outprof;\n");
	jas_cmprof_t *outprof;

	printf("assert(!gs.image);\n");
	assert(!gs.image);
	printf("assert(!gs.altimage);\n");
	assert(!gs.altimage);

	printf("gs.image = 0;\n");
gs.image = 0;
	printf("gs.altimage = 0;\n");
gs.altimage = 0;

	printf("pathname = cmdopts.filenames[gs.filenum];\n");
pathname = cmdopts.filenames[gs.filenum];

	if (pathname && pathname[0] != '\0') {printf("if(pathname && pathname[0] != '\0')\n");
	
		if (cmdopts.verbose) {printf("if(cmdopts.verbose)\n");
		
			printf("fprintf(stderr, 'opening file _s ', pathname);\n");
			fprintf(stderr, "opening file %s\n", pathname);
		}
		/* The input image is to be read from a file. */
		if (!(in = jas_stream_fopen(pathname, "rb"))) {printf("if(!(in = jas_stream_fopen(pathname, 'rb')))\n");
		
			printf("fprintf(stderr, 'error: cannot open file _s ', pathname);\n");
			fprintf(stderr, "error: cannot open file %s\n", pathname);
			printf("goto error;\n");
			goto error;
		}
	} else {
		/* The input image is to be read from standard input. */
		printf("in = streamin;\n");
in = streamin;
	}

	if (cmdopts.verbose) {printf("if(cmdopts.verbose)\n");
	
		printf("fprintf(stderr, 'decoding image ');\n");
		fprintf(stderr, "decoding image\n");
	}

	/* Get the input image data. */
	if (!(gs.image = jas_image_decode(in, -1, 0))) {printf("if(!(gs.image = jas_image_decode(in, -1, 0)))\n");
	
		printf("fprintf(stderr, 'error: cannot load image data ');\n");
		fprintf(stderr, "error: cannot load image data\n");
		printf("goto error;\n");
		goto error;
	}

	/* Close the input stream. */
	if (in != streamin) {printf("if(in != streamin)\n");
	
		printf("jas_stream_close(in);\n");
		jas_stream_close(in);
	}

	if (cmdopts.verbose) {printf("if(cmdopts.verbose)\n");
	
		printf("fprintf(stderr, 'creating color profile ');\n");
		fprintf(stderr, "creating color profile\n");
	}

	if (!(outprof = jas_cmprof_createfromclrspc(JAS_CLRSPC_SRGB)))
		{printf("if(!(outprof = jas_cmprof_createfromclrspc(JAS_CLRSPC_SRGB)))\n");
		printf("goto error;\n");
		goto error;}
	if (!(gs.altimage = jas_image_chclrspc(gs.image, outprof, JAS_CMXFORM_INTENT_PER)))
		{printf("if(!(gs.altimage = jas_image_chclrspc(gs.image, outprof, JAS_CMXFORM_INTENT_PER)))\n");
		printf("goto error;\n");
		goto error;}

	printf("vw = jas_image_width(gs.image);\n");
vw = jas_image_width(gs.image);
	printf("vh = jas_image_height(gs.image);\n");
vh = jas_image_height(gs.image);

	printf("gs.botleftx = jas_image_tlx(gs.image);\n");
gs.botleftx = jas_image_tlx(gs.image);
	printf("gs.botlefty = jas_image_tly(gs.image);\n");
gs.botlefty = jas_image_tly(gs.image);
	printf("gs.toprightx = jas_image_brx(gs.image);\n");
gs.toprightx = jas_image_brx(gs.image);
	printf("gs.toprighty = jas_image_bry(gs.image);\n");
gs.toprighty = jas_image_bry(gs.image);
	if (gs.altimage) {printf("if(gs.altimage)\n");
	
		printf("gs.monomode = 0;\n");
gs.monomode = 0;
	} else {
		printf("gs.monomode = 1;\n");
gs.monomode = 1;
		printf("gs.cmptno = 0;\n");
gs.cmptno = 0;
	}


	if (cmdopts.verbose) {printf("if(cmdopts.verbose)\n");
	
		printf("fprintf(stderr, 'num of components _d ', jas_image_numcmpts(gs.image));\n");
		fprintf(stderr, "num of components %d\n", jas_image_numcmpts(gs.image));
		printf("fprintf(stderr, 'dimensions _d _d ', jas_image_width(gs.image), jas_image_height(gs.image));\n");
		fprintf(stderr, "dimensions %d %d\n", jas_image_width(gs.image), jas_image_height(gs.image));
	}

	printf("gs.viewportwidth = vw;\n");
gs.viewportwidth = vw;
	printf("gs.viewportheight = vh;\n");
gs.viewportheight = vh;
	pixmap_resize(&gs.vp, vw, vh);
	if (cmdopts.verbose) {printf("if(cmdopts.verbose)\n");
	
		printf("fprintf(stderr, 'preparing image for viewing ');\n");
		fprintf(stderr, "preparing image for viewing\n");
	}
	printf("render();\n");
	render();
	if (cmdopts.verbose) {printf("if(cmdopts.verbose)\n");
	
		printf("fprintf(stderr, 'done preparing image for viewing ');\n");
		fprintf(stderr, "done preparing image for viewing\n");
	}

	if (vw != glutGet(GLUT_WINDOW_WIDTH) ||
	  vh != glutGet(GLUT_WINDOW_HEIGHT)) {printf("if()\n");
	  
		printf("glutReshapeWindow(vw, vh);\n");
		glutReshapeWindow(vw, vh);
	}
	if (cmdopts.title) {printf("if(cmdopts.title)\n");
	
		printf("glutSetWindowTitle(cmdopts.title);\n");
		glutSetWindowTitle(cmdopts.title);
	} else {
		printf("glutSetWindowTitle((pathname && pathname[0] != '\0') ? pathname : 		  'stdin');\n");
		glutSetWindowTitle((pathname && pathname[0] != '\0') ? pathname :
		  "stdin");
	}
	/* If we reshaped the window, GLUT will automatically invoke both
	  the reshape and display callback (in this order).  Therefore, we
	  only need to explicitly force the display callback to be invoked
	  if the window was not reshaped. */
	printf("glutPostRedisplay();\n");
	glutPostRedisplay();

	if (cmdopts.tmout != 0) {printf("if(cmdopts.tmout != 0)\n");
	
		printf("glutTimerFunc(cmdopts.tmout, timerfunc, gs.nexttmid);\n");
		glutTimerFunc(cmdopts.tmout, timerfunc, gs.nexttmid);
		printf("gs.activetmid = gs.nexttmid;\n");
gs.activetmid = gs.nexttmid;
		printf("++gs.nexttmid;\n");
++gs.nexttmid;
	}

	printf("return 0;\n");
	printf("------function end!------\n");
	return 0;

error:
	unloadimage();
	printf("return -1;\n");
	printf("------function end!------\n");
	return -1;
printf("------function end!------\n");
}

static void unloadimage()
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static void unloadimage() {\n");
	if (gs.image) {printf("if(gs.image)\n");
	
		printf("jas_image_destroy(gs.image);\n");
		jas_image_destroy(gs.image);
		printf("gs.image = 0;\n");
gs.image = 0;
	}
	if (gs.altimage) {printf("if(gs.altimage)\n");
	
		printf("jas_image_destroy(gs.altimage);\n");
		jas_image_destroy(gs.altimage);
		printf("gs.altimage = 0;\n");
gs.altimage = 0;
	}
printf("------function end!------\n");
}

/******************************************************************************\
*
\******************************************************************************/

static void pixmap_clear(pixmap_t *p)
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static void pixmap_clear(pixmap_t *p) {\n");
	memset(p->data, 0, 4 * p->width * p->height * sizeof(GLshort));
printf("------function end!------\n");
}

static int pixmap_resize(pixmap_t *p, int w, int h)
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static int pixmap_resize(pixmap_t *p, int w, int h) {\n");
	printf("p->width = w;\n");
p->width = w;
	printf("p->height = h;\n");
p->height = h;
	if (!(p->data = realloc(p->data, w * h * 4 * sizeof(GLshort)))) {printf("if()\n");
	
		printf("return -1;\n");
		printf("------function end!------\n");
		return -1;
	}
	printf("return 0;\n");
	printf("------function end!------\n");
	return 0;
printf("------function end!------\n");
}

static void dumpstate()
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static void dumpstate() {\n");
	printf("printf('blx=_f bly=_f trx=_f try=_f ', gs.botleftx, gs.botlefty, gs.toprightx, gs.toprighty);\n");
	printf("blx=%f bly=%f trx=%f try=%f\n", gs.botleftx, gs.botlefty, gs.toprightx, gs.toprighty);
printf("------function end!------\n");
}

#define	vctocc(i, co, cs, vo, vs) \
  (((vo) + (i) * (vs) - (co)) / (cs))

static int jas_image_render(jas_image_t *image, float vtlx, float vtly,
  float vsx, float vsy, int vw, int vh, GLshort *vdata)
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static int jas_image_render(jas_image_t *image, float vtlx, float vtly,   float vsx, float vsy, int vw, int vh, GLshort *vdata) {\n");
	printf("int i;\n");
	int i;
	printf("int j;\n");
	int j;
	printf("int k;\n");
	int k;
	printf("int x;\n");
	int x;
	printf("int y;\n");
	int y;
	printf("int v[3];\n");
	int v[3];
	GLshort *vdatap;
	printf("int cmptlut[3];\n");
	int cmptlut[3];
	printf("int width;\n");
	int width;
	printf("int height;\n");
	int height;
	printf("int hs;\n");
	int hs;
	printf("int vs;\n");
	int vs;
	printf("int tlx;\n");
	int tlx;
	printf("int tly;\n");
	int tly;

	if ((cmptlut[0] = jas_image_getcmptbytype(image,
	  JAS_IMAGE_CT_COLOR(JAS_CLRSPC_CHANIND_RGB_R))) < 0 ||
	  (cmptlut[1] = jas_image_getcmptbytype(image,
	  JAS_IMAGE_CT_COLOR(JAS_CLRSPC_CHANIND_RGB_G))) < 0 ||
	  (cmptlut[2] = jas_image_getcmptbytype(image,
	  JAS_IMAGE_CT_COLOR(JAS_CLRSPC_CHANIND_RGB_B))) < 0)
		{printf("if((cmptlut[0] = jas_image_getcmptbytype(image, 	  JAS_IMAGE_CT_COLOR(JAS_CLRSPC_CHANIND_RGB_R))) < 0 || 	  (cmptlut[1] = jas_image_getcmptbytype(image, 	  JAS_IMAGE_CT_COLOR(JAS_CLRSPC_CHANIND_RGB_G))) < 0 || 	  (cmptlut[2] = jas_image_getcmptbytype(image, 	  JAS_IMAGE_CT_COLOR(JAS_CLRSPC_CHANIND_RGB_B))) < 0)\n");
		printf("goto error;\n");
		goto error;}
	printf("width = jas_image_cmptwidth(image, cmptlut[0]);\n");
width = jas_image_cmptwidth(image, cmptlut[0]);
	printf("height = jas_image_cmptheight(image, cmptlut[0]);\n");
height = jas_image_cmptheight(image, cmptlut[0]);
	printf("tlx = jas_image_cmpttlx(image, cmptlut[0]);\n");
tlx = jas_image_cmpttlx(image, cmptlut[0]);
	printf("tly = jas_image_cmpttly(image, cmptlut[0]);\n");
tly = jas_image_cmpttly(image, cmptlut[0]);
	printf("vs = jas_image_cmptvstep(image, cmptlut[0]);\n");
vs = jas_image_cmptvstep(image, cmptlut[0]);
	printf("hs = jas_image_cmpthstep(image, cmptlut[0]);\n");
hs = jas_image_cmpthstep(image, cmptlut[0]);
	for (i = 1; i < 3; ++i) {printf("for(i = 1;i < 3;++i)\n");
	
		if (jas_image_cmptwidth(image, cmptlut[i]) != width ||
		  jas_image_cmptheight(image, cmptlut[i]) != height)
			{printf("if(jas_image_cmptwidth(image, cmptlut[i]) != width || 		  jas_image_cmptheight(image, cmptlut[i]) != height)\n");
			printf("goto error;\n");
			goto error;}
	}
	for (i = 0; i < vh; ++i) {printf("for(i = 0;i < vh;++i)\n");
	
		vdatap = &vdata[(vh - 1 - i) * (4 * vw)];
		for (j = 0; j < vw; ++j) {printf("for(j = 0;j < vw;++j)\n");
		
			printf("x = vctocc(j, tlx, hs, vtlx, vsx);\n");
x = vctocc(j, tlx, hs, vtlx, vsx);
			printf("y = vctocc(i, tly, vs, vtly, vsy);\n");
y = vctocc(i, tly, vs, vtly, vsy);
			if (x >= 0 && x < width && y >= 0 && y < height) {printf("if(x >= 0 && x < width && y >= 0 && y < height)\n");
			
				for (k = 0; k < 3; ++k) {printf("for(k = 0;k < 3;++k)\n");
				
					printf("v[k] = jas_image_readcmptsample(image, cmptlut[k], x, y);\n");
v[k] = jas_image_readcmptsample(image, cmptlut[k], x, y);
					printf("v[k] <<= 16 - jas_image_cmptprec(image, cmptlut[k]);\n");
v[k] <<= 16 - jas_image_cmptprec(image, cmptlut[k]);
					if (v[k] < 0) {printf("if(v[k] < 0)\n");
					
						printf("v[k] = 0;\n");
v[k] = 0;
					} else if (v[k] > 65535) {printf("if(v[k] > 65535)\n");
					
						printf("v[k] = 65535;\n");
v[k] = 65535;
					}
				}
			} else {
				printf("v[0] = 0;\n");
v[0] = 0;
				printf("v[1] = 0;\n");
v[1] = 0;
				printf("v[2] = 0;\n");
v[2] = 0;
			}	
			*vdatap++ = v[0];
			*vdatap++ = v[1];
			*vdatap++ = v[2];
			*vdatap++ = 0;
		}
	}
	printf("return 0;\n");
	printf("------function end!------\n");
	return 0;
error:
	return -1;
printf("------function end!------\n");
}

static int jas_image_render2(jas_image_t *image, int cmptno, float vtlx,
  float vtly, float vsx, float vsy, int vw, int vh, GLshort *vdata)
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static int jas_image_render2(jas_image_t *image, int cmptno, float vtlx,   float vtly, float vsx, float vsy, int vw, int vh, GLshort *vdata) {\n");
	printf("int i;\n");
	int i;
	printf("int j;\n");
	int j;
	printf("int x;\n");
	int x;
	printf("int y;\n");
	int y;
	printf("int v;\n");
	int v;
	GLshort *vdatap;

	if (cmptno < 0 || cmptno >= image->numcmpts_) {printf("if(cmptno < 0 || cmptno >= image->numcmpts_)\n");
	
		printf("fprintf(stderr, 'bad parameter ');\n");
		fprintf(stderr, "bad parameter\n");
		printf("goto error;\n");
		goto error;
	}
	for (i = 0; i < vh; ++i) {printf("for(i = 0;i < vh;++i)\n");
	
		vdatap = &vdata[(vh - 1 - i) * (4 * vw)];
		for (j = 0; j < vw; ++j) {printf("for(j = 0;j < vw;++j)\n");
		
			printf("x = vctocc(j, jas_image_cmpttlx(image, cmptno), jas_image_cmpthstep(image, cmptno), vtlx, vsx);\n");
x = vctocc(j, jas_image_cmpttlx(image, cmptno), jas_image_cmpthstep(image, cmptno), vtlx, vsx);
			printf("y = vctocc(i, jas_image_cmpttly(image, cmptno), jas_image_cmptvstep(image, cmptno), vtly, vsy);\n");
y = vctocc(i, jas_image_cmpttly(image, cmptno), jas_image_cmptvstep(image, cmptno), vtly, vsy);
			printf("v = (x >= 0 && x < jas_image_cmptwidth(image, cmptno) && y >=0 && y < jas_image_cmptheight(image, cmptno)) ? jas_image_readcmptsample(image, cmptno, x, y) : 0;\n");
v = (x >= 0 && x < jas_image_cmptwidth(image, cmptno) && y >=0 && y < jas_image_cmptheight(image, cmptno)) ? jas_image_readcmptsample(image, cmptno, x, y) : 0;
			printf("v <<= 16 - jas_image_cmptprec(image, cmptno);\n");
v <<= 16 - jas_image_cmptprec(image, cmptno);
			if (v < 0) {printf("if(v < 0)\n");
			
				printf("v = 0;\n");
v = 0;
			} else if (v > 65535) {printf("if(v > 65535)\n");
			
				printf("v = 65535;\n");
v = 65535;
			}
			*vdatap++ = v;
			*vdatap++ = v;
			*vdatap++ = v;
			*vdatap++ = 0;
		}
	}
	printf("return 0;\n");
	printf("------function end!------\n");
	return 0;
error:
	return -1;
printf("------function end!------\n");
}


static void render()
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static void render() {\n");
	printf("float vtlx;\n");
	float vtlx;
	printf("float vtly;\n");
	float vtly;

	printf("vtlx = gs.botleftx;\n");
vtlx = gs.botleftx;
	printf("vtly = gs.toprighty;\n");
vtly = gs.toprighty;
	if (cmdopts.verbose) {printf("if(cmdopts.verbose)\n");
	
//		fprintf(stderr, "vtlx=%f, vtly=%f, vsx=%f, vsy=%f\n",
//		  vtlx, vtly, gs.sx, gs.sy);
	}

	if (gs.monomode) {printf("if(gs.monomode)\n");
	
		if (cmdopts.verbose) {printf("if(cmdopts.verbose)\n");
		
			printf("fprintf(stderr, 'component _d ', gs.cmptno);\n");
			fprintf(stderr, "component %d\n", gs.cmptno);
		}
		jas_image_render2(gs.image, gs.cmptno, 0.0, 0.0,
		  1.0, 1.0, gs.vp.width, gs.vp.height, gs.vp.data);
	} else {
		if (cmdopts.verbose) {printf("if(cmdopts.verbose)\n");
		
			printf("fprintf(stderr, 'color ');\n");
			fprintf(stderr, "color\n");
		}
		jas_image_render(gs.altimage, 0.0, 0.0, 1.0, 1.0,
		  gs.vp.width, gs.vp.height, gs.vp.data);
	}

printf("------function end!------\n");
}

#if 0

#define	vctocc(i, co, cs, vo, vs) \
  (((vo) + (i) * (vs) - (co)) / (cs))

static void drawview(jas_image_t *image, float vtlx, float vtly,
  float sx, float sy, pixmap_t *p)
{
	int i;
	int j;
	int k;
	int red;
	int grn;
	int blu;
	int lum;
	GLshort *datap;
	int x;
	int y;
	int *cmptlut;
	int numcmpts;
	int v[4];
	int u[4];
	int color;

	cmptlut = gs.cmptlut;
	switch (jas_image_colorspace(gs.image)) {
	case JAS_IMAGE_CS_RGB:
	case JAS_IMAGE_CS_YCBCR:
		color = 1;
		numcmpts = 3;
		break;
	case JAS_IMAGE_CS_GRAY:
	default:
		numcmpts = 1;
		color = 0;
		break;
	}

	for (i = 0; i < p->height; ++i) {
		datap = &p->data[(p->height - 1 - i) * (4 * p->width)];
		for (j = 0; j < p->width; ++j) {
			if (!gs.monomode && color) {
				for (k = 0; k < numcmpts; ++k) {
					x = vctocc(j, jas_image_cmpttlx(gs.image, cmptlut[k]), jas_image_cmpthstep(gs.image, cmptlut[k]), vtlx, sx);
					y = vctocc(i, jas_image_cmpttly(gs.image, cmptlut[k]), jas_image_cmptvstep(gs.image, cmptlut[k]), vtly, sy);
					v[k] = (x >= 0 && x < jas_image_cmptwidth(gs.image, cmptlut[k]) && y >=0 && y < jas_image_cmptheight(gs.image, cmptlut[k])) ? jas_matrix_get(gs.cmpts[cmptlut[k]], y, x) : 0;
					v[k] <<= 16 - jas_image_cmptprec(gs.image, cmptlut[k]);
				}
				switch (jas_image_colorspace(gs.image)) {
				case JAS_IMAGE_CS_RGB:
					break;
				case JAS_IMAGE_CS_YCBCR:
					u[0] = (1/1.772) * (v[0] + 1.402 * v[2]);
					u[1] = (1/1.772) * (v[0] - 0.34413 * v[1] - 0.71414 * v[2]);
					u[2] = (1/1.772) * (v[0] + 1.772 * v[1]);
					v[0] = u[0];
					v[1] = u[1];
					v[2] = u[2];
					break;
				}
			} else {
				x = vctocc(j, jas_image_cmpttlx(gs.image, gs.cmptno), jas_image_cmpthstep(gs.image, gs.cmptno), vtlx, sx);
				y = vctocc(i, jas_image_cmpttly(gs.image, gs.cmptno), jas_image_cmptvstep(gs.image, gs.cmptno), vtly, sy);
				v[0] = (x >= 0 && x < jas_image_cmptwidth(gs.image, gs.cmptno) && y >=0 && y < jas_image_cmptheight(gs.image, gs.cmptno)) ? jas_matrix_get(gs.cmpts[gs.cmptno], y, x) : 0;
				v[0] <<= 16 - jas_image_cmptprec(gs.image, gs.cmptno);
				v[1] = v[0];
				v[2] = v[0];
				v[3] = 0;
			}

for (k = 0; k < 3; ++k) {
	if (v[k] < 0) {
		v[k] = 0;
	} else if (v[k] > 65535) {
		v[k] = 65535;
	}
}

			*datap++ = v[0];
			*datap++ = v[1];
			*datap++ = v[2];
			*datap++ = 0;
		}
	}
}

#endif

static void cleanupandexit(int status)
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static void cleanupandexit(int status) {\n");
	printf("unloadimage();\n");
	unloadimage();
	printf("exit(status);\n");
	exit(status);
printf("------function end!------\n");
}

static void init()
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("static void init() {\n");
	printf("gs.filenum = -1;\n");
gs.filenum = -1;
	printf("gs.image = 0;\n");
gs.image = 0;
	printf("gs.altimage = 0;\n");
gs.altimage = 0;
	printf("gs.nexttmid = 0;\n");
gs.nexttmid = 0;
	gs.vp.width = 0;
	gs.vp.height = 0;
	gs.vp.data = 0;
	printf("gs.viewportwidth = -1;\n");
gs.viewportwidth = -1;
	printf("gs.viewportheight = -1;\n");
gs.viewportheight = -1;
printf("------function end!------\n");
}
