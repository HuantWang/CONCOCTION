/*
 * Copyright (c) 1999-2000 Image Power, Inc. and the University of
 *   British Columbia.
 * Copyright (c) 2001-2003 Michael David Adams.
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

/*
 * JasPer Transcoder Program
 *
 * $Id$
 */

/******************************************************************************\
* Includes.
\******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include <jasper/jasper.h>

/******************************************************************************\
*
\******************************************************************************/

#define OPTSMAX	4096

/******************************************************************************\
* Types.
\******************************************************************************/

/* Encoder command line options. */

typedef struct {

	char *infile;
	/* The input image file. */

	int infmt;
	/* The input image file format. */

	char *inopts;
	char inoptsbuf[OPTSMAX + 1];

	char *outfile;
	/* The output image file. */

	int outfmt;

	char *outopts;
	char outoptsbuf[OPTSMAX + 1];

	int verbose;
	/* Verbose mode. */

	int debug;

	int version;

	int_fast32_t cmptno;

	int srgb;

} cmdopts_t;

/******************************************************************************\
* Local prototypes.
\******************************************************************************/

cmdopts_t *cmdopts_parse(int argc, char **argv);
void cmdopts_destroy(cmdopts_t *cmdopts);
void cmdusage(void);
void badusage(void);
void cmdinfo(void);
int addopt(char *optstr, int maxlen, char *s);

/******************************************************************************\
* Global data.
\******************************************************************************/

char *cmdname = "";

/******************************************************************************\
* Code.
\******************************************************************************/

int main(int argc, char **argv)
{
	jas_image_t *image;
	cmdopts_t *cmdopts;
	jas_stream_t *in;
	jas_stream_t *out;
	jas_tmr_t dectmr;
	jas_tmr_t enctmr;
	double dectime;
	double enctime;
	int_fast16_t numcmpts;
	int i;

	/* Determine the base name of this command. */
	if ((cmdname = strrchr(argv[0], '/'))) {
		++cmdname;
	} else {
		cmdname = argv[0];
	}

	if (jas_init()) {
		abort();
	}

	/* Parse the command line options. */
	if (!(cmdopts = cmdopts_parse(argc, argv))) {
		fprintf(stderr, "error: cannot parse command line\n");
		exit(EXIT_FAILURE);
	}

	if (cmdopts->version) {
		printf("%s\n", JAS_VERSION);
		fprintf(stderr, "libjasper %s\n", jas_getversion());
		exit(EXIT_SUCCESS);
	}

	jas_setdbglevel(cmdopts->debug);

	if (cmdopts->verbose) {
		cmdinfo();
	}

	/* Open the input image file. */
	if (cmdopts->infile) {
		/* The input image is to be read from a file. */
		if (!(in = jas_stream_fopen(cmdopts->infile, "rb"))) {
			fprintf(stderr, "error: cannot open input image file %s\n",
			  cmdopts->infile);
			exit(EXIT_FAILURE);
		}
	} else {
		/* The input image is to be read from standard input. */
		if (!(in = jas_stream_fdopen(0, "rb"))) {
			fprintf(stderr, "error: cannot open standard input\n");
			exit(EXIT_FAILURE);
		}
	}

	/* Open the output image file. */
	if (cmdopts->outfile) {
		/* The output image is to be written to a file. */
		if (!(out = jas_stream_fopen(cmdopts->outfile, "w+b"))) {
			fprintf(stderr, "error: cannot open output image file %s\n",
			  cmdopts->outfile);
			exit(EXIT_FAILURE);
		}
	} else {
		/* The output image is to be written to standard output. */
		if (!(out = jas_stream_fdopen(1, "w+b"))) {
			fprintf(stderr, "error: cannot open standard output\n");
			exit(EXIT_FAILURE);
		}
	}

	if (cmdopts->infmt < 0) {
		if ((cmdopts->infmt = jas_image_getfmt(in)) < 0) {
			fprintf(stderr, "error: input image has unknown format\n");
			exit(EXIT_FAILURE);
		}
	}

	/* Get the input image data. */
	jas_tmr_start(&dectmr);
	if (!(image = jas_image_decode(in, cmdopts->infmt, cmdopts->inopts))) {
		fprintf(stderr, "error: cannot load image data\n");
		exit(EXIT_FAILURE);
	}
	jas_tmr_stop(&dectmr);
	dectime = jas_tmr_get(&dectmr);

	/* If requested, throw away all of the components except one.
	  Why might this be desirable?  It is a hack, really.
	  None of the image formats other than the JPEG-2000 ones support
	  images with two, four, five, or more components.  This hack
	  allows such images to be decoded with the non-JPEG-2000 decoders,
	  one component at a time. */
	numcmpts = jas_image_numcmpts(image);
	if (cmdopts->cmptno >= 0 && cmdopts->cmptno < numcmpts) {
		for (i = numcmpts - 1; i >= 0; --i) {
			if (i != cmdopts->cmptno) {
				jas_image_delcmpt(image, i);
			}
		}
	}

	if (cmdopts->srgb) {
		jas_image_t *newimage;
		jas_cmprof_t *outprof;
		jas_eprintf("forcing conversion to sRGB\n");
		if (!(outprof = jas_cmprof_createfromclrspc(JAS_CLRSPC_SRGB))) {
			jas_eprintf("cannot create sRGB profile\n");
			exit(EXIT_FAILURE);
		}
		if (!(newimage = jas_image_chclrspc(image, outprof, JAS_CMXFORM_INTENT_PER))) {
			jas_eprintf("cannot convert to sRGB\n");
			exit(EXIT_FAILURE);
		}
		jas_image_destroy(image);
		jas_cmprof_destroy(outprof);
		image = newimage;
	}

	/* Generate the output image data. */
	jas_tmr_start(&enctmr);
	if (jas_image_encode(image, out, cmdopts->outfmt, cmdopts->outopts)) {
		fprintf(stderr, "error: cannot encode image\n");
		exit(EXIT_FAILURE);
	}
	jas_stream_flush(out);
	jas_tmr_stop(&enctmr);
	enctime = jas_tmr_get(&enctmr);

	if (cmdopts->verbose) {
		fprintf(stderr, "decoding time = %f\n", dectime);
		fprintf(stderr, "encoding time = %f\n", enctime);
	}

	/* If this fails, we don't care. */
	(void) jas_stream_close(in);

	/* Close the output image stream. */
	if (jas_stream_close(out)) {
		fprintf(stderr, "error: cannot close output image file\n");
		exit(EXIT_FAILURE);
	}

	cmdopts_destroy(cmdopts);
	jas_image_destroy(image);
	jas_image_clearfmts();

	/* Success at last! :-) */
	return EXIT_SUCCESS;
}

cmdopts_t *cmdopts_parse(int argc, char **argv)
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("cmdopts_t *cmdopts_parse(int argc, char **argv) {\n");

	printf("typedef enum { 		CMDOPT_HELP = 0, 		CMDOPT_VERBOSE, 		CMDOPT_INFILE, 		CMDOPT_INFMT, 		CMDOPT_INOPT, 		CMDOPT_OUTFILE, 		CMDOPT_OUTFMT, 		CMDOPT_OUTOPT, 		CMDOPT_VERSION, 		CMDOPT_DEBUG, 		CMDOPT_CMPTNO, 		CMDOPT_SRGB 	} cmdoptid_t;\n");
	typedef enum {
		CMDOPT_HELP = 0,
		CMDOPT_VERBOSE,
		CMDOPT_INFILE,
		CMDOPT_INFMT,
		CMDOPT_INOPT,
		CMDOPT_OUTFILE,
		CMDOPT_OUTFMT,
		CMDOPT_OUTOPT,
		CMDOPT_VERSION,
		CMDOPT_DEBUG,
		CMDOPT_CMPTNO,
		CMDOPT_SRGB
	} cmdoptid_t;

	printf("static jas_opt_t cmdoptions[] = { 		{CMDOPT_HELP, 'help', 0}, 		{CMDOPT_VERBOSE, 'verbose', 0}, 		{CMDOPT_INFILE, 'input', JAS_OPT_HASARG}, 		{CMDOPT_INFILE, 'f', JAS_OPT_HASARG}, 		{CMDOPT_INFMT, 'input-format', JAS_OPT_HASARG}, 		{CMDOPT_INFMT, 't', JAS_OPT_HASARG}, 		{CMDOPT_INOPT, 'input-option', JAS_OPT_HASARG}, 		{CMDOPT_INOPT, 'o', JAS_OPT_HASARG}, 		{CMDOPT_OUTFILE, 'output', JAS_OPT_HASARG}, 		{CMDOPT_OUTFILE, 'F', JAS_OPT_HASARG}, 		{CMDOPT_OUTFMT, 'output-format', JAS_OPT_HASARG}, 		{CMDOPT_OUTFMT, 'T', JAS_OPT_HASARG}, 		{CMDOPT_OUTOPT, 'output-option', JAS_OPT_HASARG}, 		{CMDOPT_OUTOPT, 'O', JAS_OPT_HASARG}, 		{CMDOPT_VERSION, 'version', 0}, 		{CMDOPT_DEBUG, 'debug-level', JAS_OPT_HASARG}, 		{CMDOPT_CMPTNO, 'cmptno', JAS_OPT_HASARG}, 		{CMDOPT_SRGB, 'force-srgb', 0}, 		{CMDOPT_SRGB, 'S', 0}, 		{-1, 0, 0} 	};\n");
	static jas_opt_t cmdoptions[] = {
		{CMDOPT_HELP, "help", 0},
		{CMDOPT_VERBOSE, "verbose", 0},
		{CMDOPT_INFILE, "input", JAS_OPT_HASARG},
		{CMDOPT_INFILE, "f", JAS_OPT_HASARG},
		{CMDOPT_INFMT, "input-format", JAS_OPT_HASARG},
		{CMDOPT_INFMT, "t", JAS_OPT_HASARG},
		{CMDOPT_INOPT, "input-option", JAS_OPT_HASARG},
		{CMDOPT_INOPT, "o", JAS_OPT_HASARG},
		{CMDOPT_OUTFILE, "output", JAS_OPT_HASARG},
		{CMDOPT_OUTFILE, "F", JAS_OPT_HASARG},
		{CMDOPT_OUTFMT, "output-format", JAS_OPT_HASARG},
		{CMDOPT_OUTFMT, "T", JAS_OPT_HASARG},
		{CMDOPT_OUTOPT, "output-option", JAS_OPT_HASARG},
		{CMDOPT_OUTOPT, "O", JAS_OPT_HASARG},
		{CMDOPT_VERSION, "version", 0},
		{CMDOPT_DEBUG, "debug-level", JAS_OPT_HASARG},
		{CMDOPT_CMPTNO, "cmptno", JAS_OPT_HASARG},
		{CMDOPT_SRGB, "force-srgb", 0},
		{CMDOPT_SRGB, "S", 0},
		{-1, 0, 0}
	};

	printf("cmdopts_t *cmdopts;\n");
	cmdopts_t *cmdopts;
	printf("int c;\n");
	int c;

	if (!(cmdopts = malloc(sizeof(cmdopts_t)))) {printf("if(!(cmdopts = malloc(sizeof(cmdopts_t))))\n");
	
		printf("fprintf(stderr, 'error: insufficient memory ');\n");
		fprintf(stderr, "error: insufficient memory\n");
		printf("exit(EXIT_FAILURE);\n");
		exit(EXIT_FAILURE);
	}

	printf("cmdopts->infile = 0;\n");
cmdopts->infile = 0;
	printf("cmdopts->infmt = -1;\n");
cmdopts->infmt = -1;
	printf("cmdopts->inopts = 0;\n");
cmdopts->inopts = 0;
	printf("cmdopts->inoptsbuf[0] = '\0';\n");
cmdopts->inoptsbuf[0] = '\0';
	printf("cmdopts->outfile = 0;\n");
cmdopts->outfile = 0;
	printf("cmdopts->outfmt = -1;\n");
cmdopts->outfmt = -1;
	printf("cmdopts->outopts = 0;\n");
cmdopts->outopts = 0;
	printf("cmdopts->outoptsbuf[0] = '\0';\n");
cmdopts->outoptsbuf[0] = '\0';
	printf("cmdopts->verbose = 0;\n");
cmdopts->verbose = 0;
	printf("cmdopts->version = 0;\n");
cmdopts->version = 0;
	printf("cmdopts->cmptno = -1;\n");
cmdopts->cmptno = -1;
	printf("cmdopts->debug = 0;\n");
cmdopts->debug = 0;
	printf("cmdopts->srgb = 0;\n");
cmdopts->srgb = 0;

	while ((c = jas_getopt(argc, argv, cmdoptions)) != EOF) {printf("while((c = jas_getopt(argc, argv, cmdoptions)) != EOF)\n");
	
		printf("switch(c)\n");
		switch (c) {
		case CMDOPT_HELP:
			printf("cmdusage();\n");
			cmdusage();
			break;
		case CMDOPT_VERBOSE:
			printf("cmdopts->verbose = 1;\n");
cmdopts->verbose = 1;
			break;
		case CMDOPT_VERSION:
			printf("cmdopts->version = 1;\n");
cmdopts->version = 1;
			break;
		case CMDOPT_DEBUG:
			printf("cmdopts->debug = atoi(jas_optarg);\n");
cmdopts->debug = atoi(jas_optarg);
			break;
		case CMDOPT_INFILE:
			printf("cmdopts->infile = jas_optarg;\n");
cmdopts->infile = jas_optarg;
			break;
		case CMDOPT_INFMT:
			if ((cmdopts->infmt = jas_image_strtofmt(jas_optarg)) < 0) {printf("if((cmdopts->infmt = jas_image_strtofmt(jas_optarg)) < 0)\n");
			
				printf("fprintf(stderr, 'warning: ignoring invalid input format _s ', 				  jas_optarg);\n");
				fprintf(stderr, "warning: ignoring invalid input format %s\n",
				  jas_optarg);
				printf("cmdopts->infmt = -1;\n");
cmdopts->infmt = -1;
			}
			break;
		case CMDOPT_INOPT:
			printf("addopt(cmdopts->inoptsbuf, OPTSMAX, jas_optarg);\n");
			addopt(cmdopts->inoptsbuf, OPTSMAX, jas_optarg);
			printf("cmdopts->inopts = cmdopts->inoptsbuf;\n");
cmdopts->inopts = cmdopts->inoptsbuf;
			break;
		case CMDOPT_OUTFILE:
			printf("cmdopts->outfile = jas_optarg;\n");
cmdopts->outfile = jas_optarg;
			break;
		case CMDOPT_OUTFMT:
			if ((cmdopts->outfmt = jas_image_strtofmt(jas_optarg)) < 0) {printf("if((cmdopts->outfmt = jas_image_strtofmt(jas_optarg)) < 0)\n");
			
				printf("fprintf(stderr, 'error: invalid output format _s ', jas_optarg);\n");
				fprintf(stderr, "error: invalid output format %s\n", jas_optarg);
				printf("badusage();\n");
				badusage();
			}
			break;
		case CMDOPT_OUTOPT:
			printf("addopt(cmdopts->outoptsbuf, OPTSMAX, jas_optarg);\n");
			addopt(cmdopts->outoptsbuf, OPTSMAX, jas_optarg);
			printf("cmdopts->outopts = cmdopts->outoptsbuf;\n");
cmdopts->outopts = cmdopts->outoptsbuf;
			break;
		case CMDOPT_CMPTNO:
			printf("cmdopts->cmptno = atoi(jas_optarg);\n");
cmdopts->cmptno = atoi(jas_optarg);
			break;
		case CMDOPT_SRGB:
			printf("cmdopts->srgb = 1;\n");
cmdopts->srgb = 1;
			break;
		default:
			printf("badusage();\n");
			badusage();
			break;
		}
	}

	while (jas_optind < argc) {printf("while(jas_optind < argc)\n");
	
		printf("fprintf(stderr, 		  'warning: ignoring bogus command line argument _s ', 		  argv[jas_optind]);\n");
		fprintf(stderr,
		  "warning: ignoring bogus command line argument %s\n",
		  argv[jas_optind]);
		printf("++jas_optind;\n");
++jas_optind;
	}

	if (cmdopts->version) {printf("if(cmdopts->version)\n");
	
		printf("goto done;\n");
		goto done;
	}

	if (cmdopts->outfmt < 0 && cmdopts->outfile) {printf("if(cmdopts->outfmt < 0 && cmdopts->outfile)\n");
	
		if ((cmdopts->outfmt = jas_image_fmtfromname(cmdopts->outfile)) < 0) {printf("if((cmdopts->outfmt = jas_image_fmtfromname(cmdopts->outfile)) < 0)\n");
		
			printf("fprintf(stderr, 			  'error: cannot guess image format from output file name ');\n");
			fprintf(stderr,
			  "error: cannot guess image format from output file name\n");
		}
	}

	if (cmdopts->outfmt < 0) {printf("if(cmdopts->outfmt < 0)\n");
	
		printf("fprintf(stderr, 'error: no output format specified ');\n");
		fprintf(stderr, "error: no output format specified\n");
		printf("badusage();\n");
		badusage();
	}

done:
	return cmdopts;
printf("------function end!------\n");
}

void cmdopts_destroy(cmdopts_t *cmdopts)
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("void cmdopts_destroy(cmdopts_t *cmdopts) {\n");
	printf("free(cmdopts);\n");
	free(cmdopts);
printf("------function end!------\n");
}

int addopt(char *optstr, int maxlen, char *s)
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("int addopt(char *optstr, int maxlen, char *s) {\n");
	printf("int n;\n");
	int n;
	printf("int m;\n");
	int m;

	printf("n = strlen(optstr);\n");
n = strlen(optstr);
	printf("m = n + strlen(s) + 1;\n");
m = n + strlen(s) + 1;
	if (m > maxlen) {printf("if(m > maxlen)\n");
	
		printf("return 1;\n");
		printf("------function end!------\n");
		return 1;
	}
	if (n > 0) {printf("if(n > 0)\n");
	
		printf("strcat(optstr, ' ');\n");
		strcat(optstr, "\n");
	}
	printf("strcat(optstr, s);\n");
	strcat(optstr, s);
	printf("return 0;\n");
	printf("------function end!------\n");
	return 0;
printf("------function end!------\n");
}

void cmdinfo()
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("void cmdinfo() {\n");
	printf("fprintf(stderr, 'JasPer Transcoder (Version _s). ', 	  JAS_VERSION);\n");
	fprintf(stderr, "JasPer Transcoder (Version %s).\n",
	  JAS_VERSION);
	printf("fprintf(stderr, '_s ', JAS_COPYRIGHT);\n");
	fprintf(stderr, "%s\n", JAS_COPYRIGHT);
	printf("fprintf(stderr, '_s ', JAS_NOTES);\n");
	fprintf(stderr, "%s\n", JAS_NOTES);
printf("------function end!------\n");
}

static char *helpinfo[] = {
"The following options are supported:\n",
"    --help                  Print this help information and exit.\n",
"    --version               Print version information and exit.\n",
"    --verbose               Enable verbose mode.\n",
"    --debug-level $lev      Set the debug level to $lev.\n",
"    --input $file           Read the input image from the file named $file\n",
"                            instead of standard input.\n",
"    --input-format $fmt     Specify the format of the input image as $fmt.\n",
"                            (See below for the list of supported formats.)\n",
"    --input-option $opt     Provide the option $opt to the decoder.\n",
"    --output $file          Write the output image to the file named $file\n",
"                            instead of standard output.\n",
"    --output-format $fmt    Specify the format of the output image as $fmt.\n",
"                            (See below for the list of supported formats.)\n",
"    --output-option $opt    Provide the option $opt to the encoder.\n",
"    --force-srgb            Force conversion to the sRGB color space.\n",
"Some of the above option names can be abbreviated as follows:\n",
"    --input = -f, --input-format = -t, --input-option = -o,\n",
"    --output = -F, --output-format = -T, --output-option = -O\n",
0
};

void cmdusage()
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("void cmdusage() {\n");
	printf("int fmtid;\n");
	int fmtid;
	printf("jas_image_fmtinfo_t *fmtinfo;\n");
	jas_image_fmtinfo_t *fmtinfo;
	printf("char *s;\n");
	char *s;
	printf("int i;\n");
	int i;
	printf("cmdinfo();\n");
	cmdinfo();
	printf("fprintf(stderr, 'usage: _s [options] ', cmdname);\n");
	fprintf(stderr, "usage: %s [options]\n", cmdname);
	for (i = 0, s = helpinfo[i]; s; ++i, s = helpinfo[i]) {printf("for(i = 0, s = helpinfo[i];s;++i, s = helpinfo[i])\n");
	
		printf("fprintf(stderr, '_s', s);\n");
		fprintf(stderr, "%s", s);
	}
	printf("fprintf(stderr, 'The following formats are supported: ');\n");
	fprintf(stderr, "The following formats are supported:\n");
	for (fmtid = 0;; ++fmtid) {
		if (!(fmtinfo = jas_image_lookupfmtbyid(fmtid))) {printf("if(!(fmtinfo = jas_image_lookupfmtbyid(fmtid)))\n");
		
			break;
		}
		printf("fprintf(stderr, '    _-5s    _s ', fmtinfo->name, 		  fmtinfo->desc);\n");
		fprintf(stderr, "    %-5s    %s\n", fmtinfo->name,
		  fmtinfo->desc);
	}
	printf("exit(EXIT_FAILURE);\n");
	exit(EXIT_FAILURE);
printf("------function end!------\n");
}

void badusage()
{
printf("\nfile_name:%s\n",__FILE__);
printf("function_name:%s\n",__func__);
printf("------function start!------\n");
printf("void badusage() {\n");
	printf("fprintf(stderr, 	  'For more information on how to use this command, type: ');\n");
	fprintf(stderr,
	  "For more information on how to use this command, type:\n");
	printf("fprintf(stderr, '    _s --help ', cmdname);\n");
	fprintf(stderr, "    %s --help\n", cmdname);
	printf("exit(EXIT_FAILURE);\n");
	exit(EXIT_FAILURE);
printf("------function end!------\n");
}

#if 0
jas_image_t *converttosrgb(jas_image_t *inimage)
{
	jas_image_t *outimage;
	jas_cmpixmap_t inpixmap;
	jas_cmpixmap_t outpixmap;
	jas_cmcmptfmt_t incmptfmts[16];
	jas_cmcmptfmt_t outcmptfmts[16];

	outprof = jas_cmprof_createfromclrspc(JAS_CLRSPC_SRGB);
	assert(outprof);
	xform = jas_cmxform_create(jas_image_cmprof(inimage), outprof, 0, JAS_CMXFORM_FWD, JAS_CMXFORM_INTENT_PER, 0);
	assert(xform);

	inpixmap.numcmpts = jas_image_numcmpts(oldimage);
	outpixmap.numcmpts = 3;
	for (i = 0; i < inpixmap.numcmpts; ++i) {
		inpixmap.cmptfmts[i] = &incmptfmts[i];
	}
	for (i = 0; i < outpixmap.numcmpts; ++i)
		outpixmap.cmptfmts[i] = &outcmptfmts[i];
	if (jas_cmxform_apply(xform, &inpixmap, &outpixmap))
		abort();

	jas_xform_destroy(xform);
	jas_cmprof_destroy(outprof);
	return 0;
}
#endif
