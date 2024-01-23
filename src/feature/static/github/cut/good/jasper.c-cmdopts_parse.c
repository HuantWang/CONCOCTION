cmdopts_t *cmdopts_parse(int argc, char **argv)
{

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

	cmdopts_t *cmdopts;
	int c;

	if (!(cmdopts = malloc(sizeof(cmdopts_t)))) {
		fprintf(stderr, "error: insufficient memory\n");
		exit(EXIT_FAILURE);
	}

	cmdopts->infile = 0;
	cmdopts->infmt = -1;
	cmdopts->inopts = 0;
	cmdopts->inoptsbuf[0] = '\0';
	cmdopts->outfile = 0;
	cmdopts->outfmt = -1;
	cmdopts->outopts = 0;
	cmdopts->outoptsbuf[0] = '\0';
	cmdopts->verbose = 0;
	cmdopts->version = 0;
	cmdopts->cmptno = -1;
	cmdopts->debug = 0;
	cmdopts->srgb = 0;

	while ((c = jas_getopt(argc, argv, cmdoptions)) != EOF) {
		switch (c) {
		case CMDOPT_HELP:
			cmdusage();
			break;
		case CMDOPT_VERBOSE:
			cmdopts->verbose = 1;
			break;
		case CMDOPT_VERSION:
			cmdopts->version = 1;
			break;
		case CMDOPT_DEBUG:
			cmdopts->debug = atoi(jas_optarg);
			break;
		case CMDOPT_INFILE:
			cmdopts->infile = jas_optarg;
			break;
		case CMDOPT_INFMT:
			if ((cmdopts->infmt = jas_image_strtofmt(jas_optarg)) < 0) {
				fprintf(stderr, "warning: ignoring invalid input format %s\n",
				  jas_optarg);
				cmdopts->infmt = -1;
			}
			break;
		case CMDOPT_INOPT:
			addopt(cmdopts->inoptsbuf, OPTSMAX, jas_optarg);
			cmdopts->inopts = cmdopts->inoptsbuf;
			break;
		case CMDOPT_OUTFILE:
			cmdopts->outfile = jas_optarg;
			break;
		case CMDOPT_OUTFMT:
			if ((cmdopts->outfmt = jas_image_strtofmt(jas_optarg)) < 0) {
				fprintf(stderr, "error: invalid output format %s\n", jas_optarg);
				badusage();
			}
			break;
		case CMDOPT_OUTOPT:
			addopt(cmdopts->outoptsbuf, OPTSMAX, jas_optarg);
			cmdopts->outopts = cmdopts->outoptsbuf;
			break;
		case CMDOPT_CMPTNO:
			cmdopts->cmptno = atoi(jas_optarg);
			break;
		case CMDOPT_SRGB:
			cmdopts->srgb = 1;
			break;
		default:
			badusage();
			break;
		}
	}

	while (jas_optind < argc) {
		fprintf(stderr,
		  "warning: ignoring bogus command line argument %s\n",
		  argv[jas_optind]);
		++jas_optind;
	}

	if (cmdopts->version) {
		goto done;
	}

	if (cmdopts->outfmt < 0 && cmdopts->outfile) {
		if ((cmdopts->outfmt = jas_image_fmtfromname(cmdopts->outfile)) < 0) {
			fprintf(stderr,
			  "error: cannot guess image format from output file name\n");
		}
	}

	if (cmdopts->outfmt < 0) {
		fprintf(stderr, "error: no output format specified\n");
		badusage();
	}

done:
	return cmdopts;
}