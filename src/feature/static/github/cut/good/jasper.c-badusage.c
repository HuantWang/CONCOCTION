void badusage()
{
	fprintf(stderr,
	  "For more information on how to use this command, type:\n");
	fprintf(stderr, "    %s --help\n", cmdname);
	exit(EXIT_FAILURE);
}