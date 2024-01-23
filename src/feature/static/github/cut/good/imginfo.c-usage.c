static void usage()
{
	cmdinfo();
	fprintf(stderr, "usage:\n");
	fprintf(stderr,"%s ", cmdname);
	fprintf(stderr, "[-f image_file]\n");
	exit(EXIT_FAILURE);
}