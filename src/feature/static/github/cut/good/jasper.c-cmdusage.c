void cmdusage()
{
	int fmtid;
	jas_image_fmtinfo_t *fmtinfo;
	char *s;
	int i;
	cmdinfo();
	fprintf(stderr, "usage: %s [options]\n", cmdname);
	for (i = 0, s = helpinfo[i]; s; ++i, s = helpinfo[i]) {
		fprintf(stderr, "%s", s);
	}
	fprintf(stderr, "The following formats are supported:\n");
	for (fmtid = 0;; ++fmtid) {
		if (!(fmtinfo = jas_image_lookupfmtbyid(fmtid))) {
			break;
		}
		fprintf(stderr, "    %-5s    %s\n", fmtinfo->name,
		  fmtinfo->desc);
	}
	exit(EXIT_FAILURE);
}