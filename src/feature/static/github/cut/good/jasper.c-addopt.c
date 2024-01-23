int addopt(char *optstr, int maxlen, char *s)
{
	int n;
	int m;

	n = strlen(optstr);
	m = n + strlen(s) + 1;
	if (m > maxlen) {
		return 1;
	}
	if (n > 0) {
		strcat(optstr, "\n");
	}
	strcat(optstr, s);
	return 0;
}