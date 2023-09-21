package slice;
public class Main{
    public static void main(String[] args) {
        String spath=args[0];

        String sourceFilePath = spath+"/out";
        String storeFilePath = spath+"/cut";

        slice.AST_extract ast_extract = new slice.AST_extract();

        int batchsize = 10;
        int batchnum_lastrun = 0;

        //sard_good_bad
        //ast_extract.AST_SardGoodBad(sourceFilePath, storeFilePath,batchnum_lastrun,batchsize);
        //slice all c files without good or bad
        ast_extract.AST_Slice(sourceFilePath, storeFilePath,batchnum_lastrun,batchsize);

    }
}
