package slice;

import org.eclipse.cdt.core.dom.ast.IASTDeclaration;
import org.eclipse.cdt.internal.core.dom.parser.cpp.CPPASTFunctionDefinition;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;


public class WriteToFile {

    public static void sourceCode(IASTDeclaration decl, String savepath) {
        try {
            if (decl instanceof CPPASTFunctionDefinition) {

                String funContent = decl.getRawSignature();

                FileWriter fw = new FileWriter(new File(savepath));
                BufferedWriter bw = new BufferedWriter(fw);

                bw.write(funContent);
                bw.flush();
            }
        } catch (Exception e) {
            System.out.println("Error in class SourceCodeAndNode.WriteToFile eventually.I dont know why as well.");
            e.printStackTrace();
        }
    }
}
