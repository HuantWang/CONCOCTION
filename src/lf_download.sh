#/bin/bash
#download largefile
SCRIPT_ABS_PATH=$(readlink -f "$0")
SCRIPT_ABS_DIR=$(dirname "$SCRIPT_ABS_PATH")



id='12aN4hd3EfVNJ6N8EW6yPuE4FMwe-DQF0'
confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
echo $confirm
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$confirm&id=$id" -O lf.zip && rm -rf /tmp/cookies.txt

unzip lf.zip -d ./lf
cd lf
cp ./graphcodebert-base/pytorch_model.bin $SCRIPT_ABS_DIR/concoction/detectionModel/transformer/graphcodebert-base/pytorch_model.bin
cp ./graphcodebert-base/pytorch_model.bin $SCRIPT_ABS_DIR/concoction/pretrainedModel/staticRepresentation/graphcodebert-base/pytorch_model.bin
cp ./bert-base-uncased/pytorch_model.bin $SCRIPT_ABS_DIR/concoction/pretrainedModel/dynamicRepresentation/bert-base-uncased/pytorch_model.bin
cp ./bert-base-uncased/pytorch_model.bin $SCRIPT_ABS_DIR/concoction/detectionModel/transformer/bert-base-uncased/pytorch_model.bin
cp ./sup-simcse-bert-base-uncased/flax_model.msgpack $SCRIPT_ABS_DIR/concoction/pathSelection/princeton-nlp/sup-simcse-bert-base-uncased/flax_model.msgpack
cp ./sup-simcse-bert-base-uncased/pytorch_model.bin $SCRIPT_ABS_DIR/concoction/pathSelection/princeton-nlp/sup-simcse-bert-base-uncased/pytorch_model.bin
cp ./lib.zip $SCRIPT_ABS_DIR/feature/static/github/joern-cli_new/lib.zip
cp ./io.shiftleft.ghidra-10.1_PUBLIC_20211210a.jar  $SCRIPT_ABS_DIR/feature/static/github/joern-cli_new/frontends/ghidra2cpg/lib/io.shiftleft.ghidra-10.1_PUBLIC_20211210a.jar   
cp ./astgen-linux $SCRIPT_ABS_DIR/feature/static/github/joern-cli_new/frontends/jssrc2cpg/bin/astgen/astgen-linux
cp ./clang-addBrace $SCRIPT_ABS_DIR/feature/dynamic/instrument/clangTool/cmake-build-debug/bin/clang-addBrace
cp ./clang-diff $SCRIPT_ABS_DIR/feature/dynamic/instrument/clangTool/cmake-build-debug/bin/clang-diff

