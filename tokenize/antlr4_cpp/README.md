# Antlr4

## Antlr-4 installation
To generate the parser and lexer for the C++14 grammar, 
we need to install antlr4, [antlr4 grammar files](https://github.com/antlr/grammars-v4/tree/master/cpp) in advance.

### Install antlr4

```bash
(patch_match) kaixuan_cuda11@kaixuan_cuda11-U2141:/mnt/local/Baselines_Bugs/CodeBert$ pip install antlr4-tools
Collecting antlr4-tools
  Downloading antlr4_tools-0.2-py3-none-any.whl (4.1 kB)
Collecting install-jdk
  Downloading install_jdk-1.0.4-py3-none-any.whl (15 kB)
Installing collected packages: install-jdk, antlr4-tools
Successfully installed antlr4-tools-0.2 install-jdk-1.0.4
(patch_match) kaixuan_cuda11@kaixuan_cuda11-U2141:/mnt/local/Baselines_Bugs/CodeBert$ antlr4
Downloading antlr4-4.13.0-complete.jar
ANTLR tool needs Java to run; install Java JRE 11 yes/no (default yes)? y
Installed Java in /home/kaixuan_cuda11/.jre/jdk-11.0.19+7-jre; remove that dir to uninstall
ANTLR Parser Generator  Version 4.13.0
 -o ___              specify output directory where all output is generated
 -lib ___            specify location of grammars, tokens files
 -atn                generate rule augmented transition network diagrams
 -encoding ___       specify grammar file encoding; e.g., euc-jp
 -message-format ___ specify output style for messages in antlr, gnu, vs2005
 -long-messages      show exception details when available for errors and warnings
 -listener           generate parse tree listener (default)
 -no-listener        don't generate parse tree listener
 -visitor            generate parse tree visitor
 -no-visitor         don't generate parse tree visitor (default)
 -package ___        specify a package/namespace for the generated code
 -depend             generate file dependencies
 -D<option>=value    set/override a grammar-level option
 -Werror             treat warnings as errors
 -XdbgST             launch StringTemplate visualizer on generated code
 -XdbgSTWait         wait for STViz to close before continuing
 -Xforce-atn         use the ATN simulator for all predictions
 -Xlog               dump lots of logging info to antlr-timestamp.log
 -Xexact-output-dir  all output goes into -o dir regardless of paths/package
```

### Install antlr4 grammar files

Then, we should generate the parser and lexer for the C++14 grammar.

1. First get the antlr4 runtime for python3
```bash
pip install antlr4-python3-runtime
```

2. Use the ANTLR tool to generate the Python files needed for tokenizing and parsing C++ code. We can do this by running the following commands in the terminal:

```bash
antlr4 -Dlanguage=Python3 CPP14Lexer.g4
antlr4 -Dlanguage=Python3 CPP14Parser.g4
```

This will generate several Python files including CPP14Lexer.py and CPP14Parser.py among others.

Now, we can use these generated files in the Python script to tokenize the C++ diff code. 