import antlr4
from CPP14Lexer import CPP14Lexer

def tokenize_cpp_code(code):
    input_stream = antlr4.InputStream(code)
    lexer = CPP14Lexer(input_stream)
    token_stream = antlr4.CommonTokenStream(lexer)
    tokens = []
    for token in token_stream:
        tokens.append(token.text)
    return tokens
