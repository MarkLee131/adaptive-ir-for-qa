import antlr4
from antlr4_cpp.CPP14Lexer import CPP14Lexer

def tokenize_cpp_code(code):
    input_stream = antlr4.InputStream(code)
    lexer = CPP14Lexer(input_stream)
    token_stream = antlr4.CommonTokenStream(lexer)
    token_stream.fill()  # Filling the token stream with tokens from the lexer
    
    tokens = []
    for token in token_stream.tokens:
        # Appending text of token to tokens list
        # Exclude EOF token (it has token.type = -1)
        if token.type != -1:
            tokens.append(token.text)
    
    return tokens

if __name__ == "__main__":
    code = """
    #include <iostream>
    using namespace std;
    
    int main() {
        cout << "Hello World!";
        return 0;
    }
    """
    result = ''.join(tokenize_cpp_code(code))
    print(result)