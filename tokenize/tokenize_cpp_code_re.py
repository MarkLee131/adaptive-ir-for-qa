import re
import ast
def tokenize_cpp_code(code):
    # Remove newline characters
    code = code.replace('\n', '')

    # Define regular expression pattern for tokenization
    # This pattern includes:
    # - String literals
    # - Comments (single-line and multi-line)
    # - Preprocessor directives
    # - Identifiers and keywords
    # - Operators and punctuation (excluding spaces)
    # - Numeric literals
    pattern = r'(\".*?\"|\'.*?\'|//.*?$|/\*.*?\*/|\#.*?$|\w+|\S)'
    
    # Use re.findall to get all tokens that match the pattern
    tokens = re.findall(pattern, code, re.MULTILINE | re.DOTALL)
    
    # Filter out whitespace tokens
    tokens = [token for token in tokens if token.strip() != '']

    return ' '.join(tokens)

if __name__ == "__main__":
    code = """
    #include <iostream>
    using namespace std;
    
    int main() {
        cout << "Hello World!\n";
        return 0;
    }
    """
    result = tokenize_cpp_code(code)
    print(result)
    