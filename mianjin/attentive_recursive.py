'''
Build an lisp expression parser
// Examples:
// parse("( ADD 3 4 )") == 7
// parse("( MULT 3 4 )") == 12
// parse("( MULT 3 ( ADD 3 4 ) )") == 21
'''

# 完整版答案，考虑错误input
def parse_whole(expr):
    def eval_expr(tokens):
        if not tokens:
            raise ValueError("Empty expression")

        # Base case: if the list has only one element, try converting it to an integer
        if len(tokens) == 1:
            try:
                return int(tokens[0])
            except ValueError:
                raise ValueError(f"Invalid token: {tokens[0]}")

        op = tokens[0]
        if op not in ('ADD', 'MULT'):
            raise ValueError(f"Unsupported operation: {op}")

        if op == 'ADD':
            return sum(eval_expr([t]) for t in tokens[1:])
        elif op == 'MULT':
            result = 1
            for t in tokens[1:]:
                result *= eval_expr([t])
            return result

    # Tokenize the input expression
    # Remove the outermost parentheses, split by space, and filter out empty tokens
    tokens = [t for t in expr.strip('()').split(' ') if t]

    return eval_expr(tokens)



def parse(expr):
    def eval_expr(tokens):
        # Base case: if the list has only one element, try converting it to an integer
        if len(tokens) == 1:
            return int(tokens[0])


        op = tokens[0]
        if op == 'ADD':
            return sum(eval_expr([t]) for t in tokens[1:])
        elif op == 'MULT':
            result = 1
            for t in tokens[1:]:
                result *= eval_expr([t])
            return result

    # Tokenize the input expression
    # Remove the outermost parentheses, split by space, and filter out empty tokens
    tokens = [t for t in expr.strip('()').split(' ') if t]

    return eval_expr(tokens)

# Test the parser
print(parse("( ADD 3 4 )"))  # Output: 7
print(parse("( MULT 3 4 )"))  # Output: 12
print(parse("( MULT 3 ( ADD 3 4 ) )"))  # Output: 21
