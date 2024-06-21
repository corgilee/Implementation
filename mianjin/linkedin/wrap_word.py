def justify_text(sentence, n, m):
    words = sentence.split()
    lines = [[] for _ in range(n)]
    current_row_index = 0
    current_length = 0

    # Distribute words to lines
    for word in words:
        if current_length + len(word) + len(lines[current_row_index])-1 <= m:
            #len(lines[current_row_index])-1 我理解是有n个单词，对应n-1个空格
            lines[current_row_index].append(word)
            current_length += len(word)
        else:
            current_row_index += 1
            if current_row_index >= n:
                raise ValueError("The given sentence cannot be justified into the specified number of lines and length.")
            lines[current_row_index].append(word)
            current_length = len(word)

    # Justify each line
    justified_lines = [] #每一行是一个string， 存进去
    for line in lines:
        if len(line) == 1:  # Single word line, just append spaces to the end
            justified_lines.append(line[0] + ' ' * (m - len(line[0])))
        else:
            total_length = sum(len(word) for word in line) # 要算出以后单词的总长度
            total_spaces = m - total_length #还有多少空余
            min_spaces = total_spaces // (len(line) - 1) #(单词数-1)
            extra_spaces = total_spaces % (len(line) - 1) # extra_space 就是要在min_space 上还要多加的space

            justified_line = line[0]
            for i in range(1, len(line)):
                if i <= extra_spaces:
                    #只要i<=extra_space ，每个单词要多加一个空格
                    justified_line += ' ' * (min_spaces + 1) + line[i]
                else:
                    justified_line += ' ' * min_spaces + line[i]
            justified_lines.append(justified_line)

    return justified_lines

# Example usage
sentence = "To justify a sentence into n lines each with exactly m length words can be spaced out."
n = 3
m = 33
justified_lines = justify_text(sentence, n, m)
for line in justified_lines:
    print(f'"{line}"')
