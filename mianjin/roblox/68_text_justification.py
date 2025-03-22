'''
Steps:
Use a greedy loop to collect words that fit in the current line.
Once a line is full, format it:
If it’s the last line or has only one word → left justify.
Otherwise → fully justify with space padding.
Repeat until all words are processed.

Time: O(n) where n = total number of characters in words
Space: O(n) for output list

'''

def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
    result = []  # Final list of justified lines
    i = 0  # Pointer to track position in words list

    while i < len(words):
        # Step 1: Determine how many words fit into the current line
        line_len = len(words[i])  # Start with first word's length
        j = i + 1  # Pointer to the next word

        while j < len(words) and line_len + 1 + len(words[j]) <= maxWidth:
            # +1 for at least one space between words
            line_len += 1 + len(words[j])
            j += 1

        # Step 2: We now know words[i:j] fit in this line
        line_words = words[i:j]
        num_words = j - i
        total_chars = sum(len(word) for word in line_words)

        # Step 3: Format the line
        if j == len(words) or num_words == 1:
            # Last line OR line has only one word → left justify
            line = " ".join(line_words)
            line += " " * (maxWidth - len(line))  # Pad with spaces at end
        else:
            # Full justification
            total_spaces = maxWidth - total_chars
            spaces_between_words = total_spaces // (num_words - 1)
            extra_spaces = total_spaces % (num_words - 1)

            line = ""
            for k in range(num_words - 1):
                line += line_words[k]
                # Distribute extra space to the leftmost gaps
                space_count = spaces_between_words + (1 if k < extra_spaces else 0)
                line += " " * space_count
            line += line_words[-1]  # Add the last word (no space after it)

        result.append(line)
        i = j  # Move to the next group of words

    return result