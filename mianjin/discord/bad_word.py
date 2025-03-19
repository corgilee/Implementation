class TrieNode:
    """Represents a single node in the Trie structure."""
    def __init__(self):
        self.children = {}  # Dictionary to store child nodes
        self.is_end_of_word = False  # Marks if this node represents the end of a bad word

class Trie:
    """Trie data structure for storing bad words."""
    def __init__(self):
        self.root = TrieNode()  # Root node of the Trie

    def insert(self, word):
        """
        Inserts a bad word into the Trie.
        
        Args:
        word (str): The bad word to be added.
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()  # Create a new node if char is not in Trie
            node = node.children[char]  # Move to the next node
        node.is_end_of_word = True  # Mark the end of a bad word

    def search(self, text, char_mapping):
        """
        Searches the given text for any occurrence of bad words, considering obfuscations.

        Args:
        text (str): The input string to scan for bad words.
        char_mapping (dict): Dictionary mapping original characters to possible obfuscations.

        Returns:
        bool: True if any bad word (including obfuscated ones) is found, else False.
        """
        for i in range(len(text)):  # Start search from every possible position in text
            if self.dfs(text, i, self.root, char_mapping):
                return True  # If any match is found, return True immediately
        return False  # No matches found in the entire text

    def dfs(self, text, index, node, char_mapping):
        """
        Performs a depth-first search (DFS) to check if a bad word can be found in `text`,
        allowing character substitutions based on the given mapping.

        Args:
        text (str): The input text being scanned.
        index (int): Current position in the text.
        node (TrieNode): Current Trie node in search traversal.
        char_mapping (dict): Dictionary mapping characters to possible obfuscations.

        Returns:
        bool: True if a bad word is detected, otherwise False.
        """
        # If we reach a Trie node that marks the end of a bad word, return True
        if node.is_end_of_word:
            return True
        
        # If index exceeds text length, stop recursion
        if index >= len(text):
            return False

        char = text[index]  # Get the current character in the text

        # Case 1: If the character is directly in the Trie, continue matching
        if char in node.children:
            if self.dfs(text, index + 1, node.children[char], char_mapping):
                return True

        # Case 2: If this character is an obfuscated form of a valid character, try all substitutions
        for original_char, mappings in char_mapping.items():
            if char in mappings and original_char in node.children:
                if self.dfs(text, index + 1, node.children[original_char], char_mapping):
                    return True
        
        return False  # If no match is found, return False

def contains_bad_word(text, bad_words, char_mapping):
    """
    Checks whether the given text contains any bad words, considering possible obfuscations.

    Args:
    text (str): The input string to be checked.
    bad_words (List[str]): List of predefined bad words.
    char_mapping (dict): Mapping of characters to their obfuscated representations.

    Returns:
    bool: True if any bad word (including obfuscated ones) is found, else False.
    """
    # Step 1: Construct the Trie and insert all bad words
    trie = Trie()
    for word in bad_words:
        trie.insert(word)

    # Step 2: Search for bad words in the text using Trie and DFS
    return trie.search(text, char_mapping)

# Example usage
text = "applef00lbanana"
bad_words = ["fool", "silly"]
char_mapping = {'o': ['0'], 'l': ['1', 'i']}

# Expected output: True (since "f00l" is an obfuscated form of "fool")
print(contains_bad_word(text, bad_words, char_mapping))  # Output: True

'''

Trie Construction: O(M⋅K), where M is the number of bad words and K is the average length of a word.
Search Process: O(N⋅D), where N is the length of the input text and D is the branching factor due to character mappings.
Overall Complexity: The worst case is O(N⋅D), but in practice, it's much faster due to early exits in DFS.

'''