
'''
step1: Split the input string into words.
step2: Calculate the total syllables for each word using the provided map.
step3: Search for a sequence of words that follows the 5-7-5 syllable pattern.
step4: Return the substring that matches the Haiku pattern or None if no such sequence is found.
'''

import string
# Example usage
syllable_map = {
    'the': 7, 'old': 1, 'pond': 5, 'a': 1, 'frog': 1,
    'jumps': 1, 'in': 1, 'sound': 1, 'of': 1, 'water': 2,
    'silent': 2, 'into': 2, 'splash': 1, 'echo': 2,'where':5
}

pattern = [5, 7, 5]

def find_first_haiku_substring(text, syllable_map):
    new_text="".join(t.lower() for t in text if t not in string.punctuation)
    text_list=new_text.split()
    

    index=0

    for i in range(len(text_list)):
        if syllable_map.get(text_list[i],0)==pattern[index]:
            index+=1
            if index==len(pattern):
                return " ".join(text_list[i-2:i+1])
        else:
            index=0

        
    return None

text = "In a WORLD, Where The OLD POND! A Frog JUMPS INTO THE Water, SILENT SPLASH ECHO."




haiku_substring = find_first_haiku_substring(text, syllable_map)
print(haiku_substring)
