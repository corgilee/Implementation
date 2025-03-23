'''
You are building an alerting system for a game moderation platform.
Given:
    A list of detected keywords (e.g., ["hack", "heroin"])
    A list of strings mapping keywords to categories (e.g., ["heroin,Drug", "hack,Cheating"])
    A list of strings mapping categories to moderation instructions (e.g., ["Drug,Please be aware of drug dealing in the game, report to police if necessary", "Cheating,Please report cheating behavior to moderators."])

Write a function that generates an email body string in the following format:

    The system detected keywords related to the following categories: [Category1, Category2, ...].
    You may take the following actions:
    [Instruction1]
    [Instruction2]

Solution steps:
1. Build a dictionary from keyword → category using the second input.
2. Build a dictionary from category → instruction using the third input.
3. Use the detected keywords to collect the set of categories (avoid duplicates).
4. For each category, retrieve the instruction.
5. Assemble and return the formatted email body string.

Time:
O(n + m + k), where
n = number of detected keywords,
m = keyword mappings,
k = category instructions
'''

def generate_email_body(detected_keywords, keyword_category_map, category_instruction_map):
    # Step 1: Build keyword to category mapping
    keyword_to_category = {}
    for item in keyword_category_map:
        keyword, category = item.split(",")
        keyword_to_category[keyword.strip()] = category.strip()

    # Step 2: Build category to instruction mapping
    category_to_instruction = {}
    for item in category_instruction_map:
        category, instruction = item.split(",", 1)  # only split at first comma
        category_to_instruction[category.strip()] = instruction.strip()

    # Step 3: Collect detected categories
    detected_categories = set()
    for keyword in detected_keywords:
        if keyword in keyword_to_category:
            detected_categories.add(keyword_to_category[keyword])

    # Step 4: Collect instructions for the detected categories
    instructions = []
    for category in detected_categories:
        if category in category_to_instruction:
            instructions.append(category_to_instruction[category])

    # Step 5: Format email body
    category_list = ", ".join(sorted(detected_categories))
    instruction_lines = "\n".join(instructions)

    email_body = (
        f"The system detected keywords related to the following categories: {category_list}.\n"
        f"You may take the following actions:\n"
        f"{instruction_lines}"
    )

    return email_body
