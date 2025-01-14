import re

def classify_and_group_refined_with_context(text):
    def is_enumeration(line):
        return bool(re.match(r"^\s*[A-Za-z0-9]+\.", line.strip()))

    lines = [l for l in text.split('\n') if l.strip()]
    groups, current = [], []
    random_count = 0

    for line in lines:
        if is_enumeration(line):
            if random_count == 1 and current:
                groups.append("\n".join(current))
                current = []
            current.append(line)
            random_count = 0
        else:
            random_count += 1
            if random_count > 1 and current:
                groups.append("\n".join(current))
                current = []
                random_count = 1
            current.append(line)

    if current: groups.append("\n".join(current))
    return groups

def run_test_cases():
    t1 = """
A. Section One
1. Sub-item one
a. Sub-sub-item one
b. Sub-sub-item two
B. Section Two
Some non-enumerated text here.
C. Section Three
"""
    t2 = """
A. First level
B. Another first-level enumeration
1. First sub-item
2. Second sub-item
a. Nested sub-item
Some unrelated text follows here.
C. A new first-level enumeration
"""
    t3 = """
1. First item
2. Second item
Some unrelated text in between.
3. Third item
4. Fourth item
"""
    t4 = """
A. First section
    1. Sub-item 1
        a. Sub-sub-item 1
        b. Sub-sub-item 2
    2. Sub-item 2
Non-enumerated text related to this.
B. New section starts here
"""
    t5 = """
A. Outer enumeration
B. Second outer enumeration
Some descriptive text here.
C. Third enumeration with no sub-items
Non-related text follows.
D. Final section
"""
    cases = {"Test Case 1":t1,"Test Case 2":t2,"Test Case 3":t3,"Test Case 4":t4,"Test Case 5":t5}
    results = {}
    for name,case in cases.items():
        results[name] = classify_and_group_refined_with_context(case)
    return results

if __name__ == "__main__":
    test_results = run_test_cases()
    for test_name, result in test_results.items():
        print(f"\n{test_name} Results:")
        for i, chunk in enumerate(result, start=1):
            print(f"Chunk {i}:\n{chunk}\n")
