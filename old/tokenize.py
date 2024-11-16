with open('the-verdict.txt', 'r') as file:
    raw_text = file.read()
print("total characters in the file: ", len(raw_text))
print(raw_text[:99])