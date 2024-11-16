import re

with open('the-verdict.txt', 'r') as file:
    text = file.read()
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]

print(result[:100])

# Sort all tokens in the list
all_words = sorted(set(result))
print(len(all_words))

# Token ids
vocab = {word: i for i, word in enumerate(all_words)}
