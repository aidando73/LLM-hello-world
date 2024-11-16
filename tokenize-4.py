import re

with open('the-verdict.txt', 'r') as file:
    text = file.read()
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]

print(result[:100])