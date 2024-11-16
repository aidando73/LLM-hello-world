import re

text = "Hello World, this is a test"
words = re.split(r'([.,]|\s)', text)
words = [word for word in words if word not in [' ', '']]
print(words)
