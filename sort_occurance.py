
input_string = "sajid"
characters = set(input_string)

occurrences = {char: input_string.count(char) for char in characters}

print(occurrences)