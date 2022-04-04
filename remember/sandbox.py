outcomes = []
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                for m in range(3):
                    outcome = str(i) + str(j) + str(k) + str(l) + str(m)
                    outcomes.append(outcome)


def WordleGuesser(D, H):
    # Create copy of dictionary
    valid_D = D

    # Loop through each guess in History
    for (guess, outcome) in H:
        # Loop through each word in Dictionary
        for word in valid_D:
            # Loop through each character of the guess
            for i, (g, o) in enumerate(zip(guess, outcome)):
                # Determine if word is valid
                if not CharValidator(g, i, o, word):
                    del valid_D[word]

    # Initialize best guess and lowest expected words
    current_best_guess = None
    current_lowest_expected_words = len(D)

    # Loop through each word in D
    for guess in D:
        # If the word has been guessed before, skip it
        if guess in H:
            continue

        # Initialized the total expected words
        total_expected_words = 0

        # Loop through each outcome
        for outcome in outcomes:
            # Make a copy of valid Dictionary
            copy_D = valid_D

            # Loop through each word in copy Dictionary
            for word in copy_D:
                # Determine if word is valid
                for i, (g, o) in enumerate(zip(guess, outcome)):
                    if not ValidateWord(g, i, o, word):
                        # If not, remove from copy Dictionary
                        del copy_D[word]

            # Add total expected words for this outcome
            total_expected_words += len(copy_D) * len(copy_D) / len(valid_D)

        # Replace best guess if new lowest expected words
        if total_expected_words < current_lowest_expected_words:
            current_best_guess = guess
            current_lowest_expected_words = total_expected_words

    return guess


def ValidateWord(char, idx, outcome, word):
    if outcome == "0" and char not in word:
        return True
    elif outcome == "1" and char in word and word[idx] != char:
        return True
    elif outcome == "2" and word[idx] == char:
        return True

    return False


def GuessChecker(guess, wordle):
    outcome = ""

    for i, char in enumerate(guess):
        if wordle[i] == char:
            outcome += str(2)
        elif char in wordle:
            outcome += str(1)
        else:
            outcome += str(0)

    return (guess, outcome)


# class SortedStack:
#     def __init__(self):
#         self.stack = []
#         self.temp_stack = []

#     def __repr__(self):
#         return str(self.stack)

#     def push(self, value):
#         while len(self.stack) > 0:
#             if value <= self.stack[len(self.stack)-1]:
#                 break

#             self.temp_stack.append(self.stack.pop())

#         self.temp_stack.append(value)

#         for _ in range(len(self.temp_stack)):
#             self.stack.append(self.temp_stack.pop())

#     def pop(self):
#         if len(self.stack) == 0:
#             print("No elements in stack!")
#             return

#         return self.stack.pop()

#     def peek(self):
#         if len(self.stack) == 0:
#             print("No elements in stack!")
#             return

#         return self.stack[len(self.stack) - 1]

#     def isEmpty(self):
#         return len(self.stack) == 0


# s = SortedStack()
# s.push(4)
# print(s)
# s.push(2)
# print(s)
# print(s.pop())
# print(s)
# s.push(3)
# print(s)
# s.push(1)
# print(s)
# print(s.pop())
# print(s)
