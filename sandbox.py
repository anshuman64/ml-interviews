def is_rotation(string1, string2):
    if len(string1) != len(string2):
        return False
    
    
    string1 = string1 + string1
    if string2 in string1:
        return True

    return False

print(is_rotation("abcabd", "bdabca"))