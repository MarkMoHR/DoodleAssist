def append_prompt(text, append_words):
    assert type(append_words) is list
    tokens = text.split(',')
    tokens = [item.strip() for item in tokens]
    tokens = [item for item in tokens if item not in append_words] + append_words
    appended_text = ', '.join(tokens)
    return appended_text
