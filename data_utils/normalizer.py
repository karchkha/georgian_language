import re
import string


chars_to_ignore = [
    ",", "?", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�",
    "#", "!", "؟", "?", "«", "»", "،", "(", ")", "؛", "'ٔ", "٬", 'ٔ', ",", "?",
    ".", "!", "-", ";", ":", '"', "“", "%", "‘", "”", "�", "–", "…", "_", "”", '“', '„',
]
chars_to_ignore = f"""[{"".join(chars_to_ignore)}]"""

dictionary_mapping = {
    "\u200c": " ", 
    "\u200d": " ", 
    "\u200e": " ", 
    "\u200f": " ", 
    "\ufeff": " ",
    "\u0307": " ",
}


def multiple_replace(text, chars_to_mapping):
    pattern = "|".join(map(re.escape, chars_to_mapping.keys()))
    return re.sub(pattern, lambda m: chars_to_mapping[m.group()], str(text))


def remove_special_characters(text, chars_to_ignore_regex):
    text = re.sub(chars_to_ignore_regex, '', text).lower() + " "
    return text


def normalizer_at_word_level(text):
    words = text.split()
    _text = []

    for word in words:
        # Normalizer at word level
        _text.append(word)

    return " ".join(_text) + " "

def normalizer(batch, return_dict=True, filter_trivials=False, remove_extra_space=False):
    text = batch["sentence"].lower().strip()

    # Dictionary mapping
    text = multiple_replace(text, dictionary_mapping)
    text = re.sub(" +", " ", text)

    # Remove specials
    text = remove_special_characters(text, chars_to_ignore)
    text = re.sub(" +", " ", text)

    # Normalizer at word level
    text = normalizer_at_word_level(text)
    text = re.sub(" +", " ", text)
    
    if remove_extra_space:
        text = text.strip()
    else:
        text = text.strip() + " "

    if filter_trivials:
        if not len(text) > 2:
            text = None

    if not return_dict:
        return text

    batch["sentence"] = text
    return batch