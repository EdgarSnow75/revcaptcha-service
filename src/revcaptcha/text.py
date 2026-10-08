def normalize_transcript(text: str) -> list:
    punctuation_list = [",", ".", "-", "'", "\"", "(", ")", "&", "!", "?", ";", "[", "]", "/"]
    normalized_text = text.strip().upper()
    for word in normalized_text:
        if word in punctuation_list:
            normalized_text = normalized_text.replace(word, "")
    normalized_text = normalized_text.split()
    if normalized_text == "":
        return []
    return normalized_text

def main():
    print(normalize_transcript("hello world!"))
    print(normalize_transcript("   hello     hi,  .  okay!"))


if __name__ == "__main__":
    main()