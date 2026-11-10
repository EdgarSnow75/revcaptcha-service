def normalize_transcript(text: str) -> list[str]:
    punctuation_list = [
        ",",
        ".",
        "-",
        "'",
        '"',
        "(",
        ")",
        "&",
        "!",
        "?",
        ";",
        "[",
        "]",
        "/",
    ]
    normalization_map = {
        "Zero": "0",
        "OH": "0",
        "O": "0",
        "ONE": "1",
        "WON": "1",
        "TWO": "2",
        "TO": "2",
        "TOO": "2",
        "THREE": "3",
        "FOUR": "4",
        "FOR": "4",
        "FIVE": "5",
        "SIX": "6",
        "SEVEN": "7",
        "EIGHT": "8",
        "ATE": "8",
        "NINE": "9",
        "BE": "B",
        "BEE": "B",
        "SEE": "C",
        "SEA": "C",
        "DEE": "D",
        "EF": "F",  # Check
        "GEE": "G",
        "AITCH": "H",  # Re-add double meaning from proposal
        "JAY": "J",
        "KAY": "K",
        "EL": "L",
        "EM": "M",
        "EN": "N",
        "PEE": "P",
        "PEA": "P",
        "QUEUE": "Q",
        "CUE": "Q",
        "ARE": "R",
        "ES": "S",
        "TEE": "T",
        "TEA": "T",
        "YOU": "U",
        "VEE": "V",
        "DOUBLE U": "W",
        "DOUBLE YOU": "W",
        "EX": "X",
        "WHY": "Y",
        "ZEE": "Z",
        "ZED": "Z",
    }

    normalized_text = text.strip().upper()
    for word in normalized_text:
        if word in punctuation_list:
            normalized_text = normalized_text.replace(word, " ")
    if " " in normalized_text:
        normalized_text = normalized_text.split()
    elif normalized_text not in normalization_map:
        normalized_text = list(normalized_text)
    else:
        return [normalization_map[normalized_text]]

    l, r = 0, 1
    while l < len(normalized_text):
        r = l + 1
        current_text = normalized_text[l]
        if normalized_text[l] in normalization_map:
            normalized_text.pop(l)
            normalized_text.insert(l, normalization_map[current_text])
            l += 1
            continue
        while r < len(normalized_text):
            if current_text in normalization_map:
                for _ in range(l, r):
                    normalized_text.pop(0)
                normalized_text.insert(l, normalization_map[current_text])
                break
            current_text += " " + normalized_text[r]
            r += 1
        l += 1

    # if normalized_text == "":
    #     return []
    return normalized_text


def change_list_to_str(text_list: list[str]) -> str:
    return "".join(text_list)


def lexical_match(transcript: str, code: str) -> bool:
    transcript = normalize_transcript(transcript)
    transcript = change_list_to_str(transcript)
    return transcript == code


def main():
    # print(normalize_transcript("hello world!"))
    # print(normalize_transcript("   hello     hi,  .  okay!"))
    # print(normalize_transcript("double u,  .  okay!"))
    # print(normalize_transcript("double.,you  okay!"))
    # print(normalize_transcript("zee.,two  pee!"))
    # print(normalize_transcript("Pee Bee SEE"))
    # print(normalize_transcript("ZEE K 1 AITCH 8"))
    # print(normalize_transcript("5G5LB"))
    print(normalize_transcript("SEVEN"))
    print(change_list_to_str(["5", "G", "5", "L", "B"]))


if __name__ == "__main__":
    main()
