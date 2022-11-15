# from nltk.tokenize import sent_tokenize
import re
import pandas as pd
import os


def clean_text(data):
    # # corpus specific cleaning
    # data = re.sub(r"#.{1,50}#", " ", data)
    # # Eg. #MERGER PROPOSED#
    # data = re.sub(r"_.{1,50}_", " ", data)
    # # Eg. _AUSTIN, TEXAS_

    data = data.replace("\\n", " ")
    data = re.sub(r"\s+", " ", data)

    data = re.sub(r"[~&@{}<>]", "", data)
    return data


def write_file(lines, filename):
    with open(filename, "w", encoding="utf8") as f:
        for line in lines:
            f.write(line + "\n")


if __name__ == "__main__":

    print("Cleaning the data...")
    files = ["dev", "test", "train"]

    # files = ["dev"]

    RES_DIR = "processed_data/"
    os.makedirs(RES_DIR, exist_ok=True)

    for file in files:
        data = pd.read_csv(f"anlp-assgn2-data/yelp-subset.{file}.csv")
        data = data["text"].apply(clean_text)
        data.to_csv(f"{RES_DIR}clean_yelp-subset.{file}.txt", index=False)

    # with open("./brown.txt", "r", encoding="utf8") as f:
    #     data = f.read()

    # data = re.sub(r"\s+", " ", data)
    # data = clean_text(data)
    # data = re.sub(r"\s+", " ", data)

    # sents = sent_tokenize(data)
    # count = len(sents)
    # # print(count)

    # offset_incr = int(count // 10)
    # offset = offset_incr * 7

    # lines = sents[:offset]
    # write_file(lines, "./train.txt")

    # lines = sents[offset : offset + (offset_incr)]
    # write_file(lines, "./valid.txt")

    # offset = offset + (offset_incr)
    # lines = sents[offset:]
    # write_file(lines, "./test.txt")
