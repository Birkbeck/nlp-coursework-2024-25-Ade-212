#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
from pathlib import Path
from spacy.tokens import Doc
from collections import Counter
import pandas as pd
import string
import re 
import pickle
import math

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    # tokenize sentences & words
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    # filter out non-words (e.g. pure punctuation)
    words = [w for w in words if any(c.isalpha() for c in w)]
    num_sents = max(len(sentences), 1)
    num_words = len(words) or 1
    # total syllables
    total_syl = sum(count_syl(w, d) for w in words)
    # Flesch-Kincaid formula
    fk = 0.39 * (num_words / num_sents) + 11.8 * (total_syl / num_words) - 15.59
    return fk


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    w = word.lower()
    # 1) Lookup in CMU dict
    if w in d:
        # take first pronunciation
        pron = d[w][0]
        # count phonemes ending in a digit (0,1,2)
        return sum(1 for p in pron if p[-1].isdigit())
    # 2) Fallback: count vowel clusters
    clusters = re.findall(r"[aeiouy]+", w)
    return max(1, len(clusters))


def read_novels(path=Path.cwd() / "p1-texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year
    
    Args:
        path (Path): path to the directory containing novel .txt files
    Returns:
        pandas.DataFrame: columns = ["text", "title", "author", "year"]
    """
    records = []
    for txt_file in sorted(path.glob("*.txt")):
        # filename without .txt
        stem = txt_file.stem
        parts = stem.split("-")
        # assume last part is year, second last is author, rest is title
        year = int(parts[-1])
        author = parts[-2]
        title = "-".join(parts[:-2])
        # read full text
        text = txt_file.read_text(encoding="utf-8")
        records.append({
            "text": text,
            "title": title,
            "author": author,
            "year": year
        })

    df = pd.DataFrame.from_records(records, columns=["text", "title", "author", "year"])
    df = df.sort_values("year").reset_index(drop=True)
    return df


def _chunk_text(text, size=50_000):
    """Yield overlapping character slices so as to not split words mid-chunk."""
    start = 0
    N = len(text)
    while start < N:
        end = min(start + size, N)
        # expand to next whitespace to avoid chopping word
        while end < N and not text[end].isspace():
            end += 1
        yield text[start:end]
        start = end

def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    store_path.mkdir(exist_ok=True)
    
    parsed_docs = []
    for text in df["text"]:
        if len(text) > nlp.max_length:          # oversize novel
            slabs = list(_chunk_text(text, size=50_000))
            slab_docs = list(nlp.pipe(slabs, disable=["ner", "textcat"]))
            full_doc = Doc.from_docs(*slab_docs)
            parsed_docs.append(full_doc)
        else:                                   # small enough
            parsed_docs.append(nlp(text))
    
    df = df.copy()
    df["parsed"] = parsed_docs

    with open(store_path / out_name, "wb") as f:
        pickle.dump(df, f)

    return df


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    # 1. Tokenize the text
    tokens = nltk.word_tokenize(text)
    # 2. Remove tokens that are only punctuation
    tokens = [t for t in tokens
              if not all(char in string.punctuation for char in t)]
    # 3. Lowercase for type counts
    tokens = [t.lower() for t in tokens]
    # 4. Handle empty text
    if not tokens:
        return 0.0
    # 5. Type-token ratio
    return len(set(tokens)) / len(tokens)


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def object_counts(doc: Doc):
    """Returns the 10 most common syntactic objects in the parsed Doc."""
    counts = Counter()
    for token in doc:
        # find syntactic objects
        if token.dep_ == "dobj" and token.head.pos_ == "VERB":
            counts[token.text.lower()] += 1
    return [obj for obj, _ in counts.most_common(10)]


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    tverb = target_verb.lower()
    subj_counts = Counter()
    co_counts = Counter()
    total_rels = 0

    # 1) Gather counts
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            subj = token.text.lower()
            subj_counts[subj] += 1
            total_rels += 1
            if token.head.lemma_.lower() == tverb:
                co_counts[subj] += 1

    verb_rel_count = sum(co_counts.values()) or 1  # avoid zero

    # 2) Coalculate PMI for each subject that co-occurs
    pmi_scores = {}
    for subj, co in co_counts.items():
        p_w = subj_counts[subj] / total_rels
        p_v = verb_rel_count / total_rels
        p_wv = co / total_rels
        pmi = math.log(p_wv / (p_w * p_v), 2)
        pmi_scores[subj] = pmi

    # 3) Return top 10 by PMI
    return [subj for subj, _ in
            sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)[:10]]


def subjects_by_verb_count(doc: Doc, verb: str):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    verb = verb.lower()
    counts = Counter()
    for token in doc:
        # match any inflection of the target verb
        if token.pos_ == "VERB" and token.lemma_.lower() == verb:
            for child in token.children:
                if child.dep_ == "nsubj":
                    counts[child.text.lower()] += 1
    return [subj for subj, _ in counts.most_common(10)]



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    counts = Counter()
    for token in doc:
        if token.pos_ == "ADJ":
            counts[token.text.lower()] += 1
    return counts.most_common(10)



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    #nltk.download("cmudict")
    df = parse(df)
    print(df.head())
    print(get_ttrs(df))
    print(get_fks(df))
    df = pd.read_pickle(Path.cwd() / "pickles"/ "parsed.pickle")
    print(df.head())
    #print(adjective_counts(df))
 
    for _, row in df.iterrows():
        title = row["title"]
        doc   = row["parsed"]
        
        # 10 most common syntactic objects
        print(f"{title} — syntactic objects")
        print(object_counts(doc))
        
        # 10 most common 'hear' subjects by frequency
        print(f"{title} — hear-subjects (count)")
        print(subjects_by_verb_count(doc, "hear"))
        
        # 10 most common 'hear' subjects by PMI
        print(f"{title} — hear-subjects (PMI)")
        print(subjects_by_verb_pmi(doc, "hear"))
        
        # Adjective count
        print(f"{title} — adjectives")
        print(adjective_counts(doc))
        
        print()  # blank line between novels for readability

    


