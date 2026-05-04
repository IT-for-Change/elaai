import spacy

# Load English model
nlp = spacy.load('en_core_web_trf')

ELA_STOPWORDS = {
    # articles
    "a", "an", "the",

    # pronouns
    "i", "me", "my", "mine",
    "you", "your", "yours",
    "he", "him", "his",
    "she", "her", "hers",
    "it", "its",
    "we", "us", "our", "ours",
    "they", "them", "their", "theirs",

    # auxiliary verbs
    "am", "is", "are", "was", "were",
    "be", "been", "being",
    "have", "has", "had",
    "do", "does", "did",

    # conjunctions
    "and", "or", "but", "so",

    # prepositions (minimal)
    "in", "on", "at", "of", "to", "for", "by", "with",

    # determiners
    "this", "that", "these", "those",

    # common question words
    "what", "when", "why", "where", "which", "how", "whose"

    # common Whisper hallucinations
    "thank", "thank you", "subscribe", "channel", "click"
}


def preprocess(text):
    doc = nlp(text)

    tokens = []
    for token in doc:
        lemma = token.lemma_.lower()
        if (
            lemma not in ELA_STOPWORDS and      # remove stopwords
            not token.is_punct and     # remove punctuation
            not token.is_space         # remove spaces
        ):
            tokens.append(lemma)

    return tokens


def jaccard_similarity(text1, text2):
    words1 = preprocess(text1)
    words2 = preprocess(text2)

    set1 = set(words1)
    set2 = set(words2)
    intersection = set1 & set2
    union = set1 | set2

    similarity = len(intersection) / len(union) if union else 0

    return round(similarity, 2), intersection


def compute_assist_text_comparison(reference_text, asr_text):
    similarity_score, common_words = jaccard_similarity(
        reference_text, asr_text)
    return similarity_score, common_words


'''
# Example
ref = "This is a long paragraph containing several words and concepts."
# short = "This paragraph has some words."
short = "This paragraph has some words containing words and concepts."
'''
