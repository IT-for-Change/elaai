from lemminflect import getLemma
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


# This is required to collapse the difference between verbal nouns and verbs
# for the purpose of getting a word list for calculation
# If the assist text used "cooking" as a verb form, but the learner speech used "cooking" in such a way it is a noun form,
# they should be treated alike, as conveying the lexical idea of "cook"
def canonical_lemma(token):
    # First try the verb lemma
    verb = getLemma(token.text, upos="VERB")
    if verb:
        return verb[0].lower()

    # Fall back to the noun lemma
    noun = getLemma(token.text, upos="NOUN")
    if noun:
        return noun[0].lower()

    # Finally, use spaCy's lemma
    return token.lemma_.lower()


def preprocess(text):
    doc = nlp(text)

    tokens = []
    for token in doc:
        lemma = canonical_lemma(token)
        if (
            lemma not in ELA_STOPWORDS and      # remove stopwords
            not token.is_punct and     # remove punctuation
            not token.is_space         # remove spaces
        ):
            tokens.append(lemma)

    return tokens


def check_similarity(text_assist, asr_text):
    words_text_assist = preprocess(text_assist)
    words_asr_text = preprocess(asr_text)

    text_assist_wordset = set(words_text_assist)
    asr_wordset = set(words_asr_text)
    common_wordset = text_assist_wordset & asr_wordset  # intersection
    text_assist_similarity_score = len(
        common_wordset)*100 / len(asr_wordset)
    return text_assist_similarity_score, common_wordset


def compute_assist_text_comparison(text_assist, asr_text):
    text_assist_similarity_score, common_words = check_similarity(
        text_assist, asr_text)
    return round(text_assist_similarity_score, 0), list(common_words)
