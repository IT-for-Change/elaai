from collections import Counter
import csv
import spacy
import pyphen

# logger.info(f'Initializing NLP model. Loading {config.ELA_NLP_MODEL}')
_nlp = spacy.load('en_core_web_trf')
_dic = pyphen.Pyphen(lang='en')
# logger.info('NLP model loaded')


def make_doc(text):
    return _nlp(text)


def extract_noun_phrases(doc):

    noun_phrases = []
    for chunk in doc.noun_chunks:
        words_in_chunk = chunk.text.split(' ')
        words_in_chunk_count = len(words_in_chunk)
        if (words_in_chunk_count > 1):
            if (words_in_chunk_count == 2):
                if (words_in_chunk[0] not in ['a', 'an', 'the']):
                    noun_phrases.append(chunk.text.lower())
            else:
                noun_phrases.append(chunk.text.lower())

    return noun_phrases


def nan_to_empty_string(value):
    if pd.isna(value):
        return ''
    else:
        return str(value)


def estimate_of_syllables(word):
    syllables = len(dic.inserted(word).split('-'))
    # correct the estimate for issues in pyphen library for short 2-syllable words.
    if (len(word) > 4 and syllables == 1):
        syllables = 2
    return syllables


'''==== BEGIN: NLP TREE PARSER BASED FUNCTIONS ===='''


def traverse_dependency_tree(token, depth=0):
    print("  " * depth +
          f"{token.text} ({token.dep_}|{token.pos_}|{token.tag_})")
    for child in token.children:
        traverse_dependency_tree(child, depth + 1)


def get_word_tokens(sentence):
    word_tokens = []
    for token in sentence:
        if token.pos_ not in ['PUNCT', 'SYM', 'NUM']:
            word_tokens.append(token.text)
    return word_tokens


def is_root_token(token):
    if (token.dep_ == 'ROOT'):
        if token.pos_ in ['VERB', 'AUX']:
            return True


def get_predicate_verb(sentence):
    verb_aux = None
    verb = None
    for token in sentence:
        if (token.dep_ == 'ROOT'):
            if token.pos_ == 'VERB':
                verb = token.text
            elif token.pos_ == 'AUX':
                verb_aux = token.text
            else:
                continue
    return verb, verb_aux


def get_predicate_verb_v2(sentence):
    is_verb_aux = False
    verb = None
    for token in sentence:
        if (token.dep_ == 'ROOT'):
            if token.pos_ == 'VERB':
                verb = token.text
            elif token.pos_ == 'AUX':
                verb = token.text
                is_verb_aux = True
            else:
                continue
    return verb, is_verb_aux


def get_adjectives(sentence):
    adjectives = []
    for token in sentence:
        if token.pos_ == 'ADJ':
            adjectives.append(token.text)
        else:
            continue
    return adjectives


def get_adverbs(sentence):
    adverbs = []
    for token in sentence:
        if token.pos_ == 'ADV':
            adverbs.append(token.text)
        else:
            continue
    return adverbs


def get_pronouns(sentence):
    pronouns = []
    for token in sentence:
        if token.pos_ == 'PRON':
            pronouns.append(token.text)
        else:
            continue
    return pronouns


def get_prepositions(sentence):
    prepositions = []
    for token in sentence:
        if token.pos_ == 'ADP':
            prepositions.append(token.text)
        else:
            continue
    return prepositions


def get_all_nouns(sentence):
    nouns = []
    noun_lemma_pairs = []
    for token in sentence:
        if token.pos_ == 'NOUN':
            nouns.append(token.text)
            noun_lemma_pairs.append((token.text, token.lemma_))
        else:
            continue
    return nouns, noun_lemma_pairs


def get_all_proper_nouns(sentence):
    proper_nouns = []
    for token in sentence:
        if token.pos_ == 'PROPN':
            proper_nouns.append(token.text)
        else:
            continue
    return proper_nouns


def get_all_verbs(sentence):
    verbs = []
    verbs_lemma_pairs = []
    for token in sentence:
        if token.pos_ == 'VERB':
            verbs.append(token.text)
            verbs_lemma_pairs.append((token.text, token.lemma_))
        else:
            continue
    return verbs, verbs_lemma_pairs


def get_noun_subjects(sentence):
    noun_subjects = []
    for token in sentence:
        if (token.dep_ in ["nsubj", "nsubjpass"]):
            noun_subjects.append(token.text)
    return noun_subjects


def get_noun_objects(sentence):
    noun_objects = []
    for token in sentence:
        if (token.dep_ in ["dobj", "pobj", "dative"]):
            noun_objects.append(token.text)
    return noun_objects


def get_fragment_from_sentence(sentence, in_token, token_last_matched_index):
    fragment = ''
    token_index = 0
    for i, token in enumerate(sentence):
        if token.text == in_token.text:
            if (i == token_last_matched_index):
                # this repeats in the sentence, and has been matched in a prev iteration,
                # so ignore it and continue looking for the next occurrence of the token in the sentence
                continue
            token_index = i
            break

    tokens = [token.text for token in sentence]
    if len(tokens) == 0:
        fragment = ''
    else:
        start_index = max(0, token_index - 3)
        end_index = min(len(tokens), token_index + 3)
        fragment = ' '.join(tokens[start_index:end_index])

    return fragment, token_index


def get_clause_fragments(sentence):
    clause_fragments = []
    clause_marker_found = False
    token_last_matched_index = 0
    for token in sentence:
        if (token.dep_ in ["mark"] and clause_marker_found == False):
            clause_marker_found = True
            clause_fragment, token_last_matched_index = get_fragment_from_sentence(
                sentence, token, token_last_matched_index)
        elif (token.dep_ in ["advcl", "relcl", "csubj"] and clause_marker_found == False):
            clause_marker_found = True
            clause_fragment, token_last_matched_index = get_fragment_from_sentence(
                sentence, token, token_last_matched_index)
        else:
            continue
        clause_fragments.append(clause_fragment)
    return clause_fragments


def get_coord_conjunctions(sentence):
    cconj_fragments = []
    token_last_matched_index = 0
    for token in sentence:
        if (token.dep_ in ["cc"]):
            cconj_fragment, token_last_matched_index = get_fragment_from_sentence(
                sentence, token, token_last_matched_index)
            cconj_fragments.append(cconj_fragment)
    return cconj_fragments


def print_dependency_tree(sentence):
    for token in sentence:
        if (is_root_token(token)):
            traverse_dependency_tree(token)


def print_dependency_tree(sentences):
    for sentence in sentences:
        for token in sentence:
            if (is_root_token(token)):
                traverse_dependency_tree(token)


def predicate_verbs(doc, aux_verbs=False):
    predicate_verbs = []
    predicate_verbs_aux = []
    sentences = list(doc.sents)
    for sentence in sentences:
        predicate_verb, predicate_verb_aux = get_predicate_verb(sentence)
        if (predicate_verb != None):
            predicate_verbs.append(predicate_verb)
        if (predicate_verb_aux != None and aux_verbs == True):
            predicate_verbs_aux.append(predicate_verb_aux)

    return predicate_verbs, predicate_verbs_aux


def noun_subjects(doc):
    noun_subjects_all = []
    sentences = list(doc.sents)
    for sentence in sentences:
        noun_subjects = get_noun_subjects(sentence)
        noun_subjects_all.extend(noun_subjects)
    return noun_subjects_all


def noun_objects(doc):
    noun_objects_all = []
    sentences = list(doc.sents)
    for sentence in sentences:
        noun_objects = get_noun_objects(sentence)
        noun_objects_all.extend(noun_objects)
    return noun_objects_all


def get_noun_phrases(sentence):
    # noun_chunks is only available on doc, not sentence, so convert doc to sentence
    text = sentence.text.strip()
    if not text:
        return []
    doc = _nlp(sentence.text)
    noun_phrases = extract_noun_phrases(doc)
    return noun_phrases


def clauses_as_fragments(doc):
    clause_fragments_all = []
    sentences = list(doc.sents)
    for sentence in sentences:
        clause_fragments = get_clause_fragments(sentence)
        clause_fragments_all.extend(clause_fragments)
    return clause_fragments_all


def coord_conjugations(doc):
    cconj_fragments_all = []
    sentences = list(doc.sents)
    for sentence in sentences:
        cconj_fragments = get_coord_conjunctions(sentence)
        cconj_fragments_all.extend(cconj_fragments)
    return cconj_fragments_all


'''==== END: NLP TREE PARSER BASED FUNCTIONS ===='''


def analyze(doc):
    sentences = doc.sents
    lexical_analysis = []
    syntactic_analysis = []
    morphological_analysis = []
    semantic_analysis = []

    sentence_id = 0
    for sentence in sentences:
        sentence_id += 1
        word_tokens = get_word_tokens(sentence)
        len_sentence = len(word_tokens)
        predicate_verb, is_predicate_verb_aux = get_predicate_verb_v2(sentence)
        verbs, verb_lemma_pairs = get_all_verbs(sentence)
        count_of_verbs = len(verbs)
        nouns, noun_lemma_pairs = get_all_nouns(sentence)
        count_of_nouns = len(nouns)
        proper_nouns = get_all_proper_nouns(sentence)
        count_of_proper_nouns = len(proper_nouns)
        noun_subjects = get_noun_subjects(sentence)
        count_of_noun_subjects = len(noun_subjects)
        noun_objects = get_noun_objects(sentence)
        count_of_noun_objects = len(noun_objects)
        adjectives = get_adjectives(sentence)
        count_of_adjectives = len(adjectives)
        adverbs = get_adverbs(sentence)
        count_of_adverbs = len(adverbs)
        pronouns = get_pronouns(sentence)
        count_of_pronouns = len(pronouns)
        prepositions = get_prepositions(sentence)
        count_of_prepositions = len(prepositions)
        noun_phrases = get_noun_phrases(sentence)
        count_of_noun_phrases = len(noun_phrases)
        clause_fragments = get_clause_fragments(sentence)
        count_of_clause_fragments = len(clause_fragments)

        lexical_properties = {}
        lexical_properties['idx'] = sentence_id
        lexical_properties['length'] = len_sentence
        lexical_properties['sentence'] = sentence.text
        lexical_properties['words'] = word_tokens
        lexical_properties['verbs'] = verbs
        lexical_properties['count_of_verbs'] = count_of_verbs
        lexical_properties['count_of_nouns'] = count_of_nouns
        lexical_properties['nouns'] = nouns
        lexical_properties['count_of_proper_nouns'] = count_of_proper_nouns
        lexical_properties['proper_nouns'] = proper_nouns
        lexical_properties['count_of_adjectives'] = count_of_adjectives
        lexical_properties['adjectives'] = adjectives
        lexical_properties['count_of_adverbs'] = count_of_adverbs
        lexical_properties['adverbs'] = adverbs
        lexical_properties['count_of_pronouns'] = count_of_pronouns
        lexical_properties['pronouns'] = pronouns

        syntactic_properties = {}
        syntactic_properties['predicate_verb'] = predicate_verb
        syntactic_properties['is_predicate_verb_aux'] = is_predicate_verb_aux
        syntactic_properties['count_of_noun_subjects'] = count_of_noun_subjects
        syntactic_properties['noun_subjects'] = noun_subjects
        syntactic_properties['count_of_noun_objects'] = count_of_noun_objects
        syntactic_properties['noun_objects'] = noun_objects
        syntactic_properties['count_of_prepositions'] = count_of_prepositions
        syntactic_properties['prepositions'] = prepositions
        syntactic_properties['count_of_noun_phrases'] = count_of_noun_phrases
        syntactic_properties['noun_phrases'] = noun_phrases
        syntactic_properties['count_of_clause_fragments'] = count_of_clause_fragments
        syntactic_properties['clause_fragments'] = clause_fragments

        morphological_properties = {}
        morphological_properties['verb_lemma_pairs'] = verb_lemma_pairs
        morphological_properties['noun_lemma_pairs'] = noun_lemma_pairs

        lexical_analysis.append(lexical_properties)
        syntactic_analysis.append(syntactic_properties)
        morphological_analysis.append(morphological_properties)

    return lexical_analysis, syntactic_analysis, morphological_analysis, semantic_analysis


def do_grammar_check(happy_tt, args, sentence):

    import errant

    text = sentence.text
    # Add the prefix "grammar: " before each input
    result = happy_tt.generate_text(f"grammar: {text}", args=args)
    corr_text = result.text

    annotator = errant.load('en', _nlp)
    orig = annotator.parse(text)
    cor = annotator.parse(corr_text)
    alignment = annotator.align(orig, cor)
    edits = annotator.merge(alignment)
    corrections = ''
    for e in edits:
        e = annotator.classify(e)
        if (e.type in ['R:ORTH', 'R:SPELL', 'R:OTHER']):
            continue
        # corrections += f'{e.type}|{sentence[e.o_start:e.o_end]}|hint: {corr_text}?\n'
        corrections += f'{e.type}|{sentence[e.o_start:e.o_end]}|'

    return corrections


def analyze_grammar(doc):

    from happytransformer import HappyTextToText, TTSettings

    happy_tt = HappyTextToText("T5", settings.models.ELA_GRAM_MODEL_EN_T5)
    args = TTSettings(num_beams=5, min_length=1)

    sentences = list(doc.sents)
    grammar_corrections = []
    for sentence in sentences:
        sentence_corrections = do_grammar_check(happy_tt, args, sentence)
        if (sentence_corrections == ''):
            continue
        grammar_corrections.append([sentence.text, sentence_corrections])
    return grammar_corrections
