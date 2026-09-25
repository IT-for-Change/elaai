from collections import Counter
import string
import csv
import spacy
from nlp import nlp_functions as nlpfunc
from nlp.util import compute_assist_text_comparison


def analyze_text(asr_text, text_assist, language_code, check_grammar):

    translator = str.maketrans('', '', string.punctuation)
    tokens = asr_text.translate(translator).lower().split()
    token_count = len(tokens)

    # initialize with the minimum output of the analysis.
    analyzed_text = {
        'token_count': token_count,
        'lang_code': language_code,
        'lexical_density': 0,
        'lexical_analysis': '',
        'morphological_analysis': '',
        'syntactic_analysis': '',
        'grammar_analysis': '',
        'semantic_analysis': ''
    }

    if (language_code == 'en'):
        doc = nlpfunc.make_doc(asr_text)
        lexical_analysis, syntactic_analysis, morphological_analysis, semantic_analysis = nlpfunc.analyze(
            doc)
        lexical_density = nlpfunc.calculate_lexical_density(lexical_analysis)
        grammar_analysis = {}
        if (check_grammar == '1'):
            grammar_analysis = nlpfunc.analyze_grammar(doc)

        analyzed_text['lexical_analysis'] = lexical_analysis
        analyzed_text['lexical_density'] = lexical_density
        analyzed_text['syntactic_analysis'] = syntactic_analysis
        analyzed_text['morphological_analysis'] = morphological_analysis
        analyzed_text['semantic_analysis'] = semantic_analysis
        analyzed_text['grammar_analysis'] = grammar_analysis
        
        if (text_assist):
            text_assist_similarity_score, common_words = compute_assist_text_comparison(
                    text_assist, asr_text)
            analyzed_text["text_assist_common_words"] = common_words
            analyzed_text["text_assist_similarity_score"] = text_assist_similarity_score
            #logger.info(f'Result from text-assist scoring: common words {common_words} and text assist similarity score {text_assist_similarity_score}')

    return {"analyzed_text": analyzed_text}
