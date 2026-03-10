import json
from collections import defaultdict


def report_from_text_analysis(json_str: str):
    data = json.loads(json_str)

    lexical_items = data.get("lexical_analysis", [])
    syntactic_items = data.get("syntactic_analysis", [])

    totals = {
        "total_nouns": 0,
        "total_proper_nouns": 0,
        "total_verbs": 0,
        "total_adjectives": 0,
        "total_adverbs": 0,
        "total_prepositions": 0,
        "total_noun_phrases": 0,
        "total_clause_fragments": 0,
    }

    # Sum lexical counts
    for item in lexical_items:
        totals["total_nouns"] += item.get("count_of_nouns", 0)
        totals["total_proper_nouns"] += item.get("count_of_proper_nouns", 0)
        totals["total_verbs"] += item.get("count_of_verbs", 0)
        totals["total_adjectives"] += item.get("count_of_adjectives", 0)
        totals["total_adverbs"] += item.get("count_of_adverbs", 0)

    # Sum syntactic counts
    for item in syntactic_items:
        totals["total_prepositions"] += item.get("total_prepositions", 0)
        totals["total_noun_phrases"] += item.get("count_of_noun_phrases", 0)
        totals["total_clause_fragments"] += item.get(
            "count_of_clause_fragments", 0)

    return totals


def count_word_lengths(json_str: str):
    data = json.loads(json_str)
    lexical_items = data.get("lexical_analysis", [])

    # Initialize counters
    counts = {i: 0 for i in range(2, 11)}
    counts["gt_10"] = 0

    for item in lexical_items:
        words = item.get("words", [])
        for w in words:
            word = w.strip()
            if not word:
                continue

            length = len(word)

            if 2 <= length <= 10:
                counts[length] += 1
            elif length > 10:
                counts["gt_10"] += 1

    return counts


def get_estimated_word_count(learner_duration):
    if (learner_duration <= 10):  # this will never happen as long as this is flagged as LANGID_INSUFFICIENT_SPEECH upstream
        return int(0.5 * learner_duration)  # 30 words per minute
    if (10 < learner_duration <= 20):
        return int(1 * learner_duration)
    if (20 < learner_duration <= 30):
        return int(1.5 * learner_duration)
    if (learner_duration > 30):
        return int(2 * learner_duration)


def calc_nine_point_score(transcription_language, transcription_language_reason, word_count, learner_duration):

    score = 0

    if transcription_language == 'en' and transcription_language_reason == 'LANGID_ELAAI_CONFIRMED_EN':
        if (0 < word_count <= 10):
            score = 7
        if (10 < word_count <= 25):
            score = 8
        if (word_count > 25):
            score = 9
        return score

    if transcription_language == 'en' and transcription_language_reason == 'LANGID_ELAAI_MIXED_EN':
        if (0 < word_count <= 10):
            score = 4
        if (11 < word_count <= 25):
            score = 5
        if (word_count > 25):
            score = 6
        return score

    if transcription_language != 'en' and transcription_language_reason == 'LANGID_ELAAI_MIXED_OTHER':
        word_count = get_estimated_word_count(learner_duration)
        if (0 < word_count <= 10):
            score = 4
        if (11 < word_count <= 25):
            score = 5
        if (word_count > 25):
            score = 6
        return score

    if transcription_language != 'en' and transcription_language_reason == 'LANGID_ELAAI_CONFIRMED_OTHER':
        word_count = get_estimated_word_count(learner_duration)
        if (0 < word_count <= 10):
            score = 1
        if (11 < word_count <= 25):
            score = 2
        if (word_count > 25):
            score = 3
        return score

    if transcription_language_reason == 'LANGID_NO_SPEECH':
        score = 0
        return score

    if transcription_language_reason == 'LANGID_INSUFFICIENT_SPEECH':
        score = 1
        return score

    return score


def calc_conversation_contribution(learner_duration, teacher_duration):
    total_duration = learner_duration + teacher_duration
    if (total_duration == 0):
        return 0
    if (teacher_duration == 0 and learner_duration > 0):
        return 100

    conversation_contribution_pct = round(
        (learner_duration / total_duration) * 100)

    return conversation_contribution_pct


def do_report(report_inputs):
    # print(report_inputs)
    transcription_language = report_inputs['transcription_language']
    transcription_language_reason = report_inputs['transcription_language_reason']
    word_count = report_inputs['word_count']
    lexical_density = report_inputs['lexical_density']
    learner_duration = report_inputs['learner_duration']
    teacher_duration = report_inputs['teacher_duration']
    nine_point_score = calc_nine_point_score(
        transcription_language, transcription_language_reason, word_count, learner_duration)
    conversation_contribution_pct = calc_conversation_contribution(
        learner_duration, teacher_duration)
    totals = report_from_text_analysis(report_inputs['text_analysis'])
    counts = count_word_lengths(report_inputs['text_analysis'])

    report_outputs = {
        'nine_point_score': nine_point_score,
        'lexical_density': lexical_density,
        'word_count': word_count,
        'conversation_contribution_pct': conversation_contribution_pct,
        'total_nouns': totals["total_nouns"],
        'total_proper_nouns': totals["total_proper_nouns"],
        'total_verbs': totals["total_verbs"],
        'total_adverbs': totals["total_adverbs"],
        'total_adjectives': totals["total_adjectives"],
        'total_prepositions': totals["total_prepositions"],
        'total_noun_phrases': totals["total_noun_phrases"],
        'total_clause_fragments': totals["total_clause_fragments"],
        'two_letter_words': counts[2],
        'three_letter_words': counts[3],
        'four_letter_words': counts[4],
        'five_letter_words': counts[5],
        'six_letter_words': counts[6],
        'seven_letter_words': counts[7],
        'eight_letter_words': counts[8],
        'nine_letter_words': counts[9],
        'ten_letter_words': counts[10],
        'greater_than_10_letter_words': counts["gt_10"]
    }

    return {"report_outputs": report_outputs}
