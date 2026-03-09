"""Bilingual science exam deduplication helper.

Post-process bilingual science exam questions (IKSC format).
Removes English duplicates and cleans bilingual options.
"""

import re
from typing import List


def _dedup_bilingual_science(questions: List[dict]) -> List[dict]:
    """Post-process bilingual science exam questions (IKSC format).

    These documents have each question in both English and Vietnamese.
    This function removes English duplicates and cleans bilingual options.
    """
    def _has_vietnamese(text):
        return any(ord(c) > 127 for c in text)

    def _is_section_header(text):
        t = text.strip()
        return bool(re.match(r'^\d+\s*[–\-]\s*Point\s+Questions?$', t, re.IGNORECASE))

    # Step 1: Remove section headers and merge EN+VN pairs into bilingual questions
    cleaned = []
    i = 0
    while i < len(questions):
        q = questions[i]
        qtext = q.get('question', '').strip()

        # Skip section headers
        if _is_section_header(qtext):
            i += 1
            continue

        # If pure English question, check if next question is Vietnamese version
        if not _has_vietnamese(qtext):
            if i + 1 < len(questions):
                next_q = questions[i + 1]
                next_text = next_q.get('question', '').strip()
                if _has_vietnamese(next_text):
                    # Merge EN + VN into bilingual question
                    merged_q = dict(next_q)
                    merged_q['question'] = qtext + '\n' + next_text
                    # Use whichever has more options; merge bilingual opts if both have same count
                    en_opts = q.get('options', [])
                    vn_opts = next_q.get('options', [])
                    if len(en_opts) >= len(vn_opts) and len(en_opts) >= 3:
                        # EN has options — merge with VN if same count
                        if len(en_opts) == len(vn_opts):
                            merged_opts = []
                            for en_o, vn_o in zip(en_opts, vn_opts):
                                if _has_vietnamese(vn_o) and not _has_vietnamese(en_o):
                                    merged_opts.append(en_o + ' / ' + vn_o)
                                else:
                                    merged_opts.append(en_o)
                            merged_q['options'] = merged_opts
                        else:
                            merged_q['options'] = en_opts
                    # else: keep VN options from next_q
                    cleaned.append(merged_q)
                    i += 2
                    continue
            # Standalone English with few options — skip
            if len(q.get('options', [])) < 5:
                i += 1
                continue

        cleaned.append(q)
        i += 1

    # Step 2: Remove near-duplicate questions (same content from different sources)
    # When texts match after cleaning, merge: keep clean text + best options
    def _has_garbled_prefix(text):
        """Check if text starts with garbled diagram labels like K1K2K3."""
        m = re.match(r'^[A-Z0-9]{4,}', text)
        return bool(m)

    def _clean_garbled(text):
        return re.sub(r'(?:[A-Z]\d){2,}', '', text)

    deduped = []
    skip_indices = set()
    for idx, q in enumerate(cleaned):
        if idx in skip_indices:
            continue
        qtext = q.get('question', '').strip()
        opts = q.get('options', [])
        opts_count = len(opts)
        qtext_clean = _clean_garbled(qtext)
        merged = False
        for idx2 in range(idx + 1, min(idx + 5, len(cleaned))):
            if idx2 in skip_indices:
                continue
            q2 = cleaned[idx2]
            q2text = q2.get('question', '').strip()
            q2_opts = q2.get('options', [])
            q2text_clean = _clean_garbled(q2text)
            # Check if one text contains the other (near-duplicate)
            if len(qtext_clean) > 10 and len(q2text_clean) > 10:
                if (qtext_clean in q2text_clean or q2text_clean in qtext_clean
                        or qtext in q2text or q2text in qtext):
                    # Merge: prefer clean text, take more options
                    has_garbled1 = qtext_clean != qtext
                    has_garbled2 = q2text_clean != q2text
                    best_text = q2text if has_garbled1 and not has_garbled2 else qtext if not has_garbled1 else qtext_clean
                    best_opts = opts if opts_count >= len(q2_opts) else q2_opts
                    merged_q = dict(q)
                    merged_q['question'] = best_text
                    merged_q['options'] = best_opts
                    deduped.append(merged_q)
                    skip_indices.add(idx2)
                    merged = True
                    break
        if not merged:
            deduped.append(q)
    cleaned = deduped

    # Step 3: Keep bilingual options as-is (EN / VN format)
    # No stripping — options like "Refraction / Khúc xạ" stay bilingual

    return cleaned
