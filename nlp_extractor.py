import re
import spacy
from dateutil.parser import parse as date_parse
from datetime import datetime
from collections import Counter

# Ensure spacy model is downloaded
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

BUZZWORDS = {
    "synergy", "blockchain", "web3", "ai", "machine learning", "deep learning",
    "big data", "cloud-native", "agile", "scrum", "microservices", "disruptive",
    "innovative", "thought leadership", "ninja", "rockstar", "guru"
}

def parse_date(date_str):
    if not date_str or date_str.lower() in ['present', 'current']:
        return None  # Represents current date
    try:
        return date_parse(date_str, default=None, fuzzy=True)
    except:
        return None

def extract_months_diff(start_str, end_str):
    start = parse_date(start_str)
    end = parse_date(end_str)
    if not start:
        return 0
    if not end:
        end = datetime.now()
    
    diff = end - start
    return diff.days / 30.0

def analyze_traits(parsed_data):
    raw_text = parsed_data.get("raw_text", "")
    raw_text_lower = raw_text.lower()
    sections = parsed_data.get("sections", {})
    
    traits = {
        "Very short stints across multiple roles": False,
        "Responsibilities listed instead of outcomes": False,
        "Excessive or inconsistent bolding": False,
        "Missing dates or hidden timeline gaps": False,
        "No indication of collaboration or teamwork": False,
        "No mention of testing practices": False,
        "Overuse of “etc.”": False,
        "Excessive coursework listings": False,
        "Inclusion of irrelevant early education details": False,
        "Repetitive action verbs across bullet points": False,
        "No indication of ownership or responsibility": False,
        "No reference to real-world constraints": False,
        "Unprofessional email address": False,
        "Overuse of buzzwords in project names": False,
        "Claims without supporting links or evidence": False,
        "Repetition of the same skills across sections": False,
        "Confusing or non-linear timeline": False,
        "Only internship experience without depth": False,
        "Mentioning multiple versions of the same technology": False
    }

    # 1. Experiences (Short stints, timeline gaps, non-linear)
    exp = sections.get("EXPERIENCE", [])
    if isinstance(exp, list) and len(exp) > 0:
        short_roles = 0
        all_dates = []
        is_intern_only = True
        
        for entry in exp:
            title = entry.get('title', '').lower()
            if 'intern' not in title and 'trainee' not in title:
                is_intern_only = False

            duration = entry.get('duration', '')
            parts = [d.strip() for d in re.split(r'-|to|–|—', duration)]
            if len(parts) == 2:
                months = extract_months_diff(parts[0], parts[1])
                if 0 < months < 6:
                    short_roles += 1
                
                start_d = parse_date(parts[0])
                end_d = parse_date(parts[1])
                if start_d:
                    all_dates.append((start_d, end_d))

        if short_roles >= 2:
            traits["Very short stints across multiple roles"] = True
            
        if is_intern_only:
            traits["Only internship experience without depth"] = True

        all_dates.sort(key=lambda x: x[0] if x[0] else datetime(1900, 1, 1), reverse=True)
        if len(all_dates) >= 2:
            last_end = all_dates[0][0] # start of most recent
            for i in range(1, len(all_dates)):
                curr_end = all_dates[i][1] # end of previous role
                if last_end and curr_end:
                    gap = (last_end - curr_end).days / 30.0
                    if gap > 3.0: # 3 months gap
                        traits["Missing dates or hidden timeline gaps"] = True
                last_end = all_dates[i][0]

    # 2. Responsibilities instead of outcomes
    if re.search(r'(?i)\b(responsible for|duties included|tasks included|worked on|job was)\b', raw_text):
        traits["Responsibilities listed instead of outcomes"] = True

    # 5. No indication of collaboration
    if not re.search(r'(?i)\b(team|collaborate|collaborated|managed|led|paired)\b', raw_text):
        traits["No indication of collaboration or teamwork"] = True

    # 6. No mention of testing
    if not re.search(r'(?i)\b(test|testing|qa|junit|pytest|tdd|mock|unit test)\b', raw_text):
        traits["No mention of testing practices"] = True

    # 7. Overuse of etc.
    etc_count = len(re.findall(r'(?i)\betc\.?\b', raw_text))
    if etc_count >= 2:
        traits["Overuse of “etc.”"] = True

    # 8. Excessive coursework
    edu_text = str(sections.get("EDUCATION", ""))
    if "coursework" in edu_text.lower() or "courses" in edu_text.lower():
        if len(re.split(r'\s+', edu_text)) > 60:
            traits["Excessive coursework listings"] = True

    # 9. Irrelevant early education
    if re.search(r'(?i)\b(high school|class xii|12th|secondary|class x|10th)\b', raw_text):
        traits["Inclusion of irrelevant early education details"] = True

    # 10. Repetitive action verbs
    all_bullets = []
    if isinstance(exp, list):
        for e in exp:
            all_bullets.extend(e.get('bullets', []))
    
    first_verbs = []
    for b in all_bullets:
        doc = nlp(b)
        if len(doc) > 0 and doc[0].pos_ == 'VERB':
            first_verbs.append(doc[0].lemma_.lower())
    
    counts = Counter(first_verbs)
    if any(count >= 3 for count in counts.values()):
        traits["Repetitive action verbs across bullet points"] = True

    # 11. No ownership
    if not re.search(r'(?i)\b(led|owned|architected|delivered|spearheaded|drove|founded|built)\b', raw_text):
        traits["No indication of ownership or responsibility"] = True

    # 12. No real world constraints
    if not re.search(r'(?i)\b(latency|qps|ms|scale|scalable|million|billion|throughput|optimization|cost)\b', raw_text):
        traits["No reference to real-world constraints"] = True

    # 13. Unprofessional email
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', raw_text)
    for e in emails:
        local_part = e.split('@')[0]
        # Check for multiple numbers or excessive length
        if len(re.findall(r'\d', local_part)) >= 4 or len(local_part) > 15: 
            traits["Unprofessional email address"] = True

    # 14. Buzzwords in projects
    proj = str(sections.get("PROJECTS", "")).lower()
    proj_tokens = set(re.findall(r'\w+', proj))
    buzz_count = len(BUZZWORDS.intersection(proj_tokens))
    if buzz_count >= 2:
        traits["Overuse of buzzwords in project names"] = True

    # 15. No link
    if not re.search(r'(?i)(http|https|github\.com|linkedin\.com)', raw_text):
        traits["Claims without supporting links or evidence"] = True

    # 16. Skill repetition across sections
    skills_text = str(sections.get("SKILLS", "")).lower()
    if skills_text:
        skill_tokens = [s.strip() for s in re.split(r',|\||\n', skills_text) if s.strip()]
        # We will not implement a full robust check here for NLP simplicity
        
    # 19. Multiple tech versions
    if re.search(r'(?i)(java \d|angular \d|react \d).*?(java \d|angular \d|react \d)', raw_text):
        traits["Mentioning multiple versions of the same technology"] = True

    return traits
