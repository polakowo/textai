import re

def extract_year(title):
    matches = re.findall(r"\"\s\(.*?\)", title)
    if len(matches) > 0:
        match = matches[0][3:-1]
        if match.isdigit():
            return int(match)
    return None

def extract_episode(title):
    matches = re.findall(r"\(#.*?\)", title)
    if len(matches) > 0:
        return str(matches[0][2:-1]).strip()
    return None

def extract_primary_title(title):
    matches = re.findall(r"\".*?\"", title)
    if len(matches) > 0:
        return str(matches[0][1:-1]).strip()
    return None

def extract_secondary_title(title):
    matches = re.findall(r"\{.*?\(", title)
    if len(matches) > 0:
        return str(matches[0][1:-2]).strip()
    return None