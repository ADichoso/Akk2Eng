import re

trans_subscripts = str.maketrans('₀₁₂₃₄₅₆₇₈₉ₓ', '0123456789x')
trans_h   = str.maketrans('ḫḪ', 'hH') # From dataset description: Only one type of H in akkadian, so this can be one-to-one
trans_specials = str.maketrans('', '', '!?"——<>⌈⌋⌊[]+ʾ/;„') # Special Characters
trans_specials_train = str.maketrans('', '', '<>⌈⌋⌊[]/;"') # Special Characters

DECIMAL_TO_FRAC = {
    r'0\.3{1,5}': "1/3",
    r'0\.6{4,5}': "2/3",
    r'0\.16{3,4}': "1/6",
    r'0\.83{3,4}': "5/6",
    r'0\.5': "1/2",
    r'0\.25': "1/4",
    r'0\.75': "3/4",
    r'0\.125': "1/8",
    r'0\.375': "3/8",
    r'0\.625': "5/8",
    r'0\.875': "7/8",
}

UNICODE_FRACTIONS = {
    "½": "1/2",
    "⅓": "1/3",
    "⅔": "2/3",
    "¼": "1/4",
    "¾": "3/4",
    "⅕": "1/5",
    "⅖": "2/5",
    "⅗": "3/5",
    "⅘": "4/5",
    "⅙": "1/6",
    "⅚": "5/6",
    "⅛": "1/8",
    "⅜": "3/8",
    "⅝": "5/8",
    "⅞": "7/8",
}

def convert_fractions(s):
    #Look for decimals to turn to fractions

    #Separate mixed numbers first (whole.dec => whole 0.dec)
    s = s.str.replace(re.compile(r'([0-9]|\s)\.([0-9]+)'), r'\1 0.\2', regex=True)

    #Look for decimals and convert into fractions (0.dec => num/den)
    for dec in DECIMAL_TO_FRAC.keys():
        s = s.str.replace(dec, DECIMAL_TO_FRAC[dec], regex=True)
    
    for unicode in UNICODE_FRACTIONS.keys():
        s = s.str.replace(unicode, UNICODE_FRACTIONS[unicode])
    
    #Remove lone zeroes just in case
    s = s.str.replace(" 0 ", " ")
    return s


def process_transliterations(df):
    s = df["transliteration"]

    #Replace colons -> spaces
    s = s.str.replace(r'\:', " ", regex=True) 
    
    #Remove unecessary periods (All except in numbers).
    s = s.str.replace(r'\s*\.\s', " ", regex=True) 
    s = s.str.replace(r' \.\s*', " ", regex=True)

    #Turn ellipsis (...) into <big_gap>
    s = s.str.replace(r'(\.{3,}|…+)', "<big_gap>", regex=True)

    # x -> <gap>
    s = s.str.replace('x', "<gap>", regex=False)

    #Subscripts and H cleaning
    s = s.str.translate(trans_subscripts)
    s = s.str.translate(trans_h)

    #Replace annotations for broken lines with gaps
    s = s.str.replace(r'\(broken line\)', '<gap>', regex=True)
    s = s.str.replace(r'\(large break|5 broken lines|4 broken lines|3 broken lines|2 broken lines|broken line\)', '<big_gap>', regex=True)

    #Remove special characters while still keeping the gaps (pad gaps with special characters so that they stay safe)
    #Protect special words inside parentheses so that we can remove the other parentheses

    s = s.str.replace('<gap>', '\x00gap\x00')
    s = s.str.replace('<big_gap>', '\x00big\x00')        
    s = s.str.translate(trans_specials)
    s = s.str.replace('\x00gap\x00', '<gap>')
    s = s.str.replace('\x00big\x00', '<big_gap>')

    #Convert all fractions
    s = convert_fractions(s)

    #Combine multiple gaps together to form a big gap
    s = s.str.replace(r'(<gap>(\s{0,}<gap>+)+)', "<big_gap>", regex=True)
    s = s.str.replace(r'(<big_gap>(\s{0,}<big_gap>+)+)', "<big_gap>", regex=True)
    s = s.str.replace(r'((\w[-]{0,1})<big_gap>(\s{0,}<big_gap>+)+)', r'\2<big_gap>', regex=True)
    s = s.str.replace(r"<gap>\s*<big_gap>", "<big_gap>", regex=True)
    s = s.str.replace(r"<big_gap>\s*<gap>", "<big_gap>", regex=True)
    
    s = s.str.replace(re.compile(r'\b(\w+)(?:\s+\1\b)+'), r'\1', regex=True) #Remove duplicate words next to each other
    s = s.str.replace(re.compile(r'\s+([.,:])'), r'\1', regex=True) #Remove long whitespaces before punctuations (             .)
    s = s.str.replace(re.compile(r'([.,])\1+'), r'\1', regex=True) #Remove duplicate punctations (,,,,,,,)

    #Remove duplicate whitespaces
    s = s.str.replace(r'\s+', ' ', regex=True) 

    #Remove unneeded parentheses while keeping those that have words in them
    s = s.str.replace(re.compile(r'\((mushen|TÚG|lu2|kush|gesh|dub|uru|na4|id2|kur|mi|HI|ki|e2|u2|m|d)\)'), r"[\1]", regex=True) 
    s = s.str.replace(r'\(|\)', "", regex=True) 
    s = s.str.replace(re.compile(r'\[(mushen|TÚG|lu2|kush|gesh|dub|uru|na4|id2|kur|mi|HI|ki|e2|u2|m|d)\]'), r"(\1)", regex=True) 
    
    #Last cleaning of white spaces
    s = s.str.strip()

    return s

def process_translations(df):
    s = df["translation"]

    #Turn ellipsis (...) into <big_gap>
    s = s.str.replace(r'(\.{3,}|…+)', "<big_gap>", regex=True)
    s = s.str.replace(r'\s*x ', "<gap>", regex=True)

    #Subscripts and H cleaning
    s = s.str.translate(trans_subscripts)
    s = s.str.translate(trans_h)

    #Remove special characters while still keeping the gaps
    s = s.str.replace('<gap>', '\x00gap\x00')
    s = s.str.replace('<big_gap>', '\x00big\x00')

    s = s.str.replace(r'\(((\w+\s*)+)(\?|\!)\)', r'(\1)', regex=True) #Remove ? and ! inside parentheses with actual notes (note?)
    s = s.str.replace(r'\[((\w+\s*)+)(\?|\!)\]', r'[\1]', regex=True) #Remove ? and ! inside brackets with actual notes [note?]

    #Remove annotations inside parentheses
    regex_annotations = re.compile(r'\((fem|plur|pl|sing|singular|plural|\?|\!)\..*?\)', re.I)
    s = s.str.replace(regex_annotations, '', regex=True)
    s = s.str.replace(r'\((\?|\!)\)', '', regex=True) #Remove (?) and (!)
    s = s.str.replace(r'\s*(—|—|-) ', ' ', regex=True)
    s = s.str.replace(r' (—|—|-)(\w)', r' \2', regex=True)
    s = s.str.translate(trans_specials_train)
    s = s.str.replace('\x00gap\x00', '<gap>')
    s = s.str.replace('\x00big\x00', '<big_gap>')

    #Convert all fractions
    s = convert_fractions(s)
    
    #Combine multiple gaps together to form a big gap
    s = s.str.replace(r'(<gap>(\s{0,}<gap>+)+)', "<big_gap>", regex=True)
    s = s.str.replace(r'(<big_gap>(\s{0,}<big_gap>+)+)', "<big_gap>", regex=True)
    s = s.str.replace(r'((\w[-]{0,1})<big_gap>(\s{0,}<big_gap>+)+)', r'\2<big_gap>', regex=True)
    s = s.str.replace(r"<gap>\s*<big_gap>", "<big_gap>", regex=True)
    s = s.str.replace(r"<big_gap>\s*<gap>", "<big_gap>", regex=True)

    #Seperate words that are attached to gaps
    s = s.str.replace(r'(.)(<gap>|<big_gap>)(.)', r'\1 \2 \3', regex=True)

    s = s.str.replace(re.compile(r'\b(\w+)(?:\s+\1\b)+'), r'\1', regex=True) #Remove duplicate words next to each other
    s = s.str.replace(re.compile(r'\s+([.,:])'), r'\1', regex=True) #Remove long whitespaces before punctuations (             .)
    s = s.str.replace(re.compile(r'([.,])\1+'), r'\1', regex=True) #Remove duplicate punctations (,,,,,,,)

    #Remove duplicate whitespaces
    s = s.str.replace(r'\s+', ' ', regex=True) 

    #Remove parentheses
    s = s.str.replace(r'\(|\)', "", regex=True) 

    #Last cleaning of white spaces
    s = s.str.strip()
    
    return s
