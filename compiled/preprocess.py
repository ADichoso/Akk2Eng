import pandas as pd
import utils.bpe as bpe

cleaned_complete_df = pd.read_csv("processed/cleaned_train_complete.csv")
cleaned_incomplete_df = pd.read_csv("processed/cleaned_train_incomplete.csv")

#Parentheses in the cleaned data were not removed for determinatives, so look inside these for them.
def parse(text):
    stack = []
    for char in text:
        if char == '(':
            #stack push
            stack.append([])
        elif char == ')':
            yield ''.join(stack.pop())
        elif len(stack) > 0:
            #stack peek
            stack[-1].append(char)
    
    return stack

parses = []
for i, row in pd.concat([cleaned_complete_df, cleaned_incomplete_df]).iterrows():
    parses.append(tuple(parse(row['transliteration'])))

#Loop through parse tuples
determinatives = dict()
for p in parses:
    for d in p:
        if determinatives.get(d):
            determinatives[d] += 1
        else:
            determinatives[d] = 1

DET_TOKENS = {
    "(d)": "<god>",
    "(HI)": "<star>",
    "(ki)": "<place>",
    "(lu2)": "<person>",
    "(e2)": "<building>",
    "(uru)": "<city>",
    "(kur)": "<land>",
    "(mi)": "<female>",
    "(m)": "<male>",
    "(gesh)": "<wood>",
    "(TÚG)": "<textile>",
    "(dub)": "<tablet>",
    "(id2)": "<river>",
    "(mushen)": "<bird>",
    "(na4)": "<stone>",
    "(kush)": "<hide>",
    "(u2)": "<plant>",
}

for key in DET_TOKENS.keys():
    cleaned_complete_df['transliteration'] = cleaned_complete_df['transliteration'].str.replace(key, DET_TOKENS[key])
    cleaned_incomplete_df['transliteration'] = cleaned_incomplete_df['transliteration'].str.replace(key, DET_TOKENS[key])

#OPTIONAL
#cleaned_complete_df['transliteration'] = cleaned_complete_df['transliteration'].str.replace(re.compile(r'(\w+)*<(\w+)>(\w+)*'), r'\1 <\2> \3', regex=True)
#cleaned_incomplete_df['transliteration'] = cleaned_incomplete_df['transliteration'].str.replace(re.compile(r'(\w+)*<(\w+)>(\w+)*'), r'\1 <\2> \3', regex=True)


BPE = bpe.BytePairEncoder()
BPE_akk = bpe.BytePairEncoder()
cleaned_complete_df["transliteration"] = cleaned_complete_df["transliteration"].astype("object")
cleaned_incomplete_df["transliteration"] = cleaned_incomplete_df["transliteration"].astype("object")
cleaned_complete_df["translation"] = cleaned_complete_df["translation"].astype("object")
cleaned_incomplete_df["translation"] = cleaned_incomplete_df["translation"].astype("object")

#Get the whole text from across the dataset for BPE
akkadian_only = ""
whole_text = ""
for i, row in pd.concat([cleaned_complete_df, cleaned_incomplete_df]).iterrows():
    #Seperate entries by <sos> (Start of Sequence) and <eos> (End of Sequence) tokens.
    new_row = " <sos> " + row['transliteration'] + " <eos> "
    akkadian_only += new_row
    new_row += " <sos> " + row['translation'] + " <eos> "
    whole_text += new_row

#Fit BPE onto the text.
BPE.fit(whole_text, 4000)
BPE_akk.fit(akkadian_only, 1000)

for k in BPE.vocab.keys():
    print(k, ":", BPE.vocab[k])

#Save the vocabulary and tokens
BPE.save("processed/akk2eng.json")
BPE_akk.save("processed/akkonly.json")

for i, row in cleaned_complete_df.iterrows():
    new_row = " <sos> " + row['transliteration'] + " <eos> "
    encoded = BPE.encode(new_row)
    cleaned_complete_df.at[i, 'transliteration'] = encoded

    new_row = " <sos> " + row['translation'] + " <eos> "
    encoded = BPE.encode(new_row)
    cleaned_complete_df.at[i, 'translation'] = encoded

for i, row in cleaned_incomplete_df.iterrows():
    new_row = " <sos> " + row['transliteration'] + " <eos> "
    encoded = BPE.encode(new_row)
    cleaned_incomplete_df.at[i, 'transliteration'] = encoded

    new_row = " <sos> " + row['translation'] + " <eos> "
    encoded = BPE.encode(new_row)
    cleaned_incomplete_df.at[i, 'translation'] = encoded

cleaned_complete_df.to_csv("processed/processed_train_complete.csv")
cleaned_incomplete_df.to_csv("processed/processed_train_incomplete.csv")
