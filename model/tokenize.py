import utils.bpe as bpe

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

def tokenize(df):
    for key in DET_TOKENS.keys():
        df['transliteration'] = df['transliteration'].str.replace(key, DET_TOKENS[key])
        df['translation'] = df['translation'].str.replace(key, DET_TOKENS[key])

    BPE = bpe.BytePairEncoder()
    BPE.load("processed/akk2eng.json")

    tokenized_df = df.copy(True)

    tokenized_df["transliteration"] = tokenized_df["transliteration"].astype("object")
    tokenized_df["translation"] = tokenized_df["translation"].astype("object")

    for i, row in tokenized_df.iterrows():
        new_row = " <sos> " + row['transliteration'] + " <eos> "
        encoded = BPE.encode(new_row)
        tokenized_df.at[i, 'transliteration'] = encoded

        new_row = " <sos> " + row['translation'] + " <eos> "
        encoded = BPE.encode(new_row)
        tokenized_df.at[i, 'translation'] = encoded
    
    return tokenized_df