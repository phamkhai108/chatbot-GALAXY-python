import spacy

def detect_city(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    cities = []
    for ent in doc.ents:
        if ent.label_ == "GPE":  
            cities.append(ent.text)
    if not cities:
        return "Không tìm thấy thành phố hoặc quốc gia nào."
    return cities
