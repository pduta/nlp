import spacy 
 
# Load the pre-trained NLP model 
nlp = spacy.load("en_core_web_sm") 
 
# Collection of news headlines 
headlines = [ 
    "Apple is looking at buying U.K. startup for $1 billion", 
    "Tesla Inc is going to acquire Twitter for $45 billion", 
    "NASA's Artemis mission aims to land the first woman on the Moon", 
    "Elon Musk met with world leaders at the summit in Davos" 
] 
 
for headline in headlines: 
    print(f"\nHeadline: {headline}") 
    doc = nlp(headline) 
     
    # Identify and print named entities 
    if not doc.ents: 
        print("No entities found.") 
    for ent in doc.ents: 
print(f" - {ent.text} ({ent.label_})")
