#!/usr/bin/env python3
import gensim
import numpy as np
## plan is to reproduce some IAT scores from Caliskan et al, Science paper

def get_association(vecs, x, A, B):
    wx = np.copy(vecs[x])
    wA = np.copy(vecs[A])
    wB = np.copy(vecs[B])
    wx /= np.linalg.norm(wx)
    wA /= np.linalg.norm(wA)
    wB /= np.linalg.norm(wB)
    return np.mean(np.dot(wx, wA.T)) - np.mean(np.dot(wx, wB.T))

def effect_size(vecs, X, Y, A, B):
    xnums = []
    for x in X:
        xnums.append(get_association(vecs, x, A, B))
    ynums = []
    for y in Y:
        ynums.append(get_association(vecs, y, A, B))
    return (np.mean(xnums) - np.mean(ynums)) / np.std(xnums+ynums)
        

## flower vs insects as pleasant vs unpleasant
flowers = ["aster", "clover", "hyacinth", "marigold", "poppy",
           "azalea", "crocus", "iris", "orchid", "rose",
           # "bluebell",
           "daffodil", "lilac", "pansy", "tulip", "buttercup",
           "daisy", "lily", "peony", "violet", "carnation",
           # "gladiola",
           "magnolia", "petunia",
           # "zinnia"
]
insects = ["ant", "caterpillar", "flea", "locust", "spider", "bedbug",
           "centipede", "fly", "maggot", "tarantula", "bee",
           "cockroach", "gnat", "mosquito", "termite", "beetle",
           "cricket", "hornet", "moth", "wasp",
           # "blackfly", "horsefly",
           "dragonfly", "roach", "weevil"]

pleasant = ["caress", "freedom", "health", "love", "peace", "cheer",
            "friend", "heaven", "loyal", "pleasure", "diamond",
            "gentle", "honest", "lucky", "rainbow", "diploma", "gift",
            "honor", "miracle", "sunrise", "family", "happy",
            "laughter", "paradise", "vacation"]

unpleasant = ["abuse", "crash", "filth", "murder", "sickness",
              "accident", "death", "grief", "poison", "stink", "assault",
              "disaster", "hatred", "pollute", "tragedy", "divorce", "jail",
              "poverty", "ugly", "cancer", "kill", "rotten", "vomit", "agony",
              "prison"]


instruments = ["bagpipe", "cello", "guitar", "lute", "trombone",
               "banjo", "clarinet", "harmonica", "mandolin",
               "trumpet", "bassoon", "drum", "harp", "oboe", "tuba",
               "bell", "fiddle", "harpsichord", "piano", "viola",
               "bongo", "flute", "horn", "saxophone", "violin"]

weapons = ["arrow", "club", "gun", "missile", "spear",
           # "axe",
           "dagger", "harpoon", "pistol", "sword", "blade", "dynamite",
           "hatchet", "rifle", "tank", "bomb", "firearm", "knife", "shotgun",
           "teargas", "cannon", "grenade", "mace", "slingshot", "whip"]

european_names = ["Adam", "Harry", "Josh", "Roger", "Alan", "Frank",
                  "Justin", "Ryan", "Andrew", "Jack", "Matthew",
                  "Stephen", "Brad", "Greg", "Paul", "Jonathan",
                  "Peter", "Amanda", "Courtney", "Heather", "Melanie",
                  "Katie", "Betsy", "Kristin", "Nancy", "Stephanie",
                  "Ellen", "Lauren", "Colleen", "Emily", "Megan",
                  "Rachel"]
african_american_names = ["Alonzo", "Jamel", "Theo", "Alphonse",
                          "Jerome", "Leroy", "Torrance", "Darnell",
                          "Lamar", "Lionel", "Tyree", "Deion",
                          "Lamont", "Malik", "Terrence", "Tyrone",
                          "Lavon", "Marcellus", "Wardell", "Nichelle",
                          "Shereen", "Ebony", "Latisha", "Shaniqua",
                          "Jasmine", "Tanisha", "Tia", "Lakisha",
                          "Latoya", "Yolanda", "Malika", "Yvette"]

european_2 = ["Brad", "Brendan", "Geoffrey", "Greg", "Brett",
              "Matthew", "Neil", "Todd", "Allison", "Anne", "Carrie",
              "Emily", "Jill", "Laurie", "Meredith", "Sarah"]
african_2 = ["Darnell", "Hakim", "Jermaine", "Kareem", "Jamal",
             "Leroy", "Rasheed", "Tyrone", "Aisha", "Ebony", "Keisha",
             "Kenya", "Lakisha", "Latoya", "Tamika", "Tanisha"]

male_names = ["John", "Paul", "Mike", "Kevin", "Steve", "Greg", "Jeff", "Bill"]
female_names = ["Amy", "Joan", "Lisa", "Sarah", "Diana", "Kate", "Ann", "Donna"]
career = ["executive", "management", "professional", "corporation",
          "salary", "office", "business", "career"]
family = ["home", "parents", "children", "family", "cousins",
          "marriage", "wedding", "relatives"]

math = ["math", "algebra", "geometry", "calculus", "equations",
        "computation", "numbers", "addition"]
poetry = ["poetry", "art", "dance", "literature", "novel", "symphony",
          "drama", "sculpture"]
male_terms = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
female_terms = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]

science = ["science", "technology", "physics", "chemistry",
           "Einstein", "NASA", "experiment", "astronomy"]
arts = ["poetry", "art", "Shakespeare", "dance", "literature",
        "novel", "symphony", "drama"]

mental_disease = ["sad", "hopeless", "gloomy", "tearful", "miserable", "depressed"]
physical_disease = ["sick", "illness", "influenza", "disease", "virus", "cancer"]
temporary = ["impermanent", "unstable", "variable", "fleeting", "short", "brief", "occasional"]
permanent = ["stable", "always", "constant", "persistent", "chronic", "prolonged", "forever"]

young_names = ["Tiffany", "Michelle", "Cindy", "Kristy", "Brad",
               "Eric", "Joey", "Billy"]
old_names = ["Ethel", "Bernice", "Gertrude", "Agnes", "Cecil",
             "Wilbert", "Mortimer", "Edgar"]
pleasant_2 = ["joy", "love", "peace", "wonderful", "pleasure", "friend", "laughter", "happy"]
unpleasant_2 = ["agony", "terrible", "horrible", "nasty", "evil", "war", "awful", "failure"]


def evaluate_embedding(vecs):
    e = effect_size(vecs, flowers, insects, pleasant, unpleasant)
    print("flowers vs insects, pleasant vs unpleasant: d = {:.2f}".format(e))
    e = effect_size(vecs, instruments, weapons, pleasant, unpleasant)
    print("instruments vs weapons, pleasant vs unpleasant: d = {:.2f}".format(e))
    e = effect_size(vecs, european_names, african_american_names, pleasant, unpleasant)
    print("european vs african american names, pleasant vs unpleasant: d = {:.2f}".format(e))
    e = effect_size(vecs, european_2, african_2, pleasant, unpleasant)
    print("european vs african american names 2, pleasant vs unpleasant: d = {:.2f}".format(e))
    e = effect_size(vecs, male_names, female_names, career, family)
    print("male vs female names, career vs family: d = {:.2f}".format(e))
    e = effect_size(vecs, math, poetry, male_terms, female_terms)
    print("math vs arts, male vs female terms: d = {:.2f}".format(e))
    e = effect_size(vecs, science, arts, male_terms, female_terms)
    print("science vs arts, male vs female terms: d = {:.2f}".format(e))
    e = effect_size(vecs, mental_disease, physical_disease, temporary, permanent)
    print("mental vs physical disease, temporary vs permanent terms: d = {:.2f}".format(e))
    e = effect_size(vecs, young_names, old_names, pleasant_2, unpleasant_2)
    print("young vs old names, pleasant vs unpleasant: d = {:.2f}".format(e))
    

if __name__ == '__main__':
    vec_path = '../data/GoogleNews-vectors-negative300.bin'
    word_vecs = gensim.models.KeyedVectors.load_word2vec_format(vec_path, binary=True,  limit=400000)
    
    evaluate_embedding(word_vecs)
