import pandas as pd

creative = r"output_csv\CREATIVE PROMPTS.csv"
faq = r"output_csv\FAQ.csv"
multiple = r"output_csv\MULTIPLE.csv"
non = r"output_csv\NON.csv"
riddles = r"output_csv\RIDDLES.csv"
stem = r"output_csv\STEM.csv"

D_creative = pd.read_csv(creative, sep=',')
D_faq = pd.read_csv(faq, sep=',')
D_multiple = pd.read_csv(multiple, sep=',')
D_non = pd.read_csv(non, sep=',')
D_riddles = pd.read_csv(riddles, sep=',')
D_stem = pd.read_csv(stem, sep=',')

def dcreative():
    return D_creative
def dfaq():
    return D_faq
def dmultiple():
    return D_multiple
def dnon():
    return D_non    
def driddles():
    return D_riddles
def dstem():
    return D_stem
