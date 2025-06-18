import pandas as pd


creative = r"output_1_5_csv\CREATIVE PROMPTS_1.5.csv"
faq = r"output_1_5_csv\FAQ_1.5.csv"
multiple = r"output_1_5_csv\MULTIPLE_1.5.csv"
non = r"output_1_5_csv\NON_1.5.csv"
riddles = r"output_1_5_csv\RIDDLES_1.5.csv"
stem = r"output_1_5_csv\STEM_1.5.csv"

D_creative15 = pd.read_csv(creative, sep=',')
D_faq15 = pd.read_csv(faq, sep=',')
D_multiple15 = pd.read_csv(multiple, sep=',')
D_non15 = pd.read_csv(non, sep=',')
D_riddles15 = pd.read_csv(riddles, sep=',')
D_stem15 = pd.read_csv(stem, sep=',')

def dcreative15():
    return D_creative15
def dfaq15():
    return D_faq15
def dmultiple15():
    return D_multiple15
def dnon15():
    return D_non15
def driddles15():
    return D_riddles15  
def dstem15():
    return D_stem15

