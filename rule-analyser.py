import re
import pandas as pd


def contains_only_long_words(item_set_str):
    """
    Tjekker om en streng af elementer kun indeholder ord, der er længere end 3 bogstaver.
    'HAS_ENGLISH_WORD_IN_ANSWER' betragtes som et 'langt' ord.
    Fjerner 'P_' præfikset før længdekontrol for P-ord.
    """
    if not item_set_str:
        return False  # En tom streng indeholder ikke 'lange' ord

    # Fjern krøllede parenteser og opdel i individuelle elementer
    items = item_set_str.replace('{', '').replace('}', '').split(', ')

    for item in items:
        # Håndter 'HAS_ENGLISH_WORD_IN_ANSWER' som et undtagelse
        if item == 'HAS_ENGLISH_WORD_IN_ANSWER':
            continue

        # Fjern 'P_' præfikset før længdekontrol, hvis det findes
        word_to_check = item
        if word_to_check.startswith('P_'):
            word_to_check = word_to_check[2:]  # Fjern 'P_'

        # Tjek længden af ordet
        if len(word_to_check) <= 3:
            return False  # Hvis et ord er 3 bogstaver eller kortere, returneres False

    return True  # Alle ord er lange nok (eller undtagelser)


def analyze_association_rules_filtered_by_word_length(file_path):
    """
    Analyserer en tekstfil med associationsregler for at finde de mest relevante.
    Kun inkluderer regler, hvor alle ord i antecedent og consequent er længere end 3 bogstaver.

    Args:
        file_path (str): Stien til tekstfilen, der indeholder associationsreglerne.

    Returns:
        pandas.DataFrame: Et DataFrame indeholdende de sorterede og filtrerede regler,
                          eller None hvis filen ikke blev fundet eller der opstod en fejl.
    """
    rules_data = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('Rule:'):
                    # Ekstraher hele reglen (antecedent => consequent)
                    rule_parts_match = re.match(r'Rule: (.*?) => (.*?) \(Support:', line)
                    if rule_parts_match:
                        antecedent_raw = rule_parts_match.group(1).strip()
                        consequent_raw = rule_parts_match.group(2).strip()
                    else:
                        continue  # Spring over linjer, der ikke matcher forventet format

                    # Tjek om både antecedent og consequent KUN indeholder ord > 3 bogstaver
                    if not contains_only_long_words(antecedent_raw) or \
                            not contains_only_long_words(consequent_raw):
                        continue  # Spring reglen over, hvis den ikke opfylder kravet

                    # Ekstraher metrics ved hjælp af regulære udtryk
                    support_match = re.search(r'Support: ([\d.]+)', line)
                    confidence_match = re.search(r'Confidence: ([\d.]+)', line)
                    lift_match = re.search(r'Lift: ([\d.]+)', line)
                    representativity_match = re.search(r'Representativity: ([\d.]+)', line)
                    leverage_match = re.search(r'Leverage: ([\d.-]+)', line)
                    conviction_match = re.search(r'Conviction: ([\d.]+)', line)
                    zhangs_metric_match = re.search(r'Zhangs_Metric: ([\d.-]+)', line)
                    jaccard_match = re.search(r'Jaccard: ([\d.]+)', line)
                    certainty_match = re.search(r'Certainty: ([\d.]+)', line)
                    kulczynski_match = re.search(r'Kulczynski: ([\d.]+)', line)

                    support = float(support_match.group(1)) if support_match else 0.0
                    confidence = float(confidence_match.group(1)) if confidence_match else 0.0
                    lift = float(lift_match.group(1)) if lift_match else 0.0
                    representativity = float(representativity_match.group(1)) if representativity_match else 0.0
                    leverage = float(leverage_match.group(1)) if leverage_match else 0.0
                    conviction = float(conviction_match.group(1)) if conviction_match else 0.0
                    zhangs_metric = float(zhangs_metric_match.group(1)) if zhangs_metric_match else 0.0
                    jaccard = float(jaccard_match.group(1)) if jaccard_match else 0.0
                    certainty = float(certainty_match.group(1)) if certainty_match else 0.0
                    kulczynski = float(kulczynski_match.group(1)) if kulczynski_match else 0.0

                    rules_data.append({
                        'Rule': f"{antecedent_raw} => {consequent_raw}",  # Brug den originale regel
                        'Support': support,
                        'Confidence': confidence,
                        'Lift': lift,
                        'Representativity': representativity,
                        'Leverage': leverage,
                        'Conviction': conviction,
                        'Zhangs_Metric': zhangs_metric,
                        'Jaccard': jaccard,
                        'Certainty': certainty,
                        'Kulczynski': kulczynski
                    })

        df_rules = pd.DataFrame(rules_data)

        # Sorter reglerne for at finde de mest relevante.
        df_rules_sorted = df_rules.sort_values(by=['Lift', 'Confidence'], ascending=[False, False])

        print("\n--- Analyse af Associationsregler (Kun regler med ord > 3 bogstaver) ---")
        print(f"Total antal regler fundet (efter ordlængde filtrering): {len(df_rules_sorted)}")

        # Vis de top 10 mest relevante regler
        print("\nTop 10 mest relevante regler (sorteret efter Lift og Confidence - kun lange ord):")
        print(df_rules_sorted.head(10).to_string())

        # Gem de sorterede regler til en CSV-fil for yderligere analyse
        output_csv_file = 'sorted_association_rules_long_words_only.csv'
        df_rules_sorted.to_csv(output_csv_file, index=False)
        print(f"\nAlle sorterede og filtrerede regler er gemt i '{output_csv_file}'")

        return df_rules_sorted

    except FileNotFoundError:
        print(
            f"Fejl: Filen '{file_path}' blev ikke fundet. Sørg for, at filen ligger i samme mappe som scriptet, eller angiv den fulde sti.")
        return None
    except Exception as e:
        print(f"Der opstod en fejl under behandlingen af filen: {e}")
        return None


# --- Brug af scriptet ---
# Angiv stien til din tekstfil her
file_to_analyze = 'rules_len3_sup0.10_conf0.50_20250617_191711.txt'  # Sørg for at denne fil eksisterer i samme mappe
analyzed_rules = analyze_association_rules_filtered_by_word_length(file_to_analyze)

# Hvis du vil arbejde videre med de analyserede regler i Python:
if analyzed_rules is not None:
    # print(analyzed_rules.head())
    pass