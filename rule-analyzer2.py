import re
import pandas as pd


def contains_only_long_words(item_set_str):
    """
    Checks if a string of items only contains words that are STRICTLY longer than 3 characters.
    'HAS_ENGLISH_WORD_IN_ANSWER' is an exception and considered 'long'.
    The 'P_' prefix is removed before the length check.
    """
    if not item_set_str:
        return False

    # Fjern krøllede parenteser og opdel strengen i individuelle elementer
    items = item_set_str.replace('{', '').replace('}', '').split(', ')

    for item in items:
        # Specialhåndtering for "HAS_ENGLISH_WORD_IN_ANSWER", der altid tæller som et 'langt' ord
        if item == 'HAS_ENGLISH_WORD_IN_ANSWER':
            continue

        # Fjern 'P_' præfikset, hvis det findes, før længden tjekkes
        word_to_check = item
        if word_to_check.startswith('P_'):
            word_to_check = word_to_check[2:]

        # Tjek om ordet er 3 bogstaver eller kortere
        if len(word_to_check) <= 3:
            return False  # Returner False så snart et kort ord findes
    return True  # Hvis ingen korte ord blev fundet, returner True


def analyze_association_rules_filtered_by_word_length(file_path):
    """
    Analyzes a text file with association rules to find the most relevant ones.
    Only includes rules where all words in the antecedent and consequent are strictly longer than 3 characters,
    AND all metrics are successfully parsed.
    """
    rules_data = []
    line_number = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_number += 1

                # Spring tomme linjer eller headere over
                if not line.strip() or line.strip().startswith('---') or line.strip().startswith('Configuration:'):
                    continue

                if line.strip().startswith('Rule:'):
                    rule_parts_match = re.match(r'Rule: (.*?) => (.*?) \(Support:', line)
                    if rule_parts_match:
                        antecedent_raw = rule_parts_match.group(1).strip()
                        consequent_raw = rule_parts_match.group(2).strip()
                    else:
                        print(
                            f"Advarsel: Linje {line_number} matchede ikke 'Rule:' mønster. Springer over: {line.strip()}")
                        continue  # Springer linjen over, hvis den ikke matcher forventet regelformat

                    # Første filtrering: Ignorer linjen hvis ordene er 3 bogstaver eller kortere
                    if not contains_only_long_words(antecedent_raw) or \
                            not contains_only_long_words(consequent_raw):
                        print(f"Info: Linje {line_number} filtreret fra pga. ordlængde (<= 3 tegn): {line.strip()}")
                        continue  # Springer linjen over

                    metrics = {}
                    metric_patterns = {
                        'Support': r'Support: ([\d.]+)',
                        'Confidence': r'Confidence: ([\d.]+)',
                        'Lift': r'Lift: ([\d.]+)',
                        'Representativity': r'Representativity: ([\d.]+)',
                        'Leverage': r'Leverage: ([\d.-]+)',
                        'Conviction': r'Conviction: ([\d.]+)',
                        'Zhangs_Metric': r'Zhangs_Metric: ([\d.-]+)',
                        'Jaccard': r'Jaccard: ([\d.]+)',
                        'Certainty': r'Certainty: ([\d.]+)',
                        'Kulczynski': r'Kulczynski: ([\d.]+)'
                    }

                    # Flag til at spore om alle metrikker blev fundet og parset
                    all_metrics_parsed_successfully = True

                    for metric_name, pattern in metric_patterns.items():
                        match = re.search(pattern, line)
                        if match:
                            try:
                                metrics[metric_name] = float(match.group(1))
                            except ValueError:
                                print(
                                    f"FEJL HÅNDTERET: Linje {line_number} - Kunne ikke konvertere '{metric_name}' værdi til float. Rå værdi: '{match.group(1)}'. Linje: {line.strip()}")
                                all_metrics_parsed_successfully = False
                                break  # Stop med at behandle denne linje, hvis en konvertering fejler
                        else:
                            print(
                                f"FEJL HÅNDTERET: Linje {line_number} - Kunne ikke finde '{metric_name}' i linjen. Linjen er muligvis ufuldstændig. Linje: {line.strip()}")
                            all_metrics_parsed_successfully = False
                            break  # Stop med at behandle denne linje, hvis en metrik mangler

                    # Tilføj kun reglen, hvis alle metrikker blev fundet og parset korrekt
                    if all_metrics_parsed_successfully:
                        rules_data.append({
                            'Rule': f"{antecedent_raw} => {consequent_raw}",
                            'Support': metrics['Support'],
                            'Confidence': metrics['Confidence'],
                            'Lift': metrics['Lift'],
                            'Representativity': metrics['Representativity'],
                            'Leverage': metrics['Leverage'],
                            'Conviction': metrics['Conviction'],
                            'Zhangs_Metric': metrics['Zhangs_Metric'],
                            'Jaccard': metrics['Jaccard'],
                            'Certainty': metrics['Certainty'],
                            'Kulczynski': metrics['Kulczynski']
                        })
                    else:
                        print(
                            f"Info: Linje {line_number} blev sprunget over, fordi ikke alle metrikker blev fundet eller kunne parses korrekt: {line.strip()}")

        df_rules = pd.DataFrame(rules_data)

        if df_rules.empty:
            print(
                "\nIngen regler fundet efter filtrering for ordlængde og fuldstændige metrikker. Den resulterende DataFrame er tom.")
            return None

        # Sorter data efter 'Lift' og derefter 'Confidence' i faldende rækkefølge
        df_rules_sorted = df_rules.sort_values(by=['Lift', 'Confidence'], ascending=[False, False])

        print("\n--- Analyse af Associationsregler (Kun regler med ord STRENGT længere end 3 bogstaver) ---")
        print(f"Total antal regler fundet (efter ordlængde og metrik-filtrering): {len(df_rules_sorted)}")
        print("\nTop 10 mest relevante regler (sorteret efter Lift og Confidence - kun ord > 3 bogstaver):")
        print(df_rules_sorted.head(10).to_string())

        output_csv_file = 'sorted_association_rules_strict_long_words_only.csv'
        df_rules_sorted.to_csv(output_csv_file, index=False)
        print(f"\nAlle sorterede og strengt filtrerede regler er gemt i '{output_csv_file}'")

        return df_rules_sorted

    except FileNotFoundError:
        print(
            f"Fejl: Filen '{file_path}' blev ikke fundet. Sørg for, at filen ligger i samme mappe som scriptet, eller angiv den fulde sti.")
        return None
    except Exception as e:
        print(f"*** DER OPSTOD EN UVENTET TOP-NIVEAU FEJL UNDER BEHANDLINGEN AF FILEN: {e} ***")
        import traceback
        traceback.print_exc()  # Print fuld traceback for at se præcis, hvor fejlen opstår
        return None


# --- Brug af scriptet ---
# Husk at opdatere stien til din fil, hvis den ikke ligger i samme mappe som scriptet
file_to_analyze = 'C:/Users/Matti/Desktop/Code Projects/Clion/Project-statistical-eval/rules_len3_sup0.10_conf0.50_20250617_191711.txt'
analyzed_rules = analyze_association_rules_filtered_by_word_length(file_to_analyze)

if analyzed_rules is not None:
    print("\nAnalyse afsluttet.")
else:
    print("\nAnalyse mislykkedes eller ingen regler fundet.")