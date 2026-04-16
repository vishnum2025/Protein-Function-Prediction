import pandas as pd
import re
from typing import List
from Bio import pairwise2

class TextProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def scrub_data_leakage(self, text: str, go_terms: List[str]) -> str:
        """Removes explicit GO terms and EC numbers to prevent data leakage."""
        if pd.isna(text) or not text:
            return ""
        
        scrubbed_text = text
        # Remove direct GO term IDs
        for go_term in go_terms:
            scrubbed_text = scrubbed_text.replace(go_term, "[MASKED_GO]")
            
        # Regex to mask EC numbers (e.g., EC 1.2.3.4)
        scrubbed_text = re.sub(r'EC\s\d+\.\d+\.\d+\.\d+', '[MASKED_EC]', scrubbed_text)
        return scrubbed_text

    def implement_fallback_logic(self):
        """Imputes missing text annotations using the nearest sequence neighbor."""
        missing_text_mask = self.df['raw_text'] == ""
        df_missing = self.df[missing_text_mask]
        df_valid = self.df[~missing_text_mask]

        for index, row in df_missing.iterrows():
            best_score = -1
            best_text = ""
            
            for _, valid_row in df_valid.iterrows():
                alignments = pairwise2.align.globalxx(row['sequence'], valid_row['sequence'])
                score = alignments[0].score if alignments else 0
                if score > best_score:
                    best_score = score
                    best_text = valid_row['raw_text']
            
            self.df.at[index, 'clean_text'] = best_text