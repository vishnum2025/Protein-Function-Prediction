import requests
import pandas as pd
from typing import List, Dict

class UniProtPipeline:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def fetch_swissprot_entries(self, query: str = "reviewed:true", limit: int = 100) -> List[Dict]:
        """Fetches manually curated Swiss-Prot entries[cite: 39]."""
        url = f"{self.base_url}/search?query={query}&format=json&size={limit}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get('results', [])

    def extract_features(self, entries: List[Dict]) -> pd.DataFrame:
        """Extracts UniProt ID, sequence, text annotations, and GO terms."""
        data = []
        for entry in entries:
            uniprot_id = entry.get('primaryAccession')
            sequence = entry.get('sequence', {}).get('value')
            
            # Extract Text Annotations (Functions)
            annotations = []
            for comment in entry.get('comments', []):
                if comment.get('commentType') == 'FUNCTION':
                    for text_node in comment.get('texts', []):
                        annotations.append(text_node.get('value'))
            text_annotation = " ".join(annotations)

            # Extract GO Terms
            go_terms = []
            for db_ref in entry.get('uniProtKBCrossReferences', []):
                if db_ref.get('database') == 'GO':
                    go_terms.append(db_ref.get('id'))

            data.append({
                'uniprot_id': uniprot_id,
                'sequence': sequence,
                'raw_text': text_annotation,
                'go_terms': go_terms
            })
        return pd.DataFrame(data)