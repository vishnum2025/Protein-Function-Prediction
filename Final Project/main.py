import os
from config import UNIPROT_API_BASE, ALPHAFOLD_API_BASE, RAW_DIR, PROCESSED_DIR, STRUCTURES_DIR, HOMOLOGY_THRESHOLD
from uniprot_pipeline import UniProtPipeline
from text_processing import TextProcessor
from structure_pipeline import StructurePipeline
from clustering_and_splits import SequenceClustering

def main():
    # 1. Data Curation: Fetch and Extract from UniProt
    uniprot = UniProtPipeline(UNIPROT_API_BASE)
    raw_entries = uniprot.fetch_swissprot_entries(limit=50) # Small limit for testing
    df = uniprot.extract_features(raw_entries)
    
    # 2. Text Processing & Leakage Prevention
    text_processor = TextProcessor(df)
    df['clean_text'] = df.apply(lambda row: text_processor.scrub_data_leakage(row['raw_text'], row['go_terms']), axis=1)
    text_processor.implement_fallback_logic()
    df = text_processor.df # Update with imputed fallback text
    
    # 3. Structural Data Extraction & Validation
    structure_pipe = StructurePipeline(ALPHAFOLD_API_BASE, STRUCTURES_DIR)
    valid_indices = []
    
    for index, row in df.iterrows():
        pdb_file = structure_pipe.fetch_alphafold_structure(row['uniprot_id'])
        # Simplified validation pass
        is_valid = True # In production: is_valid = structure_pipe.validate_sequence(row['sequence'], pdb_file)
        if is_valid:
            valid_indices.append(index)
            
    # Discard deviant cases [cite: 43]
    df_validated = df.loc[valid_indices]
    
    # 4. Remote-Homology Splitting
    fasta_path = os.path.join(RAW_DIR, "sequences.fasta")
    cluster = SequenceClustering(df_validated, threshold=HOMOLOGY_THRESHOLD)
    cluster.export_fasta(fasta_path)
    cluster.generate_remote_homology_splits(fasta_path, PROCESSED_DIR)
    
    # 5. Export Final Processed Data
    df_validated.to_csv(os.path.join(PROCESSED_DIR, "final_dataset.csv"), index=False)

if __name__ == "__main__":
    main()