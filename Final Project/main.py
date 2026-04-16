import os
from config import UNIPROT_API_BASE, ALPHAFOLD_API_BASE, RAW_DIR, PROCESSED_DIR, STRUCTURES_DIR, HOMOLOGY_THRESHOLD
from uniprot_pipeline import UniProtPipeline
from text_processing import TextProcessor
from structure_pipeline import StructurePipeline
from clustering_and_splits import SequenceClustering

def main():
    uniprot = UniProtPipeline(UNIPROT_API_BASE)
    raw_entries = uniprot.fetch_swissprot_entries(limit=200)
    df = uniprot.extract_features(raw_entries)
    
    text_processor = TextProcessor(df)
    df['clean_text'] = df.apply(lambda row: text_processor.scrub_data_leakage(row['raw_text'], row['go_terms']), axis=1)
    text_processor.implement_fallback_logic()
    df = text_processor.df
    
    structure_pipe = StructurePipeline(ALPHAFOLD_API_BASE, STRUCTURES_DIR)
    valid_indices = []
    
    for index, row in df.iterrows():
        pdb_file = structure_pipe.fetch_alphafold_structure(row['uniprot_id'])
        is_valid = structure_pipe.validate_sequence(row['sequence'], pdb_file)
        if is_valid:
            valid_indices.append(index)
        else:
            if pdb_file and os.path.exists(pdb_file):
                os.remove(pdb_file)
                
    df_validated = df.loc[valid_indices]
    
    if df_validated.empty:
        print("No sequences passed the structural validation pass.")
        return
        
    fasta_path = os.path.join(RAW_DIR, "sequences.fasta")
    cluster = SequenceClustering(df_validated, threshold=HOMOLOGY_THRESHOLD)
    cluster.export_fasta(fasta_path)
    
    # This parses the .clstr file and saves train_dataset.csv, val_dataset.csv, and test_dataset.csv
    cluster.generate_remote_homology_splits(fasta_path, PROCESSED_DIR)

if __name__ == "__main__":
    main()