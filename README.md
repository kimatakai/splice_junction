#README

## Molecular Biology (Splice-junction Gene Sequences) dataset
You can get this dataset from the URL (https://archive.ics.uci.edu/dataset/69/molecular+biology+splice+junction+gene+sequences)

Dataset to classify if there is a fixed length sequence containing exons and introns and if the boundary lies between them.

## process

1. get splice-junction dataset

2. data processing

3. get blast dataset

https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/RefSeqGene/

4. get blastn output
blastn -query splice_junction.fasta -db refseqgene.fna -out blastn_splice_junction_fmt6.txt -num_threads 16 -word_size 4 -outfmt "6 qseqid sseqid qseq sseq qstart qend sstart send mismatch length"
-> blastn_splice_junction_fmt6.tsv
blastn -query splice_junction.fasta -db refseqgene.fna -out blastn_splice_junction_fmt10.txt -num_threads 16 -word_size 4 -outfmt 10
-> blastn_splice_junction_fmt10.tsv







