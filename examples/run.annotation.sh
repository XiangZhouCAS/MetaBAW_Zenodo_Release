#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
    printf 'Usage: bash %s READS.tsv GENOMES.txt OUTPUT_DIR [annotation options...]\n' "$0" >&2
    printf 'Niche runs only with >=4 genomes AND >=4 read samples. Paths are relative to the list file.\n' >&2
    exit 2
fi

reads_manifest=$1
genome_manifest=$2
annotation_output=$3
shift 3

exec metabaw annotation \
    --input_reads_files "$reads_manifest" \
    --input_genome_files "$genome_manifest" \
    -t 32 --task 1 --max-memory 200 \
    -o "$annotation_output" "$@"
