#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 && $# != 4 )); then
    printf 'Usage: bash %s READS.tsv [CONTIGS.tsv] COASSEMBLY.tsv OUTPUT_DIR\n' "$0" >&2
    printf 'Omit CONTIGS.tsv to auto-assemble ungrouped samples from reads.\n' >&2
    exit 2
fi

contig_args=()
if (( $# == 4 )); then
    contig_args=(--input_contig_files "$2")
    coassembly_file="$3"
    output_dir="$4"
else
    coassembly_file="$2"
    output_dir="$3"
fi

exec metabaw bin \
    --input_reads_files "$1" \
    "${contig_args[@]}" \
    --assembly-strategy coassembly \
    --coassembly-file "$coassembly_file" \
    --tools metabat2 metadecoder vamb comebin semibin2 \
    --refinement magscot --quality-control checkm2 \
    --gunc --trna --rrna \
    --dereplication-tool galah --ani 99 --min-aligned-fraction 30 \
    --con 10 --com 50 --min-contig-length 1500 --minfasta-kbs 200 \
    --no-gpu -t 32 --task 1 --max-memory 200 \
    -o "$output_dir"
