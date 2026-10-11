#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 4 )); then
    printf 'Usage: bash %s READS.tsv CONTIGS.tsv|- OUTPUT_DIR [MULTI.txt]\n' "$0" >&2
    exit 2
fi

contig_args=()
if [[ "$2" != "-" ]]; then
    contig_args=(--input_contig_files "$2")
fi

group_args=()
if (( $# == 4 )); then
    group_args=(--multi-files "$4")
fi

exec metabaw bin \
    --input_reads_files "$1" \
    "${contig_args[@]}" \
    "${group_args[@]}" \
    --tools metabat2 metadecoder vamb comebin semibin2 lorbin \
    --refinement magscot --quality-control checkm2 \
    --gunc --trna --rrna \
    --dereplication-tool galah --ani 99 --min-aligned-fraction 30 \
    --con 10 --com 50 --min-contig-length 1500 --minfasta-kbs 200 \
    --no-gpu -t 32 --task 1 --max-memory 200 \
    -o "$3"
