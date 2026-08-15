#!/usr/bin/env bash
set -euo pipefail

compiler="$1"
shift
arguments=()
for argument in "$@"; do
    case "$argument" in
        -Zcoverage-options=mcdc)
            arguments+=("-Zcoverage-options=condition")
            ;;
        coverage-options=mcdc)
            arguments+=("coverage-options=condition")
            ;;
        *)
            arguments+=("$argument")
            ;;
    esac
done
exec "$compiler" "${arguments[@]}"
