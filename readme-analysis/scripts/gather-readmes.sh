#!/usr/bin/env bash

DATA_DIR="data"
OUTPUT_FILE="readmes.csv"

add_entry() {
    repo_path="$1"
    repo="$(basename "$repo_path")"
    mapfile -t readme_files < <(cd "$repo_path" || return 1; fd -tf -d1 -i readme)
    if [ -z "${readme_files[*]}" ]; then
        return 1;
    fi
    echo "$repo," >> "$OUTPUT_FILE"
    for readme in "${readme_files[@]}"; do
        sed -e 's/,//g' "${repo_path}/${readme}" >> "$OUTPUT_FILE"
        if [ $? != 0 ]; then
            echo "Couldn't read ${repo_path}/${readme}"
            continue
        fi
        echo -e '\n' >> "$OUTPUT_FILE"
    done
}

main() {
    cd "$DATA_DIR" || exit 1
    test -d repos || exit 1

    if ! test -f "$OUTPUT_FILE"; then
        OVERWRITE=true
    fi

    if [ "$OVERWRITE" = true ]; then
        echo 'project,readmes_contents' > "$OUTPUT_FILE"
    fi

    for repo_path in repos/*; do
        repo="$(basename "$repo_path")"
        if grep -qi "$repo" <(cut -d, -f1 "$OUTPUT_FILE"); then
            if [ "$OVERWRITE" = true ]; then
                sed -i "/^$repo/d" "$OUTPUT_FILE"
            else
                continue
            fi
        fi
        add_entry "$repo_path"
    done
}

OVERWRITE=false

while [ $# -gt 0 ]; do
    case "$1" in
        --force|-f)
            OVERWRITE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

main
