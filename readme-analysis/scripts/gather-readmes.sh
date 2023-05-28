#!/usr/bin/env bash

DATA_DIR="data"
OUTPUT_FILE="readmes.csv"

add_entry() {
    repo_path="$1"
    repo="$(basename "$repo_path")"
    mapfile -t readme_files < <(cd "$repo_path" || return 1; fd -tf -d1 -i readme)
    if [ -z "$readme_files" ]; then
        return 1;
    fi
    echo "$repo," >> "$OUTPUT_FILE"
    for readme in "$readme_files"; do
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
    check_file="${1:-names.raw}"
    OUTPUT_FILE="${2:-$OUTPUT_FILE}"
    test -f "$check_file" || touch "$check_file"

    echo 'project,readmes_contents' > "$OUTPUT_FILE"

    for repo_path in repos/*; do
        repo="$(basename "$repo_path")"
        grep -qi "$repo" "$check_file"
        if [ $? != 0 ]; then
            add_entry "$repo_path"
            echo "$repo" >> "$check_file"
        fi
    done
}

main "$@"
