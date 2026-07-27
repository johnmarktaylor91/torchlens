#!/bin/sh
set -eu

if [ "${1:-}" = "--python" ]; then
    if [ "$#" -lt 3 ]; then
        echo "crawler_supervisor.sh: --python requires an executable and command" >&2
        exit 64
    fi
    crawler_python=$2
    shift 2
else
    crawler_python=${MENAGERIE_CRAWLER_PYTHON:-python3}
fi

exec "$crawler_python" -m menagerie.crawler.supervisor "$@"
