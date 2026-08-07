#!/bin/sh
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT_DIR"
exec bash "$ROOT_DIR/run.sh"
