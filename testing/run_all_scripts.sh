#!/bin/bash

# Generate Transcripts for Submission
# Sam Redmond, 2019-03-09
#
# IMPORTANT NOTES:
#   The script must be run from the //Product/ directory.
#
# NOTES:
#   Why do we bother restarting Unity at all? Once Unity loads a plugin, it
#   holds that plugin's contents in memory until Unity itself is exited. Thus,
#   if the actual plugin file (*.bundle) changes on the filesystem, Unity will
#   not acknowledge this change until it is restarted.
#
# PARAMS:
#   $1: The name of the Unity project into which to copy the built plugin.
#   $BUILD_TARGET: The build target for buck to make.
#     Default: //UnityNativePlugin:UnityNativePluginMac#shared,macosx-x86_64
#   $BUILD_OUTPUT: Relative path to the built plugin.
#     Default: buck-out/gen/UnityNativePlugin/UnityNativePluginMac#macosx-x86_64,shared/UnityNativePlugin.bundle
#   $PROJECT_DIRECTORY: Directory containing Unity projects (relative to //Product/ folder).
#     Default: ../UnityDemos
#   $ASSETS_PATH: Path that contains Hedron assets (relative to project root).
#     Default: Assets/Hedron
#   $UNITY_EXECUTABLE: Path to Unity executable.
#     Default: /Applications/Unity/Unity.app/Contents/MacOS/Unity
#
# USAGE:
#   ./rebuild-unity-plugin.sh TrashDash

###########################
# CONFIGURABLE PARAMETERS #
###########################

# The build target for buck to make.
readonly BUILD_TARGET="${BUILD_TARGET:-//UnityNativePlugin:UnityNativePluginMac#shared,macosx-x86_64}"

# Relative path to the built plugin. This should be the location at which buck
# places the output. This should almost certainly start with `buck-out/gen/...`.
readonly BUILD_OUTPUT="${BUILD_OUTPUT:-buck-out/gen/UnityNativePlugin/UnityNativePluginMac#macosx-x86_64,shared/UnityNativePlugin.bundle}"

# Directory containing Unity projects (relative to //Product/ folder).
readonly PROJECT_DIRECTORY="${PROJECT_DIRECTORY:-../UnityDemos}"

# Within a project, path that contains Hedron assets (relative to project root).
readonly ASSETS_PATH="${ASSETS_PATH:-Assets/Hedron}"

# Path to Unity executable.
readonly UNITY_EXECUTABLE="${UNITY_EXECUTABLE:-/Applications/Unity/Unity.app/Contents/MacOS/Unity}"

###############################
# END CONFIGURABLE PARAMETERS #
###############################

################################
# Echo arguments to stderr, prefixed with a timestamp.
#
# Globals: None
# Arguments:
#   $@ - echoed to stderr.
# Returns: None
################################
err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $@" >&2
}

main() {
    echo "Generating transcripts for chatbot..."

    local TRANSCRIPTS_DIR="outputs-for-scripts"
    mkdir -p "${TRANSCRIPTS_DIR}/standard"
    mkdir -p "${TRANSCRIPTS_DIR}/llm_prompting"
    mkdir -p "${TRANSCRIPTS_DIR}/llm_programming"

    for file in testing/test_scripts/standard/*; do
        local TEST_ID="$(basename ${file} .txt)"
        local TRANSCRIPT_PATH="${TRANSCRIPTS_DIR}/standard/"
        local TRANSCRIPT_FILE="${TRANSCRIPT_PATH}${TEST_ID}.transcript"

        echo "(Standard) Generating transcript for ${TEST_ID} in ${TRANSCRIPT_PATH}."

        python3 repl.py < "${file}" &> "${TRANSCRIPT_FILE}"
    done

    for file in testing/test_scripts/llm_prompting/*; do
        local TEST_ID="$(basename ${file} .txt)"
        local TRANSCRIPT_PATH="${TRANSCRIPTS_DIR}/llm_prompting/"
        local TRANSCRIPT_FILE="${TRANSCRIPT_PATH}${TEST_ID}.transcript"

        echo "(LLM Prompting) Generating transcript for ${TEST_ID} in ${TRANSCRIPT_PATH}."

        python3 repl.py --llm_prompting < "${file}" &> "${TRANSCRIPT_FILE}"
    done

    for file in testing/test_scripts/llm_programming/*; do
        local TEST_ID="$(basename ${file} .txt)"
        local TRANSCRIPT_PATH="${TRANSCRIPTS_DIR}/llm_programming/"
        local TRANSCRIPT_FILE="${TRANSCRIPT_PATH}${TEST_ID}.transcript"

        echo "(LLM Programming) Generating transcript for ${TEST_ID} in ${TRANSCRIPT_PATH}."
        python3 repl.py --llm_programming < "${file}" &> "${TRANSCRIPT_FILE}"
    done
}

main "$@"
