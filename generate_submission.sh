#!/usr/bin/env bash
rm -f submission.zip

if [ -f api_keys.py ]; then
    zip -r submission.zip chatbot.py rubric.txt ./deps api_keys.py
else
    zip -r submission.zip chatbot.py rubric.txt ./deps
fi