# Automatic recovery demo

A deterministic scripted provider drives the real agent tool loop through a syntax failure, a runtime failure, and a successful repair, with zero manual interventions. Everything except the model's choices is the same code a scored run uses: the syntax gate, tool dispatch, subprocess execution, repair prompts, and log-row validation.

Run `./mle_agent/scripts/demo_recovery.sh` to regenerate the evidence.
