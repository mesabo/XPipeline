# 1) Test every handle as synthesizer (judge unchanged)
./xpipe/scripts/run_all_llms.sh

# 2) Only test HF handles on synth
./xpipe/scripts/run_all_llms.sh synth "hf/"

# 3) Sweep judge only
./xpipe/scripts/run_all_llms.sh judge

# 4) Full grid (may be many runs) but limit to first 8 combos
./xpipe/scripts/run_all_llms.sh grid "" "" 8

# 5) Only Ollama models as synthesizer
./xpipe/scripts/run_all_llms.sh synth "ollama/"